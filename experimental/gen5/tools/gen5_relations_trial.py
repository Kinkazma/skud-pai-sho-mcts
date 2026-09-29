"""Isolated shared edge encoder + policy/Q residual + tactical auxiliary heads.

Consumes native rule-derived features. Saves diagnostic sidecars only, which are
not MicroArtifact checkpoints and cannot be promoted by the campaign loader.
Current-position edge facts are inputs; successor facts are supervision only.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


class Network:
    def __init__(self, seed, relations=True, auxiliary=0.0):
        rng = np.random.default_rng(seed)
        self.relations = relations
        self.auxiliary = auxiliary
        self.w = {
            "edge": rng.normal(0, (2 / 59) ** .5, (43, 16)),
            "edge_b": np.zeros(16),
            "hidden": rng.normal(0, (2 / 549) ** .5, (485, 64)),
            "hidden_b": np.zeros(64),
            "out": rng.normal(0, .01, (64, 14)),
            "out_b": np.zeros(14),
        }
        self.w["out"][:, :2] = 0
        self.m = {k: np.zeros_like(v) for k, v in self.w.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.w.items()}
        self.steps = 0

    def forward(self, row):
        edges = row["tokens"]
        count = len(row["prior"])
        encoded = np.tanh(edges @ self.w["edge"] + self.w["edge_b"])
        pooled = encoded.mean(axis=1) if edges.shape[1] else np.zeros((count, 16))
        if not self.relations:
            pooled = np.zeros_like(pooled)
        global_features = row["global"] if self.relations else np.zeros(20)
        x = np.concatenate((np.broadcast_to(row["state"], (count, 417)), row["actions"],
                            np.broadcast_to(global_features, (count, 20)), pooled), axis=1)
        hidden = np.tanh(x @ self.w["hidden"] + self.w["hidden_b"])
        y = hidden @ self.w["out"] + self.w["out_b"]
        logits = np.log(np.maximum(row["prior"], 1e-300)) + y[:, 0]
        exp = np.exp(logits - logits.max())
        policy = exp / exp.sum()
        if np.all(y[:, 0] == 0):
            policy = row["prior"].copy()
        q = np.tanh(row["q_base_logit"] + y[:, 1])
        q = np.where(y[:, 1] == 0, row["next_values"], q)
        # Exact terminal Q is never replaced by a learned residual.
        q = np.where(row["terminal_mask"], row["next_values"], q)
        return policy, q, y[:, 2:], (x, hidden, encoded, y)

    def loss_gradient(self, row, scale, gradient=True):
        p, q, aux, (x, hidden, encoded, y) = self.forward(row)
        target = row["policy_target"]
        dy = np.zeros_like(y)
        if row["proven"] == 1:
            support = row["winning"]
            mass = p[support].sum()
            loss = -np.log(max(mass, 1e-300))
            dy[:, 0] = p
            dy[support, 0] -= p[support] / mass
        else:
            loss = -(target * np.log(np.maximum(p, 1e-300))).sum()
            dy[:, 0] = p - target
        known = row["q_mask"] & ~row["terminal_mask"]
        qs = .25 / max(1, known.sum())
        error = np.where(known, q - row["q_targets"], 0)
        loss += .5 * qs * (error ** 2).sum()
        dy[:, 1] = qs * error * (1 - q ** 2)
        residual = (aux - row["auxiliary"]) / scale
        loss += .5 * self.auxiliary * (residual ** 2).mean()
        dy[:, 2:] = self.auxiliary * residual / scale / residual.size
        if not gradient:
            return float(loss)
        g = {k: np.zeros_like(v) for k, v in self.w.items()}
        g["out"] = hidden.T @ dy
        g["out_b"] = dy.sum(axis=0)
        dh = (dy @ self.w["out"].T) * (1 - hidden ** 2)
        g["hidden"] = x.T @ dh
        g["hidden_b"] = dh.sum(axis=0)
        if self.relations and encoded.shape[1]:
            dp = dh @ self.w["hidden"][-16:].T
            de = dp[:, None, :] * (1 - encoded ** 2) / encoded.shape[1]
            g["edge"] = row["tokens"].reshape(-1, 43).T @ de.reshape(-1, 16)
            g["edge_b"] = de.sum(axis=(0, 1))
        return float(loss), g

    def update(self, batch, scale, rate=.001):
        grad = {k: np.zeros_like(v) for k, v in self.w.items()}
        for row in batch:
            _, g = self.loss_gradient(row, scale)
            for k in grad:
                grad[k] += g[k] / len(batch)
        norm = np.sqrt(sum(float((g*g).sum()) for g in grad.values()))
        factor = min(1., 5. / max(norm, 1e-30))
        self.steps += 1
        for k, g in grad.items():
            g = g * factor
            self.m[k] = .9 * self.m[k] + .1 * g
            self.v[k] = .999 * self.v[k] + .001 * g*g
            self.w[k] -= rate * (self.m[k] / (1-.9**self.steps)) / (np.sqrt(self.v[k] / (1-.999**self.steps)) + 1e-8)

    def save(self, path):
        values = {**self.w, **{"adam_m_"+k: v for k,v in self.m.items()},
                  **{"adam_v_"+k: v for k,v in self.v.items()}, "steps": self.steps}
        np.savez(path, **values)


def prepare(row):
    r = dict(row)
    for k in ["state", "actions", "prior", "next_values", "global", "policy_target", "auxiliary"]:
        r[k] = np.asarray(r[k], dtype=np.float64)
    r["tokens"] = np.asarray(row["edge_tokens"], dtype=np.float64).reshape(len(r["prior"]), -1, 43)
    r["q_mask"] = np.array([x is not None for x in row["q_targets"]])
    r["q_targets"] = np.array([x if x is not None else 0 for x in row["q_targets"]])
    r["terminal_mask"] = np.array([x is not None for x in row["terminal"]])
    r["q_base_logit"] = np.arctanh(np.clip(r["next_values"], -1+1e-12, 1-1e-12))
    r["winning"] = np.array([x == 1 for x in row["terminal"]])
    if row["proven"] == 1:
        r["winning"] |= r["policy_target"] > 0
    return r


def measure(net, rows):
    details = []
    for r in rows:
        p, q, aux, _ = net.forward(r)
        raw = int(p.argmax())
        coupled = int((np.log(np.maximum(p, 1e-300)) + 16*q).argmax())
        base = int(r["prior"].argmax())
        details.append({"key": r["key"], "route": r["route"], "source": r["source"],
            "raw_known_win": bool(r["winning"][raw]), "coupled_known_win": bool(r["winning"][coupled]),
            "base_known_win": bool(r["winning"][base]), "has_known_win": bool(r["winning"].any()),
            "raw": raw, "coupled": coupled,
            "teacher_ce": float(-(r["policy_target"] * np.log(np.maximum(p, 1e-300))).sum()),
            "auxiliary_mse": float(((aux-r["auxiliary"])**2).mean())})
    return {"positions": len(details), "known_winning_positions": sum(d["has_known_win"] for d in details),
            "raw_known_wins": sum(d["raw_known_win"] for d in details),
            "coupled_known_wins": sum(d["coupled_known_win"] for d in details),
            "teacher_ce": float(np.mean([d["teacher_ce"] for d in details])), "details": details}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("native",type=Path);ap.add_argument("output",type=Path)
    ap.add_argument("--steps",type=int,default=64)
    args=ap.parse_args()
    if args.steps<=0:ap.error("--steps must be positive")
    args.output.mkdir(exist_ok=False)
    rows=[prepare(json.loads(s)) for s in (args.native/"rows.jsonl").read_text().splitlines()]
    train=[r for r in rows if not r["held_out"]];test=[r for r in rows if r["held_out"]]
    groups=sorted({r["source"] for r in train});a_groups=set(groups[::2])
    a=[r for r in train if r["source"] in a_groups];b=[r for r in train if r["source"] not in a_groups]
    assert a and b and test and not ({r['source'] for r in train}&{r['source'] for r in test})
    scale=np.maximum(.1,np.std(np.concatenate([r["auxiliary"] for r in train]),axis=0))
    baseline=measure(Network(17),rows)
    result={"schema":"gen5-relational-residual-experiment-v1","parameters":sum(v.size for v in Network(17).w.values()),
        "steps_per_phase":args.steps,"phase_B_recall":.5,"train_groups":groups,"test_groups":sorted({r['source'] for r in test}),
        "scale_from_training_only":scale.tolist(),"baseline":baseline,"trials":[],"publication_tested":False}
    start=time.perf_counter()
    for seed in [17,29,43]:
        for label,rel,aux in [("control",False,0.),("relations",True,0.),("relations_auxiliary",True,.1)]:
            net=Network(seed,rel,aux);rng=np.random.default_rng(seed+100)
            t=time.perf_counter()
            for _ in range(args.steps):net.update([a[int(rng.integers(len(a)))] for _ in range(2)],scale)
            first={"old":measure(net,a),"new":measure(net,b),"test":measure(net,test)}
            net.save(args.output/f"{seed}-{label}-A.npz")
            for _ in range(args.steps):net.update([a[int(rng.integers(len(a)))],b[int(rng.integers(len(b)))]],scale)
            second={"old":measure(net,a),"new":measure(net,b),"test":measure(net,test)}
            duration=time.perf_counter()-t
            net.save(args.output/f"{seed}-{label}-B.npz")
            missing=sum(x["raw_known_win"] and not y["raw_known_win"] for x,y in zip(first["old"]["details"],second["old"]["details"]))
            bench=[]
            for r in test[:12]:
                net.forward(r)
                times=[]
                for _ in range(5):
                    t=time.perf_counter();net.forward(r);times.append((time.perf_counter()-t)*1000)
                bench.append({"actions":len(r["prior"]),"edges":r["tokens"].shape[1],"median_ms":float(np.median(times))})
            trial={"seed":seed,"arm":label,"A":first,"B":second,"raw_old_losses_after_B":missing,"seconds":duration,"forward_cost":bench}
            result["trials"].append(trial)
            (args.output/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
            print(json.dumps({"seed":seed,"arm":label,"test_raw":second['test']['raw_known_wins'],"test_coupled":second['test']['coupled_known_wins'],"old_lost":missing,"seconds":duration}),flush=True)
    result["total_seconds"]=time.perf_counter()-start
    (args.output/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":main()
