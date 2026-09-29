"""Common residual policy reader, or separate motif probe, for all R2 arms."""
import numpy as np
from . import graph


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def prepare(row):
    r = graph.prepare(row)
    r['state'] = np.asarray(row['state'], dtype=float)
    if row['task'] == 'policy':
        r['actions'] = np.array(row['actions'], dtype=float)
        r['prior'] = np.asarray(row['prior'], dtype=float)
        r['base_logits'] = np.log(np.maximum(r['prior'], 1e-300))
        r['winning'] = np.array(row['winning'], dtype=bool)
    else:
        r['actions'] = np.zeros((1, 32))
        r['mask'] = np.array([x is not None for x in row['targets']])[None, :]
        r['targets'] = np.array([0 if x is None else x for x in row['targets']], dtype=float)[None, :]
    return r


class Network:
    def __init__(self, seed, mode, task):
        self.mode, self.task, self.steps = mode, task, 0
        rng = np.random.default_rng(seed + 1000)
        self.w = {'encoder': graph.seeded(seed), 'hidden': rng.normal(0, np.sqrt(2/561), (497, 64)),
                  'hidden_b': np.zeros(64), 'out': np.zeros((64, 1 if task == 'policy' else 4)),
                  'out_b': np.zeros(1 if task == 'policy' else 4),
                  'query': rng.normal(0, np.sqrt(2/48), (32, 16))}
        self.m = {k: np.zeros_like(v) for k, v in self.w.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.w.items()}

    def forward(self, r):
        pooled, nodes, gc = graph.forward(r, self.w['encoder'], self.mode)
        attention, query = None, None
        context = np.zeros((len(r['actions']), 16))
        if self.task == 'policy' and self.mode in ['pieces', 'messages'] and len(nodes):
            query = r['actions'] @ self.w['query']
            attention = softmax(query @ nodes.T / 4)
            context = attention @ nodes
        w = self.w['hidden']
        # Position and graph contributions are shared by all actions.
        h = np.tanh(r['state'] @ w[:417] + r['actions'] @ w[417:449]
                    + pooled @ w[449:481] + context @ w[481:] + self.w['hidden_b'])
        y = h @ self.w['out'] + self.w['out_b']
        if self.task == 'policy':
            prediction = r['prior'].copy() if np.all(y == 0) else softmax(r['base_logits'] + y[:, 0])
        else:
            prediction = np.exp(-np.logaddexp(0, -y))
        return prediction, (pooled, nodes, gc, attention, query, context, h, y)

    def loss_gradient(self, r, gradient=True):
        prediction, c = self.forward(r)
        pooled, nodes, gc, attention, query, context, h, y = c
        if self.task == 'policy':
            logits = r['base_logits'] + y[:, 0]
            # Stable -log(sum of probabilities on every proved winning move).
            loss = np.logaddexp.reduce(logits) - np.logaddexp.reduce(logits[r['winning']])
            dy = softmax(logits)
            dy[r['winning']] -= softmax(logits[r['winning']])
            dy = dy[:, None]
        else:
            n = max(1, r['mask'].sum())
            loss = ((np.logaddexp(0, y) - r['targets'] * y) * r['mask']).sum() / n
            dy = (prediction - r['targets']) * r['mask'] / n
        if not gradient:
            return float(loss)
        dw = {k: np.zeros_like(v) for k, v in self.w.items()}
        dw['out'] = h.T @ dy
        dw['out_b'] = dy.sum(axis=0)
        dh = (dy @ self.w['out'].T) * (1 - h * h)
        total = dh.sum(axis=0)
        dw['hidden'][:417] = np.outer(r['state'], total)
        dw['hidden'][417:449] = r['actions'].T @ dh
        dw['hidden'][449:481] = np.outer(pooled, total)
        dw['hidden'][481:] = context.T @ dh
        dw['hidden_b'] = total
        dpool = self.w['hidden'][449:481] @ total
        dnodes = np.zeros_like(nodes)
        if attention is not None:
            dc = dh @ self.w['hidden'][481:].T
            dnodes += attention.T @ dc
            da = dc @ nodes.T
            ds = attention * (da - (da * attention).sum(axis=1, keepdims=True))
            dq = ds @ nodes / 4
            dnodes += ds.T @ query / 4
            dw['query'] = r['actions'].T @ dq
        dw['encoder'] = graph.backward(r, self.w['encoder'], self.mode, gc, dpool, dnodes)
        return float(loss), dw

    def update(self, batch, rate=.001):
        grads = {k: np.zeros_like(v) for k, v in self.w.items()}
        for r in batch:
            _, g = self.loss_gradient(r)
            for k in grads:
                grads[k] += g[k] / len(batch)
        factor = min(1., 5. / max(1e-30, np.sqrt(sum(float((g*g).sum()) for g in grads.values()))))
        self.steps += 1
        for k, g in grads.items():
            g *= factor
            self.m[k] = .9*self.m[k] + .1*g
            self.v[k] = .999*self.v[k] + .001*g*g
            self.w[k] -= rate * (self.m[k]/(1-.9**self.steps)) / (np.sqrt(self.v[k]/(1-.999**self.steps)) + 1e-8)

    def save(self, path):
        np.savez(path, **self.w, **{'adam_m_'+k: v for k, v in self.m.items()},
                 **{'adam_v_'+k: v for k, v in self.v.items()}, steps=self.steps)
