"""Numpy mirror of the native 1424-parameter encoder; parity checked before fit."""
import numpy as np

H, PARAMETERS = 16, 1424
NB, MW, MB, UW, UB = 256, 272, 880, 896, 1408


def seeded(seed):
    rng = np.random.default_rng(seed)
    w = np.zeros(PARAMETERS)
    for start, inputs in [(0, 16), (MW, 38), (UW, 32)]:
        scale = np.sqrt(6 / (inputs + H))
        w[start:start + inputs * H] = rng.uniform(-scale, scale, inputs * H)
    return w


def prepare(row):
    r = dict(row)
    r['nodes'] = np.asarray(row['nodes'], dtype=float).reshape(-1, 16)
    r['src'] = np.array([e['source'] for e in row['edges']], dtype=int)
    r['dst'] = np.array([e['destination'] for e in row['edges']], dtype=int)
    r['edge'] = np.array([e['features'] for e in row['edges']], dtype=float).reshape(-1, 6)
    r['pool'] = r['nodes'][:, 12:14].T / 8
    r['degree'] = np.maximum(1, np.bincount(r['dst'], minlength=len(r['nodes'])))[:, None]
    return r


def forward(r, w, mode):
    nodes, src, dst = r['nodes'], r['src'], r['dst']
    if mode == 'dense':
        return np.zeros(32), np.zeros((len(nodes), H)), None
    if mode == 'edges':
        x = np.concatenate((nodes[src], nodes[dst], r['edge']), axis=1)
        h = np.tanh(x @ w[MW:MB].reshape(38, H) + w[MB:UW])
        pool = r['edge'][:, :2].T
        pool = pool / np.maximum(1, pool.sum(axis=1))[:, None]
        return (pool @ h).ravel(), np.zeros((len(nodes), H)), (x, h, pool)
    h = np.tanh(nodes @ w[:NB].reshape(16, H) + w[NB:MW])
    states, rounds = [h], []
    for _ in range(2 if mode == 'messages' else 0):
        x = np.concatenate((h[src], h[dst], r['edge']), axis=1)
        m = np.tanh(x @ w[MW:MB].reshape(38, H) + w[MB:UW])
        aggregate = np.zeros_like(h)
        np.add.at(aggregate, dst, m / r['degree'][dst])
        u = np.concatenate((h, aggregate), axis=1)
        h = np.tanh(u @ w[UW:UB].reshape(32, H) + w[UB:])
        rounds.append((x, m, u))
        states.append(h)
    return (r['pool'] @ h).ravel(), h, (states, rounds)


def backward(r, w, mode, cache, dpool, dnodes):
    dw = np.zeros_like(w)
    if mode == 'dense':
        return dw
    if mode == 'edges':
        x, h, pool = cache
        dz = (pool.T @ dpool.reshape(2, H)) * (1 - h * h)
        dw[MW:MB] = (x.T @ dz).ravel()
        dw[MB:UW] = dz.sum(axis=0)
        return dw
    states, rounds = cache
    dh = dnodes + r['pool'].T @ dpool.reshape(2, H)
    for t in range(len(rounds) - 1, -1, -1):
        x, m, u = rounds[t]
        dz = dh * (1 - states[t + 1] ** 2)
        dw[UW:UB] += (u.T @ dz).ravel()
        dw[UB:] += dz.sum(axis=0)
        du = dz @ w[UW:UB].reshape(32, H).T
        dh = du[:, :H].copy()
        dm = du[r['dst'], H:] / r['degree'][r['dst']]
        dz = dm * (1 - m * m)
        dw[MW:MB] += (x.T @ dz).ravel()
        dw[MB:UW] += dz.sum(axis=0)
        dx = dz @ w[MW:MB].reshape(38, H).T
        np.add.at(dh, r['src'], dx[:, :H])
        np.add.at(dh, r['dst'], dx[:, H:2*H])
    dz = dh * (1 - states[0] ** 2)
    dw[:NB] = (nodes_transpose(r) @ dz).ravel()
    dw[NB:MW] = dz.sum(axis=0)
    return dw


def nodes_transpose(r):
    return r['nodes'].T
