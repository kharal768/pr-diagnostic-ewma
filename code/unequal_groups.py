"""G-BAS with unequal correlated groups (sizes 2, 3, 5, 8, 12; within-group correlation 0.6):
raw against chi-square-standardised group maximum.
Inputs: none (simulation).
Outputs: unequal_groups.json
Paper: Section 4.4 (unequal groups); Supplement S11.
"""
import json, numpy as np
from scipy.stats import chi2
p, lam, R, TMAX, TAU, w1 = 30, 0.05, 4000, 3000, 50, 0.3; c = (2 - lam) / lam
sizes = [2, 3, 5, 8, 12]; G = []; i = 0
for s in sizes: G.append(list(range(i, i + s))); i += s
S = np.zeros((p, p))
for g in G: S[np.ix_(g, g)] = 0.6; 
np.fill_diagonal(S, 1.0); L = np.linalg.cholesky(S); Si = np.linalg.inv(S); Gi = [np.linalg.inv(S[np.ix_(g, g)]) for g in G]
def stats(z):
    tot = c * np.einsum('ri,ij,rj->r', z, Si, z); q = np.stack([c * np.einsum('ri,ij,rj->r', z[:, g], gi, z[:, g]) for g, gi in zip(G, Gi)], 1)
    qs = np.stack([-chi2.logsf(q[:, k], len(g)) for k, g in enumerate(G)], 1); return tot, q.max(1), qs.max(1)
def ic_samples(seed=1):
    rng = np.random.default_rng(seed); z = np.zeros((R, p)); T = []; Q = []; QS = []
    for t in range(500):
        z = lam * (rng.standard_normal((R, p)) @ L.T) + (1 - lam) * z
        if t >= 200 and t % 15 == 0: a, b, d = stats(z); T.append(a); Q.append(b); QS.append(d)
    return np.concatenate(T), np.concatenate(Q), np.concatenate(QS)
T, Q, QS = ic_samples()
def lims(kind, a):
    if kind == 'MEWMA': return (np.inf, np.quantile(T, 1 - a))
    first = Q if kind == 'raw' else QS; return (np.quantile(first, 1 - 2 * a * w1), np.quantile(T, 1 - 2 * a * (1 - w1)))
def alarm(kind, st, lim):
    tot, q, qs = st; f = q if kind == 'raw' else qs
    return tot > lim[1] if kind == 'MEWMA' else (f > lim[0]) | (tot > lim[1])
def arl0(kind, lim, seed=5):
    rng = np.random.default_rng(seed); z = np.zeros((R, p)); fa = np.full(R, TMAX)
    for t in range(1, TMAX + 1):
        z = lam * (rng.standard_normal((R, p)) @ L.T) + (1 - lam) * z; al = alarm(kind, stats(z), lim); fa = np.where(al & (fa == TMAX), t, fa)
        if (fa < TMAX).mean() > 0.997: break
    return fa.mean()
LIM = {}
for kind in ('MEWMA', 'raw', 'std'):
    lo, hi = 1e-5, 0.2
    for _ in range(16):
        a = np.sqrt(lo * hi); A = arl0(kind, lims(kind, a)); lo, hi = (lo, a) if A < 200 else (a, hi)
    LIM[kind] = lims(kind, a); print(kind, A, LIM[kind], flush=True)
def shift(idx):
    mu = np.zeros(p); mu[idx] = 1.0; return 0.75 * mu / np.sqrt(mu @ Si @ mu)
PAT = {'one var, group of 2': [0], 'whole group of 2': G[0], 'whole group of 3': G[1], 'whole group of 5': G[2], 'whole group of 8': G[3], 'whole group of 12': G[4],
       'one var, group of 12': [G[4][0]], 'half of group of 12': G[4][:6], 'one var in each group': [g[0] for g in G], 'all 30': list(range(p))}
res = {}
for j, (nm, idx) in enumerate(PAT.items()):
    mu = shift(idx); rng = np.random.default_rng(700 + j); z = np.zeros((R, p)); out = {k: np.full(R, -1.0) for k in LIM}; alive = {k: np.ones(R, bool) for k in LIM}
    for t in range(1, TAU + TMAX + 1):
        x = rng.standard_normal((R, p)) @ L.T + (mu if t > TAU else 0); z = lam * x + (1 - lam) * z; st = stats(z)
        for k in LIM:
            al = alarm(k, st, LIM[k])
            if t <= TAU: alive[k] &= ~al
            else: out[k][al & alive[k] & (out[k] < 0)] = t - TAU
        if t > TAU + 100 and all(((out[k] > 0) | ~alive[k]).all() for k in LIM): break
    res[nm] = {k: float(out[k][alive[k] & (out[k] > 0)].mean()) for k in LIM}; print(nm, {k: round(v, 2) for k, v in res[nm].items()}, flush=True)
    json.dump({'sizes': sizes, 'rho': 0.6, 'res': res}, open('unequal_groups.json', 'w'), indent=1)
print('DONE')
