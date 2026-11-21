"""As mix_dim.py with p0 = 3/p (fixed expected number of shifted coordinates).
Usage: python mix_dimk.py p
Inputs: none (simulation).
Outputs: mix_dimk_p{p}.json
Paper: Section 4.2; Supplement S16.
"""
import json, sys, numpy as np
from scipy.stats import chi2
lam, R, TMAX, TAU = 0.05, 3000, 2500, 50; cst = (2 - lam) / lam; P0 = 3.0 / int(sys.argv[1])
def mstat(u): return np.log1p(P0 * np.expm1(np.minimum(u, 600) / 2)).sum(1)
def ic(p, f, h, seed=11):
    rng = np.random.default_rng(seed); z = np.zeros((R, p)); fa = np.full(R, TMAX)
    for t in range(1, TMAX + 1):
        z = lam * rng.standard_normal((R, p)) + (1 - lam) * z; fa = np.where((f(cst * z * z) > h) & (fa == TMAX), t, fa)
        if (fa < TMAX).mean() > 0.997: break
    return fa.mean()
def cal(p, f, lo, hi):
    for _ in range(16):
        h = (lo + hi) / 2; lo, hi = (h, hi) if ic(p, f, h) < 200 else (lo, h)
    return (lo + hi) / 2
def ooc(p, ch, k, delta, seed):
    rng = np.random.default_rng(seed); d = np.zeros(p); d[:k] = 1 / np.sqrt(k); z = np.zeros((R, p)); out = {n: np.full(R, -1.0) for n in ch}; alive = {n: np.ones(R, bool) for n in ch}
    for t in range(1, TAU + TMAX + 1):
        x = rng.standard_normal((R, p)) + (delta * d if t > TAU else 0); z = lam * x + (1 - lam) * z; u = cst * z * z
        for n, (f, h) in ch.items():
            al = f(u) > h
            if t <= TAU: alive[n] &= ~al
            else: out[n][al & alive[n] & (out[n] < 0)] = t - TAU
        if t > TAU + 100 and all(((out[n] > 0) | ~alive[n]).all() for n in ch): break
    return {n: float(out[n][alive[n] & (out[n] > 0)].mean()) for n in ch}
p = int(sys.argv[1]); hM = cal(p, lambda u: u.sum(1), p, p + 30 * np.sqrt(p) + 40); hX = cal(p, mstat, 0.1, 40 + 0.5 * p)
ch = {'MEWMA': (lambda u: u.sum(1), hM), 'mix': (mstat, hX)}; ks = sorted(set(int(round(x)) for x in np.geomspace(1, min(p, 40), 9)))
res = []
for k in ks:
    g = np.mean([1 - (r := ooc(p, ch, k, dl, 100 * k + int(10 * dl)))['mix'] / r['MEWMA'] for dl in (0.75, 1.0)]); res.append([k, float(100 * g)]); print(p, k, round(100 * g, 2), flush=True)
    json.dump({'p': p, 'hM': hM, 'hX': hX, 'gains': res}, open(f'mix_dimk_p{p}.json', 'w'))
print('DONE')
