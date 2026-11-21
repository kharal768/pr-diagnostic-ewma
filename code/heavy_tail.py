"""Gain of BAS-EWMA (w1=0.25) over MEWMA under a multivariate t distribution with 5 degrees of
freedom and under normality, with empirically calibrated limits.

p=30, lambda=0.05, zero-state in-control ARL 200.
Inputs: none (simulation).
Outputs: heavy_tail.json
Paper: Section 4.2 (heavy tails).
"""
import json, sys, numpy as np
p, lam, w1, R, TMAX, TAU = 30, 0.05, 0.25, 4000, 3000, 50; c = (2 - lam) / lam
def draw(rng, n, dist):
    z = rng.standard_normal((n, p))
    if dist == 'normal': return z
    nu = 5; g = rng.chisquare(nu, (n, 1)) / nu; return z / np.sqrt(g) * np.sqrt((nu - 2) / nu)
def ic_quantiles(dist, rng):
    z = np.zeros((R, p)); U = []; S = []
    for t in range(400):
        z = lam * draw(rng, R, dist) + (1 - lam) * z
        if t >= 200 and t % 10 == 0: u = c * z * z; U.append(u.max(1)); S.append(u.sum(1))
    return np.concatenate(U), np.concatenate(S)
def run(dist, lims, delta, d, seed):
    rng = np.random.default_rng(seed); z = np.zeros((R, p)); out = {k: np.full(R, -1.0) for k in lims}; alive = {k: np.ones(R, bool) for k in lims}
    for t in range(1, TAU + TMAX + 1):
        x = draw(rng, R, dist)
        if t > TAU: x = x + delta * d
        z = lam * x + (1 - lam) * z; u = c * z * z; mx, tot = u.max(1), u.sum(1)
        for k, (h1, h2) in lims.items():
            al = (mx > h1) | (tot > h2)
            new = al & alive[k] & (out[k] < 0)
            if t <= TAU: alive[k] &= ~al
            else: out[k][new] = t - TAU
        if t > TAU + 200 and all(((out[k] > 0) | ~alive[k]).all() for k in lims): break
    return {k: out[k][alive[k] & (out[k] > 0)].mean() for k in lims}
def calibrate(dist, seed):
    rng = np.random.default_rng(seed); U, S = ic_quantiles(dist, rng); res = {}
    for name in ('MEWMA', 'BAS'):
        lo, hi = 1e-5, 0.2
        for _ in range(18):
            a = np.sqrt(lo * hi)
            lim = (np.inf, np.quantile(S, 1 - a)) if name == 'MEWMA' else (np.quantile(U, 1 - 2 * a * w1), np.quantile(S, 1 - 2 * a * (1 - w1)))
            r2 = np.random.default_rng(seed + 1); z = np.zeros((R, p)); fa = np.full(R, TMAX)
            for t in range(1, TMAX + 1):
                z = lam * draw(r2, R, dist) + (1 - lam) * z; u = c * z * z
                al = (u.max(1) > lim[0]) | (u.sum(1) > lim[1]); fa = np.where(al & (fa == TMAX), t, fa)
                if t > 400 and (fa < TMAX).mean() > 0.995: break
            arl0 = fa.mean(); lo, hi = (lo, a) if arl0 < 200 else (a, hi)
        res[name] = [float(lim[0]), float(lim[1]), float(arl0)]; print(dist, name, res[name], flush=True)
    return res
out = {}
for dist in ('normal', 't5'):
    L = calibrate(dist, 11); lims = {k: (v[0], v[1]) for k, v in L.items()}; g = {}
    for nz in (1, 2, 3, 5, 10, 30):
        d = np.zeros(p); d[:nz] = 1 / np.sqrt(nz); r = [run(dist, lims, dl, d, 500 + nz) for dl in (0.75, 1.0)]
        g[nz] = float(100 * np.mean([1 - x['BAS'] / x['MEWMA'] for x in r])); print(dist, nz, round(g[nz], 2), flush=True)
    out[dist] = {'limits': L, 'gain': g}; json.dump(out, open('heavy_tail.json', 'w'), indent=1)
print('DONE')
