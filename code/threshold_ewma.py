"""Thresholded EWMA charts: hard threshold sum_i u_i 1{u_i > c} (Wu & Wu, 2025) and the
soft-thresholded variant sum_i (u_i - c)_+, for c in {2, 4, 6}, with MEWMA and BAS-EWMA
(w1 = 0.4) in the same framework. p = 30, lambda = 0.05.
Inputs: front.json
Outputs: threshold_ewma.json
Paper: Section 4.3, Fig. 2, Table 2.
"""
import json, numpy as np
p, lam, R, TMAX, TAU = 30, 0.05, 6000, 3000, 50; cst = (2 - lam) / lam
F = json.load(open('front.json')); LB = F['BAS-EWMA lam=0.05 w1=0.4']['lim']; LM = F['MEWMA lam=0.05']['lim'][0]
def stat(name, u):
    kind, c = name
    if kind == 'hard': return (u * (u > c)).sum(1)
    if kind == 'soft': return np.maximum(u - c, 0).sum(1)
def first_alarm_ic(name, h, seed=11):
    rng = np.random.default_rng(seed); z = np.zeros((R, p)); fa = np.full(R, TMAX)
    for t in range(1, TMAX + 1):
        z = lam * rng.standard_normal((R, p)) + (1 - lam) * z; s = stat(name, cst * z * z); fa = np.where((s > h) & (fa == TMAX), t, fa)
        if (fa < TMAX).mean() > 0.997: break
    return fa.mean()
def calibrate(name):
    lo, hi = 0.5, 400.0
    for _ in range(22):
        h = np.sqrt(lo * hi); a = first_alarm_ic(name, h); lo, hi = (h, hi) if a < 200 else (lo, h)
    return h, a
def ooc(charts, nz, seed):
    rng = np.random.default_rng(seed); d = np.zeros(p); d[:nz] = 1 / np.sqrt(nz); z = np.zeros((R, p))
    out = {k: np.full(R, -1.0) for k in charts}; alive = {k: np.ones(R, bool) for k in charts}
    for t in range(1, TAU + TMAX + 1):
        x = rng.standard_normal((R, p)) + (0.75 * d if t > TAU else 0); z = lam * x + (1 - lam) * z; u = cst * z * z
        for k, f in charts.items():
            al = f(u)
            if t <= TAU: alive[k] &= ~al
            else: out[k][al & alive[k] & (out[k] < 0)] = t - TAU
        if t > TAU + 100 and all(((out[k] > 0) | ~alive[k]).all() for k in charts): break
    return {k: float(out[k][alive[k] & (out[k] > 0)].mean()) for k in charts}
lims = {}; res = {}
for kind in ('hard', 'soft'):
    for c in (2.0, 4.0, 6.0):
        h, a = calibrate((kind, c)); lims[f'{kind} c={c}'] = (kind, c, float(h), float(a)); print(kind, c, round(h, 2), round(a, 1), flush=True)
charts = {'MEWMA': lambda u: u.sum(1) > LM, 'BAS-EWMA w1=0.40': lambda u: (u.max(1) > LB[0]) | (u.sum(1) > LB[1])}
for k, (kind, c, h, a) in lims.items(): charts[k] = (lambda kind, c, h: (lambda u: stat((kind, c), u) > h))(kind, c, h)
for nz in (1, 2, 3, 5, 10, 30):
    res[nz] = ooc(charts, nz, 600 + nz); print(nz, {k: round(v, 2) for k, v in res[nz].items()}, flush=True)
    json.dump({'limits': lims, 'ss_0.75': res}, open('threshold_ewma.json', 'w'), indent=1)
print('DONE')
