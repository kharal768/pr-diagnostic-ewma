"""Tuned families of all charts at p = 30: limits and steady-state ARL profiles used as the common
reference designs (MEWMA and BAS-EWMA limits are reused by later scripts).
Inputs: none (simulation).
Outputs: front.json
Paper: Section 4.3, Fig. 2.
"""
import numpy as np, json, os, sys
from engine import arl, lewma_path_samples
from calib import calibrate
from bas_ewma import design as bas_design

P, TAU, ARL0, MAXT = 30, 50, 200.0, 3000
RC, RO = 12000, 20000
DELTA = 0.75
OUT = 'front.json'

res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def ev(key, kind, prm, limfun, lo, hi):
    if key in res:
        return
    c, a = calibrate(kind, prm, limfun, lo, hi, p=P, R=RC, seed0=7, iters=20,
                     target=ARL0, maxT=MAXT)
    lim = limfun(c)
    a0 = arl(kind, prm, lim, p=P, R=RC, seed0=991, maxT=MAXT)[0]
    s = arl(kind, prm, lim, p=P, delta=DELTA, tau=TAU, maxT=MAXT, R=RO, seed0=1234, nz=1)[0]
    d = arl(kind, prm, lim, p=P, delta=DELTA, tau=TAU, maxT=MAXT, R=RO, seed0=1234, nz=P)[0]
    res[key] = {'arl0': float(a0), 'sparse': float(s), 'dense': float(d),
                'lim': [float(x) for x in np.atleast_1d(lim)]}
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f"{key:28s} ARL0 {a0:6.1f}  sparse {s:6.2f}  dense {d:6.2f}", flush=True)

# MEWMA
for lam in (0.03, 0.05, 0.10):
    ev(f'MEWMA lam={lam}', 0, [lam], lambda h: [h], 15., 110.)
# VS-MEWMA (= unbalanced AS-EWMA, Sigma = I)
for lam in (0.03, 0.05, 0.10):
    ev(f'VS-MEWMA lam={lam}', 5, [lam], lambda h: [h], 0.5, 14.)
# top-r CUSUM
for r in (1, 2, 3, 5, 10):
    ev(f'top-r CUSUM r={r}', 11, [0.5, r], lambda h: [h], 1., 90.)
# LEWMA
for lam in (0.02, 0.05, 0.10):
    key = f'LEWMA lam={lam}'
    if key in res:
        continue
    smp = lewma_path_samples(lam, P, P, 4000, int(10 / lam), max(1, int(1 / lam)), 50, 777)
    prm = [lam] + smp.mean(0).tolist() + smp.std(0).tolist()
    ev(key, 8, prm, lambda h: [h], 0., 20.)
# BAS-EWMA
for lam in (0.03, 0.05, 0.09):
    for w1 in (0.05, 0.2, 0.4, 0.65, 0.9):
        key = f'BAS-EWMA lam={lam} w1={w1}'
        if key in res:
            continue
        lf = bas_design(lam, [1, P], P, w=[w1, 1 - w1])
        ev(key, 6, [lam, 1, P], lf, 0.5, 4.5)
print('ALL DONE', flush=True)
