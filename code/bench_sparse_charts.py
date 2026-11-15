"""Benchmark of BAS-EWMA against the charts designed for sparse shifts (MEWMA, MCUSUM, VS-MEWMA,
LEWMA, top-r CUSUM) at p = 30, Sigma = I, steady state (tau = 50), in-control ARL 200.
Inputs: final.json
Outputs: bench.json
Paper: Section 4.3.
"""
import numpy as np, json, sys
from engine import arl
from calib import calibrate
from bas_ewma import calibrate_bas

P, TAU, ARL0 = 30, 50, 200.0
RC, RO, MAXT = 30000, 30000, 4000
DELTAS = [0.5, 0.75, 1.0, 1.5]
NZS = [1, 2, 5, 30]

def cal(kind, prm, lo, hi):
    c, a = calibrate(kind, prm, lambda h: [h], lo, hi, p=P, R=RC, seed0=7, iters=26,
                     target=ARL0, maxT=MAXT)
    return [c]

designs = {}
designs['MEWMA l=0.05']   = (0, [0.05], cal(0, [0.05], 20., 90.))
designs['MEWMA l=0.10']   = (0, [0.10], cal(0, [0.10], 20., 110.))
designs['MCUSUM k=0.25']  = (1, [0.25], cal(1, [0.25], 2., 45.))
designs['VS-MEWMA l=0.05']= (5, [0.05], cal(5, [0.05], 1., 12.))
designs['top-r CUSUM r=1']  = (11, [0.5, 1],  cal(11, [0.5, 1], 1., 30.))
designs['top-r CUSUM r=5']  = (11, [0.5, 5],  cal(11, [0.5, 5], 1., 60.))
for k, v in designs.items():
    print(k, 'limit', np.round(v[2], 4), flush=True)

# LEWMA needs its in-control standardizers; reuse the stored design if present
lew = json.load(open('final.json'))['p30']['lewma']['lam=0.05 q=30']
designs['LEWMA l=0.05'] = (8, lew['prm'], [lew['lim']])

prm6, lim6, _a6 = calibrate_bas(0.05, [1, P], P, RC, [0.25, 0.75], MAXT)
designs['BAS-EWMA l=0.05'] = (6, prm6, lim6)
print('BAS', np.round(prm6, 4), np.round(lim6, 4), flush=True)

rows = {}
for name, (kind, prm, lim) in designs.items():
    a0 = arl(kind, prm, lim, p=P, R=RC, seed0=991, maxT=MAXT)[0]
    out = {'arl0': float(a0)}
    for nz in NZS:
        for d in DELTAS:
            a = arl(kind, prm, lim, p=P, delta=d, tau=TAU, maxT=MAXT, R=RO, seed0=1234, nz=nz)[0]
            out[f'nz{nz}_d{d}'] = float(a)
    rows[name] = out
    print(name, 'ARL0', round(a0, 1), {k: round(v, 2) for k, v in out.items() if k != 'arl0'}, flush=True)

json.dump(rows, open('bench.json', 'w'), indent=1)
