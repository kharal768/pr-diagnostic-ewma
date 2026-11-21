"""Tuning of the top-r CUSUM reference value k in {0.25, 0.375, 0.5} for r in {1, 3, 5};
p = 30, delta = 0.75, in-control ARL 200, steady state, same seeds as robust2.py.
Inputs: front.json
Outputs: topr_k.json
Paper: Section 4.3, Fig. 2, Table 2.
"""
import json, os, numpy as np
from engine import arl
from calib import calibrate
P, TAU, MAXT, RC, RO = 30, 50, 3000, 12000, 20000
OUT = 'topr_k.json'; res = json.load(open(OUT)) if os.path.exists(OUT) else {}
F = json.load(open('front.json'))
for r in (1, 3, 5):
    for k in (0.25, 0.375, 0.5, 0.75):
        key = f'top-r CUSUM r={r} k={k}'
        if key in res: continue
        if k == 0.5: lim = F[f'top-r CUSUM r={r}']['lim']
        else:
            c, _ = calibrate(11, [k, r], lambda h: [h], 1., 150., p=P, R=RC, seed0=7, iters=20, maxT=MAXT); lim = [c]
        a0 = arl(11, [k, r], lim, p=P, R=RC, seed0=991, maxT=MAXT)[0]
        row = {'lim': [float(x) for x in lim], 'arl0': float(a0)}
        for nz in (1, 2, 3, 5, 10, 30): row[f'0.75_{nz}'] = float(arl(11, [k, r], lim, p=P, delta=0.75, tau=TAU, maxT=MAXT, R=RO, seed0=4321, nz=nz)[0])
        res[key] = row; json.dump(res, open(OUT, 'w'), indent=1)
        print(key, round(a0, 1), [round(row[f'0.75_{n}'], 2) for n in (1, 2, 3, 5, 10, 30)], flush=True)
print('DONE')
