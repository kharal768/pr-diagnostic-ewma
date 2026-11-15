"""Extends the MSGL-EWMA tuning of msgl_bench.py around its best region.
Inputs: none (simulation).
Outputs: group_bench.json
Paper: Section 4.4.
"""
import json
from engine import arl
from calib import calibrate
from group_bench import SCEN, P, TAU, MAXT, RO
OUT = 'group_bench.json'
res = json.load(open(OUT))
grid = [(0.05, 0.5, 0.5), (0.03, 0.8, 0.8), (0.05, 0.9, 0.9), (0.05, 0.9, 0.8),
        (0.05, 0.8, 0.9), (0.05, 0.65, 0.8), (0.03, 0.9, 0.9)]
for r, c1, c2 in grid:
    key = f'MSGL r={r} c1={c1} c2={c2}'
    if key in res:
        continue
    prm = [r, 5, c1, c2]
    h, a = calibrate(13, prm, lambda h: [h], 0.05, 40.0, p=P, R=12000, seed0=7, iters=22, maxT=MAXT)
    row = {'lim': [float(h)], 'arl0': float(arl(13, prm, [h], p=P, R=RO, seed0=991, maxT=MAXT)[0])}
    if abs(row['arl0'] - 200) > 10:
        print(key, 'CALIBRATION FAILED', h, row['arl0'], flush=True); continue
    for s, v in SCEN.items():
        for dl in (0.75, 1.0):
            row[f'{s}|{dl}'] = float(arl(13, prm, [h], p=P, delta=dl, tau=TAU,
                                         maxT=MAXT, R=RO, seed0=4321, dvec=v)[0])
    res[key] = row
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f"{key:28s} H {h:.4f} ARL0 {row['arl0']:6.1f} | " +
          ' '.join(f"{row[f'{s}|0.75']:5.1f}" for s in SCEN), flush=True)
print('DONE', flush=True)
