"""Exchange rate of G-BAS: in-group gain against worst-case cost over the budget share w1.
Inputs: none (simulation).
Outputs: group_bench.json
Paper: Section 4.4, Fig. 3.
"""
import numpy as np, json, os
from engine import arl
from gbas import calibrate_gbas
from group_bench import SCEN, P, TAU, MAXT, RO

OUT = 'group_bench.json'
res = json.load(open(OUT))
for w1 in (0.05, 0.1, 0.2):
    key = f'G-BAS w1={w1}'
    if key in res:
        continue
    lim, _ = calibrate_gbas(0.05, 5, P, w1)
    row = {'lim': list(map(float, lim)),
           'arl0': float(arl(12, [0.05, 5], lim, p=P, R=RO, seed0=991, maxT=MAXT)[0])}
    for s, v in SCEN.items():
        for dl in (0.75, 1.0):
            row[f'{s}|{dl}'] = float(arl(12, [0.05, 5], lim, p=P, delta=dl, tau=TAU,
                                         maxT=MAXT, R=RO, seed0=4321, dvec=v)[0])
    res[key] = row
    json.dump(res, open(OUT, 'w'), indent=1)
    print(key, 'ARL0', round(row['arl0'], 1), flush=True)
print('DONE', flush=True)
