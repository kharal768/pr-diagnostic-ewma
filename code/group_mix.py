"""Group-level mixture chart, sum_k ln{1 - p0 + p0 exp(Q_k / 2)}, on the group benchmark.
Inputs: none (simulation).
Outputs: group_mix.json
Paper: Section 4.4; Supplement S17.
"""
import json, numpy as np
from engine import arl
import group_bench as GB
P = 30; out = {}
def cal(p0):
    lo, hi = 0.1, 40.0
    for _ in range(16):
        h = (lo + hi) / 2; a = arl(20, [0.05, 5, p0], [h], p=P, R=6000, seed0=991, maxT=4000)[0]; lo, hi = (h, hi) if a < 200 else (lo, h)
    return (lo + hi) / 2
for p0 in (0.1, 0.3, 0.6):
    h = cal(p0); row = {'lim': [h], 'arl0': float(arl(20, [0.05, 5, p0], [h], p=P, R=20000, seed0=991, maxT=4000)[0]),
                        'ss_arl0': float(arl(20, [0.05, 5, p0], [h], p=P, delta=0.0, tau=200, maxT=20000, R=20000, seed0=777)[0])}
    for sname, v in GB.SCEN.items():
        for dl in GB.DELTAS: row[f'{sname}|{dl}'] = float(arl(20, [0.05, 5, p0], [h], p=P, delta=dl, tau=50, maxT=3000, R=20000, seed0=4321, dvec=v)[0])
    out[f'group mixture p0={p0}'] = row; print(p0, row, flush=True); json.dump(out, open('group_mix.json', 'w'), indent=1)
