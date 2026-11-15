"""Steady-state in-control ARL of MEWMA and the G-BAS designs (p = 30).
Inputs: front.json, group_bench.json
Outputs: gbas_ss.json
Paper: Section 4.4; Supplement S17.
"""
import json
from engine import arl
G = json.load(open('group_bench.json')); FR = json.load(open('front.json')); out = {}
for n, k, prm, lim in [('MEWMA', 0, [0.05], FR['MEWMA lam=0.05']['lim'])] + [(f'G-BAS w1={w}', 12, [0.05, 5], G[f'G-BAS w1={w}']['lim']) for w in (0.1, 0.3)]:
    a = arl(k, prm, lim, p=30, delta=0.0, tau=200, maxT=20000, R=20000, seed0=777); out[n] = [float(a[0]), float(a[1]) if len(a) > 1 else None]; print(n, out[n], flush=True)
json.dump(out, open('gbas_ss.json', 'w'), indent=1)
