"""Steady-state ARLs of the reference designs over shift sizes and sparsity levels, reusing the
limits in front.json.
Inputs: front.json
Outputs: robust2.json
Paper: Fig. 2, Table 2 and Section 4.3.
"""
import numpy as np, json, os
from engine import arl, lewma_path_samples

P, TAU, MAXT, RO = 30, 50, 3000, 20000
DELTAS = [0.5, 0.75, 1.0, 1.5, 2.0]
NZS = [1, 2, 3, 5, 10, 30]
F = json.load(open('front.json'))
OUT = 'robust2.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def prm_of(key):
    if key.startswith('MEWMA'):
        return 0, [float(key.split('=')[1])]
    if key.startswith('VS-MEWMA'):
        return 5, [float(key.split('=')[1])]
    if key.startswith('top-r'):
        return 11, [0.5, int(key.split('=')[1])]
    if key.startswith('LEWMA'):
        lam = float(key.split('=')[1])
        smp = lewma_path_samples(lam, P, P, 4000, int(10 / lam), max(1, int(1 / lam)), 50, 777)
        return 8, [lam] + smp.mean(0).tolist() + smp.std(0).tolist()
    parts = key.split()
    lam = float(parts[1].split('=')[1])
    return 6, [lam, 1, P]

SEL = ['BAS-EWMA lam=0.05 w1=0.4', 'BAS-EWMA lam=0.05 w1=0.65', 'LEWMA lam=0.05',
       'VS-MEWMA lam=0.05', 'top-r CUSUM r=1', 'top-r CUSUM r=3', 'top-r CUSUM r=5',
       'MEWMA lam=0.05']

for key in SEL:
    if key in res:
        continue
    kind, prm = prm_of(key)
    lim = F[key]['lim']
    row = {}
    for d in DELTAS:
        for nz in NZS:
            row[f'{d}_{nz}'] = float(arl(kind, prm, lim, p=P, delta=d, tau=TAU,
                                         maxT=MAXT, R=RO, seed0=4321, nz=nz)[0])
    res[key] = row
    json.dump(res, open(OUT, 'w'), indent=1)
    print(key, {k: round(v, 2) for k, v in row.items() if k.startswith('0.75') or k.startswith('1.5')}, flush=True)
print('DONE', flush=True)
