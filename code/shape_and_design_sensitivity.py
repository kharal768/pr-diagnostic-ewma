"""(a) Equal-PR directions with different shapes; (b) dependence of the gain curve and the
break-even on delta and w1. p=30, Sigma=I.
Inputs: final.json
Outputs: m1_sims.json
Paper: Section 4.2 (design dependence of the break-even); Supplement S1.
"""
import json, numpy as np
from scipy.optimize import brentq
from calib import ss_profile
p, R = 30, 15000; F = json.load(open('final.json'))['p30']
REF = (0, [0.05], [F['bench']['MEWMA l=0.05']['lim']])
BAS = {w: (6, F['bas'][f'w1={w}']['prm'], F['bas'][f'w1={w}']['lim']) for w in ('0.1', '0.25', '0.5')}
PR = lambda d: float(1 / np.sum(d ** 4))
def tail_dir(target):  # one dominant loading a, 29 equal loadings b, PR = target
    f = lambda a: 1 / (a ** 4 + (1 - a ** 2) ** 2 / 29) - target
    a = brentq(f, 1 / np.sqrt(30) + 1e-9, 1 - 1e-9); d = np.full(p, np.sqrt((1 - a * a) / 29)); d[0] = a; return d
dirs = []
for k in (2, 3, 5):
    d = np.zeros(p); d[:k] = 1 / np.sqrt(k); dirs.append((f'{k} equal', d)); dirs.append((f'dominant+tail PR={k}', tail_dir(k)))
out = {'shapes': [], 'delta_w1': {}}
for i, (lab, d) in enumerate(dirs):
    seed = 50_000_000 + i * 100_000; base = ss_profile(*REF, [0.75, 1.0], p=p, R=R, seed0=seed, dvec=d)
    rec = {'dir': lab, 'pr': PR(d), 'maxd2': float(d.max() ** 2)}
    for w, ch in BAS.items(): rec[f'BAS w1={w}'] = float(100 * (1 - np.mean(ss_profile(*ch, [0.75, 1.0], p=p, R=R, seed0=seed, dvec=d) / base)))
    out['shapes'].append(rec); print(rec, flush=True); json.dump(out, open('m1_sims.json', 'w'), indent=1)
DEL = [0.5, 0.75, 1.0, 1.5]
for j, nz in enumerate((1, 2, 3, 5, 10, 30)):
    d = np.zeros(p); d[:nz] = 1 / np.sqrt(nz); seed = 60_000_000 + j * 100_000
    base = ss_profile(*REF, DEL, p=p, R=R, seed0=seed, dvec=d)
    for w, ch in BAS.items():
        a = ss_profile(*ch, DEL, p=p, R=R, seed0=seed, dvec=d)
        out['delta_w1'][f'nz={nz}|w1={w}'] = (100 * (1 - a / base)).tolist()
    print(nz, {k: np.round(v, 1).tolist() for k, v in out['delta_w1'].items() if k.startswith(f'nz={nz}|')}, flush=True)
    json.dump(out, open('m1_sims.json', 'w'), indent=1)
print('DONE', flush=True)
