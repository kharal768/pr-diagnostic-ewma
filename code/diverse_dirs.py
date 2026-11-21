"""Out-of-sample test of shape summaries on 48 random directions with diverse shapes.
p=30, Sigma=I, lambda=0.05, in-control ARL 200, gain averaged over delta 0.75 and 1.0.
Inputs: final.json
Outputs: diverse_dirs.json
Paper: Section 4.2; Supplement S1.
"""
import json, numpy as np
from calib import ss_profile
p, R = 30, 8000; F = json.load(open('final.json'))['p30']
REF = (0, [0.05], [F['bench']['MEWMA l=0.05']['lim']])
CH = {'BAS w1=0.50': (6, F['bas']['w1=0.5']['prm'], F['bas']['w1=0.5']['lim']), 'BAS w1=0.25': (6, F['bas']['w1=0.25']['prm'], F['bas']['w1=0.25']['lim'])}
rng = np.random.default_rng(2026); out = []
for i in range(48):
    k = int(np.clip(np.round(np.exp(rng.uniform(0, np.log(30)))), 1, 30)); alpha = rng.choice([0.2, 0.5, 1.0, 3.0])
    d = np.zeros(p); idx = rng.choice(p, k, replace=False); d[idx] = rng.gamma(alpha, 1.0, k) + 1e-6; d /= np.linalg.norm(d)
    seed = 90_000_000 + i * 100_000; base = ss_profile(*REF, [0.75, 1.0], p=p, R=R, seed0=seed, dvec=d)
    rec = {'k': k, 'alpha': float(alpha), 'd': d.tolist()}
    for n, ch in CH.items(): rec[n] = float(100 * (1 - np.mean(ss_profile(*ch, [0.75, 1.0], p=p, R=R, seed0=seed, dvec=d) / base)))
    out.append(rec); print(i, k, alpha, round(rec['BAS w1=0.50'], 2), flush=True); json.dump(out, open('diverse_dirs.json', 'w'))
print('DONE')
