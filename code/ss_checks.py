"""Steady-state checks: change point tau=50 against tau=5/lambda, steady-state in-control ARL of
the compared designs, and the group-EWMA maximum alone in the group benchmark.
Inputs: final.json, front.json
Outputs: ss_checks.json
Paper: Sections 2.4 and 3.1; Supplement S3.
"""
import json, numpy as np
from engine import arl, lewma_path_samples
from calib import calibrate
from group_bench import SCEN
P, MAXT, R = 30, 3000, 20000; F = json.load(open('final.json'))['p30']; FR = json.load(open('front.json')); out = {}
def lew(lam):
    smp = lewma_path_samples(lam, P, P, 4000, 200, 20, 50, 777); return [lam] + smp.mean(0).tolist() + smp.std(0).tolist()
D = {'MEWMA lam=0.02': (0, [0.02], [F['bench']['MEWMA l=0.02']['lim']]), 'MEWMA lam=0.03': (0, [0.03], [F['bench']['MEWMA l=0.03']['lim']])}
out['tau'] = {}
for n, (k, prm, lim) in D.items():
    lam = prm[0]; t2 = int(round(5 / lam))
    for nz in (1, 30):
        a = arl(k, prm, lim, p=P, delta=0.75, tau=50, maxT=MAXT, R=R, seed0=4321, nz=nz)[0]; b = arl(k, prm, lim, p=P, delta=0.75, tau=t2, maxT=MAXT, R=R, seed0=4321, nz=nz)[0]
        out['tau'][f'{n}|nz={nz}'] = [float(a), float(b), t2]; print(n, nz, round(a, 2), round(b, 2), t2, flush=True)
json.dump(out, open('ss_checks.json', 'w'), indent=1)
TAB = {'MEWMA lam=0.05': (0, [0.05]), 'BAS-EWMA lam=0.05 w1=0.4': (6, [0.05, 1, P]), 'VS-MEWMA lam=0.05': (5, [0.05]), 'top-r CUSUM r=5': (11, [0.5, 5]), 'top-r CUSUM r=1': (11, [0.5, 1])}
out['ss_arl0'] = {}
for n, (k, prm) in TAB.items():
    a = arl(k, prm, FR[n]['lim'], p=P, delta=0.0, tau=200, maxT=20000, R=R, seed0=777)[0]; out['ss_arl0'][n] = [float(FR[n]['arl0']), float(a)]; print('ssARL0', n, FR[n]['arl0'], round(a, 1), flush=True)
json.dump(out, open('ss_checks.json', 'w'), indent=1)
c, a0 = calibrate(12, [0.05, 5], lambda h: [h, 1e9], 5.0, 60.0, p=P, R=12000, seed0=7, iters=22, maxT=MAXT)
row = {'lim': [float(c), 1e9], 'arl0': float(a0)}
for s, v in SCEN.items(): row[s] = float(arl(12, [0.05, 5], [c, 1e9], p=P, delta=0.75, tau=50, maxT=MAXT, R=R, seed0=4321, dvec=v)[0])
out['group_max_only'] = row; print(row, flush=True); json.dump(out, open('ss_checks.json', 'w'), indent=1); print('DONE')
