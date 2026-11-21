"""Benchmark with known groups: p = 30 in six groups of five, Sigma = I, steady state, in-control
ARL 200; G-BAS, MEWMA and coordinate-sparse charts on in-group and cross-group shifts.
Inputs: front.json
Outputs: group_bench.json
Paper: Section 4.4, Fig. 3.
"""
import numpy as np, json, os
from engine import arl, lewma_path_samples
from gbas import calibrate_gbas

P, TAU, MAXT, RO = 30, 50, 3000, 20000
DELTAS = [0.75, 1.0]
F = json.load(open('front.json'))
OUT = 'group_bench.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def d(idx):
    v = np.zeros(P); v[list(idx)] = 1.0; return v / np.linalg.norm(v)

SCEN = {
    'A1  1 coord':                 d([0]),
    'A2  2 in one group':          d([0, 1]),
    'A3  3 in one group':          d([0, 1, 2]),
    'A5  whole group (5)':         d(range(5)),
    'A10 two whole groups':        d(range(10)),
    'A15 three whole groups':      d(range(15)),
    'A30 all 30':                  d(range(30)),
    'M5s 5 straddling 2 groups':   d(range(3, 8)),
    'M5x 5 scattered, 1 per group':d([0, 5, 10, 15, 20]),
    'M10x 10 scattered':           d([0, 1, 5, 6, 10, 11, 15, 16, 20, 21]),
}

def main():
    designs = {}
    designs['MEWMA'] = (0, [0.05], F['MEWMA lam=0.05']['lim'])
    designs['BAS-EWMA w1=0.40'] = (6, [0.05, 1, P], F['BAS-EWMA lam=0.05 w1=0.4']['lim'])
    smp = lewma_path_samples(0.05, P, P, 4000, 200, 20, 50, 777)
    designs['LEWMA'] = (8, [0.05] + smp.mean(0).tolist() + smp.std(0).tolist(), F['LEWMA lam=0.05']['lim'])
    designs['top-r CUSUM r=5'] = (11, [0.5, 5], F['top-r CUSUM r=5']['lim'])
    for w1 in (0.3, 0.5, 0.7):
        key = f'G-BAS w1={w1}'
        if key in res and 'lim' in res[key]:
            lim = res[key]['lim']
        else:
            lim, _ = calibrate_gbas(0.05, 5, P, w1)
        designs[key] = (12, [0.05, 5], lim)

    for name, (kind, prm, lim) in designs.items():
        if name in res and len(res[name]) > 2:
            continue
        row = {'lim': list(map(float, lim)),
               'arl0': float(arl(kind, prm, lim, p=P, R=RO, seed0=991, maxT=MAXT)[0])}
        for sname, v in SCEN.items():
            for dl in DELTAS:
                row[f'{sname}|{dl}'] = float(arl(kind, prm, lim, p=P, delta=dl, tau=TAU,
                                                 maxT=MAXT, R=RO, seed0=4321, dvec=v)[0])
        res[name] = row
        json.dump(res, open(OUT, 'w'), indent=1)
        print(f"{name:20s} ARL0 {row['arl0']:6.1f} | " +
              ' '.join(f"{row[f'{s}|0.75']:5.1f}" for s in SCEN), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
