"""G-BAS, MEWMA and MSGL-EWMA in the setting of Kim et al. (2024): five equal groups, within-group
correlation 0.75, steady state, in-control ARL 200.
Inputs: none (simulation).
Outputs: authors_bench.json
Paper: Fig. 4 and Section 4.4.
"""
import numpy as np, json, os, sys
from engine import arl
from calib import calibrate
from gbas import limfun_for
from bas_ewma import calibrate_bas

RHO, TAU, MAXT = 0.75, 50, 2500
RC, RO = 6000, 10000
DELTAS = [0.5, 1.0]
OUT = 'authors_bench.json'
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def setup(g):
    S = (1 - RHO) * np.eye(g) + RHO * np.ones((g, g))
    return np.linalg.inv(np.linalg.cholesky(S))

CASES = {
  10: {'one variable': [0], 'two in one group': [0, 1], 'two in different groups': [0, 2],
       'three across two groups': [0, 1, 2], 'two whole groups': [0, 1, 2, 3],
       'one in every group': [0, 2, 4, 6, 8], 'all 10': list(range(10))},
  25: {'one variable': [0], 'two in one group': [0, 1], 'two in different groups': [0, 5],
       'three in one group': [0, 1, 2], 'whole group': [0, 1, 2, 3, 4],
       'two in each of two groups': [0, 1, 5, 6], 'one in every group': [0, 5, 10, 15, 20],
       'all 25': list(range(25))},
}

def run(p, g):
    R0 = setup(g); R = np.kron(np.eye(p // g), R0)
    tag = f'p{p}'
    res.setdefault(tag, {})
    designs = []
    for lam in (0.05, 0.1, 0.2):
        designs.append((f'MEWMA lam={lam}', 0, [lam], lambda h: [h], 2., 80.))
    for lam in (0.05, 0.1, 0.2):
        for w1 in (0.1, 0.3, 0.5):
            designs.append((f'G-BAS lam={lam} w1={w1}', 12, [lam, g], ('gbas', lam, w1), None, None))
    for lam in (0.05, 0.1):
        designs.append((f'BAS-EWMA lam={lam} w1=0.4', 6, [lam, 1, p], ('bas', lam, 0.4), None, None))
    msprm = lambda lam, c1, c2: [lam, g, c1, c2, float(np.linalg.eigvalsh(R0.T @ R0).max())] + \
        R0.ravel().tolist() + (R0.T @ R0).ravel().tolist()
    for lam in (0.05, 0.1, 0.2):
        for c in ((0.8, 0.8), (0.9, 0.9)):
            designs.append((f'MSGL lam={lam} c1={c[0]} c2={c[1]}', 14, msprm(lam, *c), lambda h: [h], 0.01, 30.))
    for name, kind, prm, lf, lo, hi in designs:
        if name in res[tag]:
            continue
        if isinstance(lf, tuple) and lf[0] == 'gbas':
            f = limfun_for(lf[1], g, p, lf[2])
            c, _ = calibrate(12, prm, f, 0.5, 4.5, p=p, R=RC, iters=20, maxT=MAXT); lim = f(c)
        elif isinstance(lf, tuple) and lf[0] == 'bas':
            _, lim, _ = calibrate_bas(lf[1], [1, p], p, RC, [lf[2], 1 - lf[2]], MAXT)
        else:
            h, _ = calibrate(kind, prm, lf, lo, hi, p=p, R=RC, seed0=7, iters=20, maxT=MAXT); lim = [h]
        row = {'lim': list(map(float, lim)), 'arl0': float(arl(kind, prm, lim, p=p, R=RO, seed0=991, maxT=MAXT)[0])}
        if abs(row['arl0'] - 200) > 15:
            print(tag, name, 'CALIBRATION OFF', row['arl0'], flush=True)
        for cname, idx in CASES[p].items():
            d = np.zeros(p); d[idx] = 1.0
            w = R @ d; nc = np.linalg.norm(w); dvec = w / nc
            for dl in DELTAS:
                row[f'{cname}|{dl}'] = float(arl(kind, prm, lim, p=p, delta=dl * nc, tau=TAU,
                                                 maxT=MAXT, R=RO, seed0=4321, dvec=dvec)[0])
        res[tag][name] = row
        json.dump(res, open(OUT, 'w'), indent=1)
        print(tag, f'{name:30s} ARL0 {row["arl0"]:6.1f} | ' +
              ' '.join(f"{row[f'{c}|0.5']:6.1f}" for c in CASES[p]), flush=True)

if __name__ == '__main__':
    run(10, 2)
    run(25, 5)
    print('DONE', flush=True)
