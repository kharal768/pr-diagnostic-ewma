"""MSGL-EWMA reading A re-tuned over the full (c1, c2) range in the setting of Kim et al. (2024).
Inputs: none (simulation).
Outputs: authors_bench.json
Paper: Section 4.4; Supplement S6.
"""
import numpy as np, json
from engine import arl
from calib import calibrate
from msgl_authors_setting import CASES, RHO, TAU, MAXT, RC, RO, DELTAS, setup

OUT = 'authors_bench.json'
res = json.load(open(OUT))
def run(p, g, lams):
    R0 = setup(g); R = np.kron(np.eye(p // g), R0); tag = f'p{p}'
    for lam in lams:
        for c1 in (0.2, 0.5, 0.8):
            for c2 in (0.2, 0.5, 0.8):
                name = f'MSGL lam={lam} c1={c1} c2={c2}'
                if name in res[tag]:
                    continue
                prm = [lam, g, c1, c2, float(np.linalg.eigvalsh(R0.T @ R0).max())] + \
                      R0.ravel().tolist() + (R0.T @ R0).ravel().tolist()
                h, _ = calibrate(14, prm, lambda h: [h], 0.01, 60., p=p, R=RC, seed0=7, iters=22, maxT=MAXT)
                row = {'lim': [float(h)], 'arl0': float(arl(14, prm, [h], p=p, R=RO, seed0=991, maxT=MAXT)[0])}
                if abs(row['arl0'] - 200) > 15:
                    print(tag, name, 'CALIBRATION OFF', h, row['arl0'], flush=True); continue
                for cname, idx in CASES[p].items():
                    d = np.zeros(p); d[idx] = 1.0; w = R @ d; nc = np.linalg.norm(w)
                    for dl in DELTAS:
                        row[f'{cname}|{dl}'] = float(arl(14, prm, [h], p=p, delta=dl * nc, tau=TAU,
                                                         maxT=MAXT, R=RO, seed0=4321, dvec=w / nc)[0])
                res[tag][name] = row
                json.dump(res, open(OUT, 'w'), indent=1)
                print(tag, f'{name:30s} ARL0 {row["arl0"]:6.1f} | ' +
                      ' '.join(f"{row[f'{c}|0.5']:6.1f}" for c in CASES[p]), flush=True)

if __name__ == '__main__':
    run(25, 5, (0.1, 0.05))
    run(10, 2, (0.1,))
    print('DONE', flush=True)
