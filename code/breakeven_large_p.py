"""Break-even at p = 200, 500 and 1000, used to compare logarithmic and power-law growth.
Inputs: none (simulation).
Outputs: decide.json
Paper: Fig. 1 and Section 4.2.
"""
import json, numpy as np
from engine import arl
from calib import calibrate, ss_profile
from bas_ewma import calibrate_bas

D = [0.75, 1.0]
out = {}

def scan(p, mew, bas, grid, R, tag):
    g = []
    for i, nz in enumerate(grid):
        d = np.zeros(p); d[:nz] = 1 / np.sqrt(nz)
        seed = 110_000_000 + p * 1000 + i * 100
        base = ss_profile(0, [0.05], [mew], D, p=p, R=R, seed0=seed, dvec=d)
        a = ss_profile(6, [0.05, 1.0, float(p)], bas, D, p=p, R=R, seed0=seed, dvec=d)
        v = float(100 * (1 - np.mean(a / base)))
        g.append(v); print(f"  {tag} PR={nz:3d}  {v:6.2f}%", flush=True)
    gg, pr = np.array(g), np.array(grid, float)
    neg = np.where(gg < 0)[0]
    be = float(np.exp(np.interp(0.0, [gg[neg[0]], gg[neg[0] - 1]],
                                [np.log(pr[neg[0]]), np.log(pr[neg[0] - 1])]))) if len(neg) and neg[0] > 0 else float("nan")
    out[p] = dict(grid=grid, gains=g, be=be, R=R)
    print(f"  -> p={p} break-even = {be:.2f}\n", flush=True)
    json.dump(out, open("decide.json", "w"), indent=1)

print("(a) high-precision re-measurement", flush=True)
scan(200, 242.13, [17.82, 246.00], [7, 8, 9, 10, 11], 30000, "p=200")
scan(500, 564.98, [19.65, 572.17], [8, 9, 10, 11, 12, 13], 30000, "p=500")

print("(b) p = 1000", flush=True)
p = 1000
cm, _ = calibrate(0, [0.05], lambda c: [c], 900.0, 1300.0, p=p, R=4000, iters=20, maxT=2500)
print(f"  MEWMA lim={cm:.2f} ARL0={arl(0,[0.05],[cm],p=p,R=20000,seed0=5_000_000)[0]:.1f}", flush=True)
prm_b, lim_b, _ = calibrate_bas(0.05, [1, p], p, R=4000, w=[0.25, 0.75], maxT=2500)
print(f"  BAS   lim={np.round(lim_b,2)} ARL0={arl(6,prm_b,lim_b,p=p,R=20000,seed0=5_000_000)[0]:.1f}", flush=True)
out["p1000_limits"] = dict(mewma=cm, bas=list(lim_b))
scan(p, cm, lim_b, [7, 9, 11, 13, 15, 18, 22], 8000, "p=1000")
print("DONE")
