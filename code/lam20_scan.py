"""Break-even at lambda = 0.20 using the limits from lam20_cal.py.
Inputs: lam20_lims.json
Outputs: lam20_p1000.json
Paper: Section 4.2 (slope at lambda = 0.20).
"""
import json, numpy as np
from calib import ss_profile

LAM, P, R = 0.20, 1000, 3000
D = [0.75, 1.0]
GRID = [4, 6, 8, 10, 13, 16]
L = json.load(open("lam20_lims.json"))
g = []
for i, nz in enumerate(GRID):
    d = np.zeros(P); d[:nz] = 1 / np.sqrt(nz)
    seed = 170_000_000 + i * 1000
    base = ss_profile(0, [LAM], [L["mewma"]], D, p=P, R=R, seed0=seed, dvec=d)
    a = ss_profile(6, [LAM, 1, P], L["bas"], D, p=P, R=R, seed0=seed, dvec=d)
    v = float(100 * (1 - np.mean(a / base))); g.append(v)
    print(f"PR={nz:3d}  {v:6.2f}%", flush=True)
gg, pr = np.array(g), np.array(GRID, float)
neg = np.where(gg < 0)[0]
be = float(np.exp(np.interp(0.0, [gg[neg[0]], gg[neg[0] - 1]],
                            [np.log(pr[neg[0]]), np.log(pr[neg[0] - 1])]))) if len(neg) and neg[0] > 0 else float("nan")
print(f"-> lambda=0.20, p=1000 break-even {be:.2f}", flush=True)
json.dump({"grid": GRID, "gains": g, "be": be, **L}, open("lam20_p1000.json", "w"), indent=1)
