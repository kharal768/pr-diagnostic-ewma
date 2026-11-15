"""Locates the break-even at p = 100, 200 and 500 on a dense PR grid, reusing the limits from
breakeven_by_dimension.py.
Inputs: none (simulation).
Outputs: refine.json
Paper: Fig. 1 and Section 4.2.
"""
import json, numpy as np
from calib import ss_profile

D = [0.75, 1.0]
CFG = {100: dict(R=8000, mew=130.33, bas=[16.52, 132.98], grid=[5, 6, 7, 8, 9, 10]),
       200: dict(R=6000, mew=242.13, bas=[17.82, 246.00], grid=[6, 7, 8, 9, 10, 11, 12]),
       500: dict(R=5000, mew=564.98, bas=[19.65, 572.17], grid=[7, 8, 9, 10, 11, 12, 14, 16])}
out = {}
for p, c in CFG.items():
    g = []
    for i, nz in enumerate(c["grid"]):
        d = np.zeros(p); d[:nz] = 1 / np.sqrt(nz)
        seed = 90_000_000 + p * 1000 + i * 100
        base = ss_profile(0, [0.05], [c["mew"]], D, p=p, R=c["R"], seed0=seed, dvec=d)
        a = ss_profile(6, [0.05, 1, p], c["bas"], D, p=p, R=c["R"], seed0=seed, dvec=d)
        val = float(100 * (1 - np.mean(a / base)))
        g.append(val)
        print(f"p={p} PR={nz:3d}  {val:6.2f}%", flush=True)
    gg = np.array(g); pr = np.array(c["grid"], float)
    i = np.where(gg < 0)[0]
    be = float(np.exp(np.interp(0.0, [gg[i[0]], gg[i[0] - 1]], [np.log(pr[i[0]]), np.log(pr[i[0] - 1])]))) if len(i) and i[0] > 0 else float('nan')
    out[p] = dict(grid=c["grid"], gains=g, be=be)
    print(f"  -> p={p} break-even PR = {be:.2f}", flush=True)
    json.dump(out, open("refine.json", "w"), indent=1)
print("DONE")
