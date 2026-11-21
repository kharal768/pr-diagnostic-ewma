"""Gain of BAS-EWMA over MEWMA against the participation ratio at one dimension p (command-line
argument) and its break-even.
Usage: python breakeven_by_dimension.py p
Inputs: none (simulation).
Outputs: dim_p{p}.json
Paper: Fig. 1 and Section 4.2 (break-even from p=10 to 1000).
"""
import json, sys, numpy as np
from engine import arl, lewma_path_samples
from calib import calibrate, ss_profile
from bas_ewma import calibrate_bas

p = int(sys.argv[1])
R = 10000 if p <= 30 else (8000 if p <= 100 else 5000)
D = [0.75, 1.0]
NZ = {10: [1, 2, 3, 4, 5, 7, 10], 30: [1, 2, 3, 4, 5, 7, 10, 15, 30],
      50: [1, 2, 3, 4, 5, 7, 10, 20, 50],
      100: [1, 2, 3, 5, 8, 15, 30, 60, 100],
      200: [1, 2, 3, 5, 8, 11, 15, 25, 60, 200],
      500: [1, 2, 4, 8, 12, 16, 22, 30, 80, 500]}[p]

hi = {10: 40.0, 30: 90.0, 50: 140.0, 100: 220.0, 200: 400.0, 500: 900.0}[p]
MT = 20000 if p <= 100 else 3000
cm, _ = calibrate(0, [0.05], lambda c: [c], 1, hi, p=p, R=R, iters=22, maxT=MT)
a0m = arl(0, [0.05], [cm], p=p, R=40000, seed0=5_000_000)[0]
print(f"p={p} MEWMA lim={cm:.2f} ARL0={a0m:.1f}", flush=True)

prm_b, lim_b, _ = calibrate_bas(0.05, [1, p], p, R=R, w=[0.25, 0.75], maxT=MT)
a0b = arl(6, prm_b, lim_b, p=p, R=40000, seed0=5_000_000)[0]
print(f"p={p} BAS   lim={np.round(lim_b,2)} ARL0={a0b:.1f}", flush=True)

CH = {"BAS w1=0.25": (6, prm_b, lim_b)}
if p <= 30:
    smp = lewma_path_samples(0.05, p, p, 4000, 200, 20, 50, 777)
    prm_l = [0.05] + smp.mean(0).tolist() + smp.std(0).tolist()
    cl, _ = calibrate(8, prm_l, lambda c: [c], 0.0, 20.0, p=p, R=R, iters=22)
    a0l = arl(8, prm_l, [cl], p=p, R=40000, seed0=5_000_000)[0]
    print(f"p={p} LEWMA lim={cl:.2f} ARL0={a0l:.1f}", flush=True)
    CH["LEWMA l=0.05"] = (8, prm_l, [cl])

rows = []
for i, nz in enumerate(NZ):
    d = np.zeros(p); d[:nz] = 1 / np.sqrt(nz)
    seed = 70_000_000 + i * 100_000
    base = ss_profile(0, [0.05], [cm], D, p=p, R=R, seed0=seed, dvec=d)
    rec = {"p": p, "nz": nz, "pr": float(nz), "frac": nz / p}
    for name, (k, pr_, li) in CH.items():
        a = ss_profile(k, pr_, li, D, p=p, R=R, seed0=seed, dvec=d)
        rec[name] = float(100 * (1 - np.mean(a / base)))
    rows.append(rec)
    print(f"  p={p} PR={nz:4d} (={nz/p:5.2f}p)  " +
          "  ".join(f"{n} {rec[n]:6.1f}%" for n in CH), flush=True)
    json.dump(rows, open(f"dim_p{p}.json", "w"), indent=1)
print("DONE")
