"""Calibrated limit of a pure maximum-EWMA term against p for several lambda, h1 = 2 ln p + c.
Inputs: none (simulation).
Outputs: lamscale.json
Paper: Section 6 (limit growth approximately as ln p).
"""
import json, numpy as np
from engine import arl
from calib import calibrate

ARL0 = 200.0
LAMS = [0.02, 0.05, 0.10, 0.20]
PS = [10, 60, 300, 1000]
HI = {10: 40.0, 60: 50.0, 300: 60.0, 1000: 70.0}
res = {}
for lam in LAMS:
    for p in PS:
        R = 10000 if p <= 300 else 5000
        c, _ = calibrate(6, [lam, 1], lambda c: [c], 1.0, HI[p], p=p, R=R, iters=24, maxT=4000)
        a0 = arl(6, [lam, 1], [c], p=p, R=30000, seed0=5_000_000)[0]
        res[f"{lam}_{p}"] = dict(lam=lam, p=p, h1=c, arl0=a0)
        print(f"lam={lam:4.2f} p={p:5d}  h1={c:6.2f}  ARL0={a0:6.1f}   h1-2lnp={c-2*np.log(p):6.2f}", flush=True)
        json.dump(res, open("lamscale.json", "w"), indent=1)
# consistency check: the sparse term inside BAS gets 25% of the budget, i.e. ARL0 ~ 800
c8, _ = calibrate(6, [0.05, 1], lambda c: [c], 1.0, 70.0, p=1000, R=5000, iters=24, maxT=6000, target=800.0)
print(f"\ncheck: max-only at ARL0=800, lam=0.05, p=1000 -> h1={c8:.2f}  (BAS sparse limit was 20.93)", flush=True)
res["check_arl800"] = c8
json.dump(res, open("lamscale.json", "w"), indent=1)
print("DONE")
