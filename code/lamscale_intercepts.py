"""Intercepts c(lambda) of h1 = 2 ln p + c for a pure maximum-EWMA term at in-control ARL 200.
Inputs: none (simulation).
Outputs: lamscale.json
Paper: Section 6 (limit growth approximately as ln p).
"""
import json, os, numpy as np
from engine import arl
from calib import calibrate

LAMS = [0.02, 0.05, 0.10, 0.20]
PS = [10, 60, 300]
HI = {10: 40.0, 60: 50.0, 300: 60.0}
res = json.load(open("lamscale.json")) if os.path.exists("lamscale.json") else {}
for lam in LAMS:
    for p in PS:
        k = f"{lam}_{p}"
        if k in res:
            print(f"lam={lam:4.2f} p={p:5d}  h1={res[k]['h1']:6.2f}  (cached)   "
                  f"h1-2lnp={res[k]['h1'] - 2 * np.log(p):6.2f}", flush=True)
            continue
        c, _ = calibrate(6, [lam, 1], lambda c: [c], 1.0, HI[p], p=p, R=8000, iters=22, maxT=3000)
        a0 = arl(6, [lam, 1], [c], p=p, R=20000, seed0=5_000_000, maxT=8000)[0]
        res[k] = dict(lam=lam, p=p, h1=c, arl0=a0)
        print(f"lam={lam:4.2f} p={p:5d}  h1={c:6.2f}  ARL0={a0:6.1f}   "
              f"h1-2lnp={c - 2 * np.log(p):6.2f}", flush=True)
        json.dump(res, open("lamscale.json", "w"), indent=1)
print("DONE")
