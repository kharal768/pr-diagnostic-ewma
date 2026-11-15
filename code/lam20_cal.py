"""Calibration of BAS-EWMA and MEWMA at lambda = 0.20 for p = 10, 100 and 1000.
Inputs: none (simulation).
Outputs: lam20_lims.json
Paper: Section 4.2 (slope at lambda = 0.20).
"""
import json, numpy as np
from engine import arl
from calib import calibrate
from bas_ewma import design

LAM, P = 0.20, 1000
out = {}

cm, _ = calibrate(0, [LAM], lambda c: [c], 1100.0, 1145.0, p=P, R=2500, iters=13, maxT=1000)
a0m = arl(0, [LAM], [cm], p=P, R=8000, seed0=5_000_000, maxT=3000)[0]
print(f"MEWMA lim={cm:.2f}  ARL0={a0m:.1f}", flush=True)
out["mewma"] = cm; out["arl0_mewma"] = float(a0m)
json.dump(out, open("lam20_lims.json", "w"), indent=1)

lf = design(LAM, [1, P], P, R=1500, w=[0.25, 0.75])
cb, _ = calibrate(6, [LAM, 1, P], lf, 1.5, 3.5, p=P, R=2500, iters=13, maxT=1000)
lim_b = lf(cb)
a0b = arl(6, [LAM, 1, P], lim_b, p=P, R=8000, seed0=5_000_000, maxT=3000)[0]
print(f"BAS   lim={np.round(lim_b, 2)}  ARL0={a0b:.1f}", flush=True)
print(f"sparse limit minus 2 ln p = {lim_b[0] - 2 * np.log(P):.2f}", flush=True)
out["bas"] = [float(x) for x in lim_b]; out["arl0_bas"] = float(a0b)
out["offset"] = float(lim_b[0] - 2 * np.log(P))
json.dump(out, open("lam20_lims.json", "w"), indent=1)
print("CALIBRATION DONE")
