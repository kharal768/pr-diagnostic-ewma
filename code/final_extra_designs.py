"""Adds MCUSUM (k = 0.2, 0.3) and BAS-EWMA (lambda = 0.02) designs to final.json.
Inputs: none (simulation).
Outputs: final.json
Paper: Section 4.3.
"""
import json, numpy as np
from calib import calibrate, ss_profile
from engine import arl
from bas_ewma import calibrate_bas
OUT = json.load(open("final.json")); B = OUT["p30"]; D = B["deltas"]; p = 30; R = 10000
VER = dict(R=50000, seed0=5_000_000)
for k in [0.2, 0.3]:
    c, _ = calibrate(1, [k], lambda c: [c], 1, 150, p=p, R=R, iters=24)
    a0 = arl(1, [k], [c], p=p, **VER)[0]
    B["bench"][f"MCUSUM k={k}"] = dict(lim=c, arl0=a0, ss=ss_profile(1, [k], [c], D, p=p, R=R).tolist())
    print("MCUSUM", k, round(c, 2), round(a0, 1), np.round(B["bench"][f"MCUSUM k={k}"]["ss"], 1), flush=True)
    json.dump(OUT, open("final.json", "w"), indent=1)
for w1 in [0.1, 0.25]:
    prm, lim, _ = calibrate_bas(0.02, [1, 30], p, R=R, w=[w1, 1 - w1])
    a0 = arl(6, prm, lim, p=p, **VER)[0]
    prof = {nz: ss_profile(6, prm, lim, D, p=p, R=R, nz=nz).tolist() for nz in B["nz"]}
    B["bas"][f"lam=0.02 w1={w1}"] = dict(prm=prm, lim=lim, arl0=a0, ss=prof)
    print("BAS l=.02 w1", w1, round(a0, 1), {nz: np.round(v, 1).tolist() for nz, v in prof.items()}, flush=True)
    json.dump(OUT, open("final.json", "w"), indent=1)
print("DONE")
