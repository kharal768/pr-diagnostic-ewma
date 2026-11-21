"""Adds MCUSUM designs with k = 1.5 and 2.0 to final.json.
Inputs: none (simulation).
Outputs: final.json
Paper: Section 4.3 (MCUSUM reference value range).
"""
import json, numpy as np
from calib import calibrate, ss_profile
from engine import arl
OUT = json.load(open("final.json")); B = OUT["p30"]; D = B["deltas"]; p = 30; R = 10000
for k in [1.5, 2.0]:
    c, _ = calibrate(1, [k], lambda c: [c], 0.5, 150, p=p, R=R, iters=24)
    a0 = arl(1, [k], [c], p=p, R=50000, seed0=5_000_000)[0]
    B["bench"][f"MCUSUM k={k}"] = dict(lim=c, arl0=a0, ss=ss_profile(1, [k], [c], D, p=p, R=R).tolist())
    print("MCUSUM", k, round(c, 2), round(a0, 1), np.round(B["bench"][f"MCUSUM k={k}"]["ss"], 1), flush=True)
    json.dump(OUT, open("final.json", "w"), indent=1)
print("DONE")
