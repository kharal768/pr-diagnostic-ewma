"""LEWMA (Zou & Qiu, 2009) designs over the penalty path for one lambda (command-line arguments),
calibrated and profiled; results added to final.json.
Usage: python lewma.py lambda q  (q = number of penalty levels)
Inputs: none (simulation).
Outputs: final.json
Paper: Section 4.3.
"""
import json, sys, numpy as np
from engine import lewma_path_samples, arl
from calib import calibrate, ss_profile
p = 30; R = 10000
lam = float(sys.argv[1]); q = int(sys.argv[2])
smp = lewma_path_samples(lam, q, p, 4000, int(10 / lam), max(1, int(1 / lam)), 50, 777)
E, S = smp.mean(0), smp.std(0)
prm = [lam] + E.tolist() + S.tolist()
c, _ = calibrate(8, prm, lambda c: [c], 0.0, 20.0, p=p, R=R, iters=24)
a0 = arl(8, prm, [c], p=p, R=50000, seed0=5_000_000)[0]
B = json.load(open("final.json"))["p30"]
prof = {nz: ss_profile(8, prm, [c], B["deltas"], p=p, R=R, nz=nz).tolist() for nz in B["nz"]}
print("LEWMA", lam, q, round(c, 3), round(a0, 1), {nz: np.round(v, 1).tolist() for nz, v in prof.items()}, flush=True)
O = json.load(open("final.json")); O["p30"].setdefault("lewma", {})[f"lam={lam} q={q}"] = dict(prm=prm, lim=c, arl0=a0, ss=prof)
json.dump(O, open("final.json", "w"), indent=1)
