"""Tuned designs of every chart family at p = 30 (lambda, k, w1, LEWMA penalties) and their
steady-state ARL profiles over shift sizes and sparsity levels.
Inputs: bench_p2.json
Outputs: final.json
Paper: Section 4.3, Fig. 2 and Table 2.
"""
import json, numpy as np
from calib import calibrate, ss_profile
from engine import arl
from bas_ewma import calibrate_bas
R = 10000
VER = dict(R=50000, seed0=5_000_000)  # fresh in-control seeds for ARL0 verification
OUT = {}
def save(): json.dump(OUT, open("final.json", "w"), indent=1)

# ---------- Part B: p = 30, sparse vs dense shifts ----------
p = 30; D = [0.25, 0.5, 0.75, 1.0, 1.5]; NZ = [1, 3, 10, 30]
B = OUT.setdefault("p30", {"deltas": D, "nz": NZ, "bench": {}, "bas": {}})
for lam in [0.02, 0.03, 0.05, 0.075, 0.1]:
    c, _ = calibrate(0, [lam], lambda c: [c], 5, 90, p=p, R=R, iters=22)
    a0 = arl(0, [lam], [c], p=p, **VER)[0]
    B["bench"][f"MEWMA l={lam}"] = dict(lim=c, arl0=a0, ss=ss_profile(0, [lam], [c], D, p=p, R=R).tolist())
    print("MEWMA", lam, round(a0, 1), np.round(B["bench"][f"MEWMA l={lam}"]["ss"], 1), flush=True); save()
for k in [0.2, 0.3, 0.5]:
    c, _ = calibrate(1, [k], lambda c: [c], 1, 40, p=p, R=R, iters=22)
    a0 = arl(1, [k], [c], p=p, **VER)[0]
    B["bench"][f"MCUSUM k={k}"] = dict(lim=c, arl0=a0, ss=ss_profile(1, [k], [c], D, p=p, R=R).tolist())
    print("MCUSUM", k, round(a0, 1), np.round(B["bench"][f"MCUSUM k={k}"]["ss"], 1), flush=True); save()
for w1 in [0.02, 0.1, 0.25, 0.5, 0.75]:
    prm, lim, _ = calibrate_bas(0.05, [1, 30], p, R=R, w=[w1, 1 - w1])
    a0 = arl(6, prm, lim, p=p, **VER)[0]
    prof = {nz: ss_profile(6, prm, lim, D, p=p, R=R, nz=nz).tolist() for nz in NZ}
    B["bas"][f"w1={w1}"] = dict(prm=prm, lim=lim, arl0=a0, ss=prof)
    print("BAS w1", w1, round(a0, 1), {nz: np.round(v, 1).tolist() for nz, v in prof.items()}, flush=True); save()

# ---------- Part A: p = 2, multi-lambda MEWMA validated on fresh seeds ----------
p = 2; D = [0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0]
A = OUT.setdefault("p2", {"deltas": D, "charts": {}})
bench = json.load(open("bench_p2.json"))["charts"]
for lams, tilt in [((0.02, 0.075, 0.3), 0.5), ((0.03, 0.15), 0.5), ((0.02, 0.1), 0.0)]:
    base = [bench[f"MEWMA l={l}"]["lim"][0] for l in lams]
    lf = lambda c: [base[j] + c + tilt * (j / (len(lams) - 1)) for j in range(len(lams))]
    c, _ = calibrate(0, list(lams), lf, -2, 6, p=p, R=20000, iters=26)
    a0 = arl(0, list(lams), lf(c), p=p, **VER)[0]
    ss = ss_profile(0, list(lams), lf(c), D, p=p, R=20000, seed0=20_000_000).tolist()
    A["charts"][f"{lams} tilt={tilt}"] = dict(lim=lf(c), arl0=a0, ss=ss)
    print("multi", lams, round(a0, 1), np.round(ss, 1), flush=True); save()
for name in ["MEWMA l=0.05", "MEWMA l=0.1", "MCUSUM k=0.25", "MCUSUM k=0.2"]:
    v = bench[name]
    a0 = arl(v["kind"], v["prm"], v["lim"], p=p, **VER)[0]
    ss = ss_profile(v["kind"], v["prm"], v["lim"], D, p=p, R=20000, seed0=20_000_000).tolist()
    A["charts"][name] = dict(lim=v["lim"], arl0=a0, ss=ss)
    print(name, round(a0, 1), np.round(ss, 1), flush=True); save()
print("DONE")
