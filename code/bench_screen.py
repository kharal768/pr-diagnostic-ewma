"""Calibrated limits for the competing sparsity-adaptive charts at p = 30, used as the starting
designs of the benchmark.
Usage: python bench_screen.py mode
Inputs: bench_p2.json
Outputs: screen_{mode}.json
Paper: Section 4.3.
"""
import json, sys, itertools, numpy as np
from calib import calibrate, ss_profile

B = json.load(open("bench_p2.json"))
D = B["deltas"]
env = np.min([v["ss"] for v in B["charts"].values()], axis=0)
SMALL = slice(0, 5)  # delta <= 1

def score(ss):
    r = np.asarray(ss) / env
    return r[SMALL].mean(), r.mean(), r.max()

def show(name, ss):
    s, a, m = score(ss)
    print(f"{name:40s} small={s:.3f} all={a:.3f} max={m:.3f}  ", np.round(ss, 1), flush=True)

for name, v in B["charts"].items():
    if name in ("MEWMA l=0.05", "MCUSUM k=0.2", "MCUSUM k=0.25", "MEWMA l=0.1"):
        show(name, v["ss"])

out = {}
mode = sys.argv[1]
if mode == "acusum":
    for lh, dmin in itertools.product([0.05, 0.1, 0.2], [0.25, 0.375, 0.5]):
        c, a = calibrate(2, [lh, dmin], lambda c: [c], 0.5, 15, R=10000, iters=22)
        ss = ss_profile(2, [lh, dmin], [c], D, R=10000)
        show(f"ACUSUM lh={lh} dmin={dmin} h={c:.2f}", ss); out[f"{lh},{dmin}"] = [c, ss.tolist()]
elif mode == "multiewma":
    # two/three parallel MEWMAs; split of the false-alarm budget controlled by ratio r of limits' quantile
    for lams in [(0.02, 0.1), (0.03, 0.15), (0.02, 0.2), (0.03, 0.1, 0.3), (0.02, 0.075, 0.3)]:
        base = []
        for lam in lams:  # single-chart limit giving ARL0=A for A grid, approximated by scaling
            base.append(B["charts"].get(f"MEWMA l={lam}", {"lim": [None]})["lim"][0])
        # search: limits h_j = base_j + c + s_j where s shifts budget toward small lambda
        for tilt in [0.0, 0.5, 1.0]:
            def lf(c, lams=lams, base=base, tilt=tilt):
                n = len(lams)
                return [base[j] + c + tilt * (j / (n - 1)) for j in range(n)]
            c, a = calibrate(0, list(lams), lf, -2, 6, R=10000, iters=22)
            ss = ss_profile(0, list(lams), lf(c), D, R=10000)
            show(f"MultiEWMA {lams} tilt={tilt}", ss); out[f"{lams},{tilt}"] = [lf(c), ss.tolist()]
json.dump(out, open(f"screen_{mode}.json", "w"))
