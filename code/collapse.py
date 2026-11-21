"""Gain of BAS-EWMA against the participation ratio for 54 structured shift directions
(equal-loading anchors and one-variable / all-variable shifts under eight covariance
structures, in regression-adjusted coordinates), with the cubic fit in log PR.
Inputs: final.json
Outputs: collapse.json
Paper: Section 4.2; Supplement S1.
"""
import json, numpy as np
from calib import ss_profile

p, R = 30, 15000
D = [0.75, 1.0]
KS = [0, 6, 13, 20, 29]
I = np.arange(p)

def sigma(kind, r):
    if kind == 'ar1':    return r ** np.abs(np.subtract.outer(I, I))
    if kind == 'cs':     return (1 - r) * np.eye(p) + r * np.ones((p, p))
    if kind == 'block':  return np.where((I // 6)[:, None] == (I // 6)[None, :], r, 0.0) + (1 - r) * np.eye(p)
    if kind == 'factor':
        L = np.full((p, 1), np.sqrt(r)); S = L @ L.T; np.fill_diagonal(S, 1.0); return S

def directions():
    out = []
    for nz in (1, 2, 3, 5, 10, 30):          # Sigma = I anchors: PR is exactly nz
        d = np.zeros(p); d[:nz] = 1 / np.sqrt(nz)
        out.append(("identity", f"nz={nz}", d))
    for kind, r in [('ar1', 0.3), ('ar1', 0.6), ('ar1', 0.9), ('cs', 0.3), ('cs', 0.6),
                    ('cs', 0.9), ('block', 0.8), ('factor', 0.7)]:
        Li = np.linalg.inv(np.linalg.cholesky(sigma(kind, r)))
        for k in KS:
            d = Li @ np.eye(p)[k]; out.append((f"{kind} r={r}", f"1 var (k={k})", d / np.linalg.norm(d)))
        d = Li @ np.ones(p);       out.append((f"{kind} r={r}", "all 30 vars", d / np.linalg.norm(d)))
    return out

F = json.load(open("final.json"))["p30"]
REF = (0, [0.05], [F["bench"]["MEWMA l=0.05"]["lim"]])
CH = {"LEWMA l=0.05": (8, F["lewma"]["lam=0.05 q=30"]["prm"], [F["lewma"]["lam=0.05 q=30"]["lim"]]),
      "BAS w1=0.50":  (6, F["bas"]["w1=0.5"]["prm"], F["bas"]["w1=0.5"]["lim"]),
      "BAS w1=0.25":  (6, F["bas"]["w1=0.25"]["prm"], F["bas"]["w1=0.25"]["lim"])}

rows = []
for si, (struct, label, d) in enumerate(directions()):
    seed = 40_000_000 + si * 100_000
    base = ss_profile(*REF, D, p=p, R=R, seed0=seed, dvec=d)
    rec = {"structure": struct, "shift": label, "pr": float(1 / np.sum(d ** 4)), "mewma": base.mean()}
    for name, (kind, prm, lim) in CH.items():
        a = ss_profile(kind, prm, lim, D, p=p, R=R, seed0=seed, dvec=d)
        rec[name] = float(100 * (1 - np.mean(a / base)))
    rows.append(rec)
    print(f"{struct:14s} {label:14s} PR={rec['pr']:6.2f}  "
          f"LEWMA {rec['LEWMA l=0.05']:6.1f}%  BAS.50 {rec['BAS w1=0.50']:6.1f}%  BAS.25 {rec['BAS w1=0.25']:6.1f}%", flush=True)
    json.dump(rows, open("collapse.json", "w"), indent=1)
print("DONE")
