"""In-control ARL of MEWMA under AR(1) data with independent-data limits (phi = 0 and 0.3).

p=30, lambda=0.05.
Inputs: front.json
Outputs: ar1_check.json
Paper: Section 3.1; Supplement S14.
"""
import json, numpy as np
p, lam, R = 30, 0.05, 4000; cst = (2 - lam) / lam; LM = json.load(open('front.json'))['MEWMA lam=0.05']['lim'][0]
out = {}
for phi in (0.0, 0.3):
    rng = np.random.default_rng(3); e = np.zeros((R, p)); z = np.zeros((R, p)); fa = np.full(R, 5000)
    for t in range(1, 5001):
        e = phi * e + np.sqrt(1 - phi ** 2) * rng.standard_normal((R, p)); z = lam * e + (1 - lam) * z
        a = (cst * (z * z).sum(1) > LM) & (fa == 5000); fa[a] = t
        if (fa < 5000).all(): break
    out[phi] = (float(fa.mean()), float(fa.std() / np.sqrt(R)))
print(out); json.dump(out, open('ar1_check.json', 'w'))
