"""Conditional in-control ARL over Phase I samples after matching the unconditional ARL to 200.

p=30, Sigma=I, lambda=0.05, sample covariance with Cholesky regression adjustment.
Usage: python cond_arl0.py [n ...]  (Phase I sample sizes; default 100 200)
Inputs: front.json
Outputs: cond_arl0.json
Paper: Section 3.1; Supplement S4.
"""
import json, numpy as np, sys
p, lam = 30, 0.05; c = (2 - lam) / lam; F = json.load(open('front.json'))
LM = np.array(F['MEWMA lam=0.05']['lim']); LB = np.array(F['BAS-EWMA lam=0.05 w1=0.4']['lim'])
SC = np.geomspace(1.0, 4.0, 57); B, RUN, TMAX = 200, 300, 3000
rng = np.random.default_rng(11); out = {}
for n in (int(a) for a in (sys.argv[1:] or ['100', '200'])):
    cm = np.zeros((B, len(SC))); cb = np.zeros((B, len(SC)))
    for b in range(B):
        Y = rng.standard_normal((n, p)); mu = Y.mean(0); A = np.linalg.inv(np.linalg.cholesky(np.cov(Y.T)))
        z = np.zeros((RUN, p)); fm = np.full((RUN, len(SC)), TMAX); fb = np.full((RUN, len(SC)), TMAX)
        for t in range(1, TMAX + 1):
            w = (rng.standard_normal((RUN, p)) - mu) @ A.T; z = lam * w + (1 - lam) * z; u = c * z * z
            tot = u.sum(1); mx = u.max(1)
            am = tot[:, None] > SC * LM[0]; ab = (mx[:, None] > SC * LB[0]) | (tot[:, None] > SC * LB[1])
            fm = np.where(am & (fm == TMAX), t, fm); fb = np.where(ab & (fb == TMAX), t, fb)
            if t % 500 == 0 and (fm < TMAX).all() and (fb < TMAX).all(): break
        cm[b] = fm.mean(0); cb[b] = fb.mean(0)
    res = {}
    for name, ca in (('MEWMA', cm), ('BAS-EWMA w1=0.40', cb)):
        j = int(np.argmin(np.abs(ca.mean(0) - 200))); v = ca[:, j]
        res[name] = {'scale': float(SC[j]), 'uncond': float(v.mean()), 'sd': float(v.std()), 'q10': float(np.quantile(v, .1)), 'P_below_150': float((v < 150).mean())}
    out[n] = res; print(n, res, flush=True); json.dump(out, open('cond_arl0.json', 'w'), indent=1)
