"""Fits of the break-even and of the sparse-term limit against ln p and ln p + ln ln p, and the
adjustment of out-of-control ARLs to a common steady-state in-control ARL.
Inputs: be_wls.json, calib_err.json, limits.json, robust2.json, ss_checks.json
Outputs: cheap5.json
Paper: Sections 4.2-4.3; Supplement S3 and S20.
"""
import json, numpy as np
from scipy.stats import chi2
from scipy.optimize import curve_fit
out = {}
w = json.load(open('be_wls.json')); P = np.array([int(k) for k in w['points']]); B = np.array([v['be'] for v in w['points'].values()]); S = np.array([v['se'] for v in w['points'].values()])
def fit(f, p0):
    pr, cov = curve_fit(f, P, B, p0=p0, sigma=S, absolute_sigma=True, maxfev=20000); r = (B - f(P, *pr)) / S; c2 = float((r ** 2).sum()); df = len(P) - len(pr)
    return {'par': pr.tolist(), 'chi2': c2, 'df': df, 'p': float(chi2.sf(c2, df))}
out['be'] = {'a ln p + c': fit(lambda p, a, c: a * np.log(p) + c, [2, -2]), 'a ln p - b ln ln p + c': fit(lambda p, a, b, c: a * np.log(p) - b * np.log(np.log(p)) + c, [2, 1, -2]),
             'a p^g + c': fit(lambda p, a, g, c: a * p ** g + c, [3, 0.2, -2])}
L = json.load(open('limits.json'))['sparse_term_limits_h1']; pp = np.array([int(k) for k in L]); hh = np.array(list(L.values()))
X1 = np.c_[np.log(pp), np.ones(len(pp))]; X2 = np.c_[np.log(pp), np.log(np.log(pp)), np.ones(len(pp))]
for nm, X in (('ln p', X1), ('ln p + ln ln p', X2)):
    b, *_ = np.linalg.lstsq(X, hh, rcond=None); r = hh - X @ b; out.setdefault('h1', {})[nm] = {'coef': b.tolist(), 'rss': float(r @ r)}
# adjustment of the comparison designs to a common steady-state in-control ARL (elasticity e)
ss = json.load(open('ss_checks.json'))['ss_arl0']; rb = json.load(open('robust2.json')); ce = json.load(open('calib_err.json'))
eM = np.mean([v['elast'][0] for v in ce['BAS vs MEWMA, delta=0.75'].values()]); eB = np.mean([v['elast'][1] for v in ce['BAS vs MEWMA, delta=0.75'].values()])
E = {'MEWMA lam=0.05': eM, 'BAS-EWMA lam=0.05 w1=0.4': eB, 'top-r CUSUM r=5': 0.37, 'top-r CUSUM r=1': 0.37, 'VS-MEWMA lam=0.05': 0.37}
ref = ss['MEWMA lam=0.05'][1]; adj = {}
for n, (z0, s0) in ss.items():
    a = np.array([rb[n][f'0.75_{k}'] for k in (1, 2, 3, 5, 10, 30)]); f_ss = (ref / s0) ** E[n]; f_zs = (200 / z0) ** E[n]
    adj[n] = {'raw_mean': float(a.mean()), 'raw_worst': float(a.max()), 'ss_adj_mean': float(a.mean() * f_ss), 'ss_adj_worst': float(a.max() * f_ss), 'zs_adj_mean': float(a.mean() * f_zs), 'zs_adj_worst': float(a.max() * f_zs)}
out['table1_adjusted'] = adj; out['elasticities'] = {'MEWMA': eM, 'BAS': eB}
print(json.dumps(out, indent=1)); json.dump(out, open('cheap5.json', 'w'), indent=1)
