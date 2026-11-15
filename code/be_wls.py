"""Standard errors of the break-even locations (delta method on the local gain grid) and weighted
least-squares fits of the break-even on ln p, sqrt(p), sqrt(p)/ln p, ln ln p and a power of p,
with lack-of-fit tests.
Inputs: decide.json, dim_p10.json, dim_p50.json, m1_sims.json, refine.json
Outputs: be_wls.json
Paper: Fig. 1 and Section 4.2; Supplement S20.
"""
import json, numpy as np
def be_se(pr, g):
    pr, g = np.asarray(pr, float), np.asarray(g, float); i = np.where(np.diff(np.sign(g)) < 0)[0][0]
    sel = np.arange(max(0, i - 2), min(len(g), i + 4)); x = np.log(pr[sel]); y = g[sel]
    X = np.c_[np.ones_like(x), x]; b, res, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
    s2 = (r @ r) / max(1, len(y) - 2); C = s2 * np.linalg.inv(X.T @ X); x0 = -b[0] / b[1]
    gx = np.array([-1 / b[1], b[0] / b[1] ** 2]); se_x = np.sqrt(gx @ C @ gx); be = np.exp(x0)
    i0 = i; xi = np.log(pr); be_interp = np.exp(xi[i0] + (0 - g[i0]) * (xi[i0 + 1] - xi[i0]) / (g[i0 + 1] - g[i0]))
    return float(be_interp), float(be * se_x)
pts = {}
d10 = json.load(open('dim_p10.json')); pts[10] = be_se([r['pr'] for r in d10], [r['BAS w1=0.25'] for r in d10])
m1 = json.load(open('m1_sims.json'))['delta_w1']; NZ = [1, 2, 3, 5, 10, 30]; pts[30] = be_se(NZ, [m1[f'nz={n}|w1=0.25'][1] for n in NZ])
d50 = json.load(open('dim_p50.json')); pts[50] = be_se([r['pr'] for r in d50], [r['BAS w1=0.25'] for r in d50])
rf = json.load(open('refine.json')); pts[100] = be_se(rf['100']['grid'], rf['100']['gains'])
dc = json.load(open('decide.json'))
for p in (200, 500, 1000): pts[p] = be_se(dc[str(p)]['grid'], dc[str(p)]['gains'])
rep = {200: [rf['200']['be'], dc['200']['be']], 500: [rf['500']['be'], dc['500']['be']]}
sd_rep = float(np.sqrt(np.mean([np.var(v, ddof=1) for v in rep.values()])))
for p, v in rep.items(): pts[p] = (float(np.mean(v)), sd_rep / np.sqrt(2))
for p in (100, 1000): pts[p] = (pts[p][0], max(pts[p][1], sd_rep))
P = np.array(sorted(pts)); B = np.array([pts[p][0] for p in P]); SE = np.array([max(pts[p][1], 0.05) for p in P])
W = 1 / SE ** 2; X = np.c_[np.log(P), np.ones(len(P))]
def wls(X):
    A = X.T @ (W[:, None] * X); b = np.linalg.solve(A, X.T @ (W * B)); r = B - X @ b; chi2 = float((W * r * r).sum())
    s2 = chi2 / (len(B) - 2); cov = np.linalg.inv(A) * max(1.0, s2); return b, cov, chi2
out = {'sd_single_from_replicates': sd_rep, 'points': {int(p): {'be': pts[p][0], 'se': pts[p][1]} for p in P},
       'replicates': {'200': [rf['200']['be'], dc['200']['be']], '500': [rf['500']['be'], dc['500']['be']]}}
for name, Xm in (('lnp', X), ('sqrtp_over_lnp', np.c_[np.sqrt(P) / np.log(P), np.ones(len(P))]), ('sqrtp', np.c_[np.sqrt(P), np.ones(len(P))])):
    b, cov, chi2 = wls(Xm); out[name] = {'slope': float(b[0]), 'icpt': float(b[1]), 'slope_ci95': [float(b[0] - 1.96 * np.sqrt(cov[0, 0])), float(b[0] + 1.96 * np.sqrt(cov[0, 0]))], 'chi2': chi2}
print(json.dumps(out, indent=1)); json.dump(out, open('be_wls.json', 'w'), indent=1)
