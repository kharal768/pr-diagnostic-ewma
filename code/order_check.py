"""Dependence of the participation ratio of a physical shift, and of the predicted gain, on the
order of the variables under regression adjustment; PR under the symmetric whitening
Sigma^{-1/2}.
Inputs: collapse.json
Outputs: order_check.json
Paper: Section 2.5; Supplement S2 and S20.
"""
import json, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('c', 'collapse.py')
src = open('collapse.py').read(); src = src[:src.index('F = json.load')]; ns = {}; exec(src, ns)
sigma, p = ns['sigma'], ns['p']
C = json.load(open('collapse.json')); x = np.log([c['pr'] for c in C]); y = np.array([c['BAS w1=0.50'] for c in C])
b = np.polyfit(x, y, 3); g = lambda pr: np.polyval(b, np.log(pr))
PR = lambda d: 1 / np.sum((d / np.linalg.norm(d)) ** 4)
rng = np.random.default_rng(1); out = {}
for kind, r in [('ar1', 0.6), ('ar1', 0.9), ('cs', 0.6), ('cs', 0.9), ('block', 0.8), ('factor', 0.7)]:
    S = sigma(kind, r); w, V = np.linalg.eigh(S); Sih = V @ np.diag(w ** -0.5) @ V.T
    for shift, mu in (('1 var (k=13)', np.eye(p)[13]), ('all 30 vars', np.ones(p))):
        prs = []
        for _ in range(500):
            o = rng.permutation(p); L = np.linalg.cholesky(S[np.ix_(o, o)]); prs.append(PR(np.linalg.solve(L, mu[o])))
        prs = np.array(prs); gs = g(prs)
        out[f'{kind} r={r} | {shift}'] = {'PR_q05_med_q95': np.quantile(prs, [.05, .5, .95]).round(2).tolist(),
                                          'gain_q05_med_q95': np.quantile(gs, [.05, .5, .95]).round(1).tolist(), 'PR_sym': round(float(PR(Sih @ mu)), 2), 'gain_sym': round(float(g(PR(Sih @ mu))), 1)}
for k, v in out.items(): print(f'{k:28s}', v)
json.dump(out, open('order_check.json', 'w'), indent=1)
