"""LAR-EWMA (Capizzi & Masarotto, 2011; for Sigma = I the maximum of standardised sums of the k
largest u_i) and a mixture EWMA in the spirit of Xie & Siegmund (2013),
sum_i ln{1 - p0 + p0 exp(u_i / 2)}.
Inputs: none (simulation).
Outputs: lar_mix.json
Paper: Section 4.3, Fig. 2, Table 2; Supplement S12.
"""
import json, numpy as np
exec(open('threshold_ewma.py').read().split('lims = {}')[0].replace('R, TMAX','R, TMAX').replace('def stat(name, u):','def _old(name,u):'))
R = 4000
rng = np.random.default_rng(5); z = np.zeros((20000, p))
for t in range(300): z = lam * rng.standard_normal((20000, p)) + (1 - lam) * z
U = np.sort(cst * z * z, 1)[:, ::-1].cumsum(1); MU, SD = U.mean(0), U.std(0)
def stat(name, u):
    kind, c = name
    if kind == 'lar': return ((np.sort(u, 1)[:, ::-1].cumsum(1) - MU) / SD).max(1)
    if kind == 'mix': return np.log1p(c * np.expm1(np.minimum(u, 600) / 2)).sum(1)
def first_alarm_ic(name, h, seed=11):
    rng = np.random.default_rng(seed); z = np.zeros((R, p)); fa = np.full(R, TMAX)
    for t in range(1, TMAX + 1):
        z = lam * rng.standard_normal((R, p)) + (1 - lam) * z; s = stat(name, cst * z * z); fa = np.where((s > h) & (fa == TMAX), t, fa)
        if (fa < TMAX).mean() > 0.997: break
    return fa.mean()
def calibrate(name, lo, hi):
    for _ in range(18):
        h = (lo + hi) / 2; a = first_alarm_ic(name, h); lo, hi = (h, hi) if a < 200 else (lo, h)
    return h, a
lims = {}
for name, lo, hi in [(('lar', 0), 0.5, 12.0), (('mix', 0.03), 0.1, 60.0), (('mix', 0.1), 0.1, 80.0), (('mix', 0.3), 0.1, 120.0)]:
    h, a = calibrate(name, lo, hi); lims[f'{name[0]} {name[1]}'] = (name[0], name[1], float(h), float(a)); print(name, round(h, 3), round(a, 1), flush=True)
charts = {'MEWMA': lambda u: u.sum(1) > LM, 'BAS-EWMA w1=0.40': lambda u: (u.max(1) > LB[0]) | (u.sum(1) > LB[1])}
for k, (kind, c, h, a) in lims.items(): charts[k] = (lambda kind, c, h: (lambda u: stat((kind, c), u) > h))(kind, c, h)
res = {}
for nz in (1, 2, 3, 5, 10, 30):
    res[nz] = ooc(charts, nz, 600 + nz); print(nz, {k: round(v, 2) for k, v in res[nz].items()}, flush=True)
    json.dump({'limits': lims, 'ss_0.75': res}, open('lar_mix.json', 'w'), indent=1)
print('DONE')
