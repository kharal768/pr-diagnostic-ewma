"""As pr_general.py at shift size delta=1.0.
Inputs: front.json, lar_mix.json, pr_general.json, threshold_ewma.json
Outputs: pr_general_d1.json
Paper: Section 4.2 (second shift size); Supplement S19.
"""
import json, numpy as np
p, lam, R, TMAX, TAU = 30, 0.05, 6000, 3000, 50; cst = (2 - lam) / lam
F = json.load(open('front.json')); LB = F['BAS-EWMA lam=0.05 w1=0.4']['lim']; LM = F['MEWMA lam=0.05']['lim'][0]
T = json.load(open('threshold_ewma.json'))['limits']; M = json.load(open('lar_mix.json'))['limits']
hs = T['soft c=2.0'][2]; hm = M['mix 0.3'][2]; hl = M['lar 0'][2]
rng = np.random.default_rng(5); z = np.zeros((20000, p))
for t in range(300): z = lam * rng.standard_normal((20000, p)) + (1 - lam) * z
U = np.sort(cst * z * z, 1)[:, ::-1].cumsum(1); MU, SD = U.mean(0), U.std(0)
charts = {'MEWMA': lambda u: u.sum(1) > LM, 'BAS': lambda u: (u.max(1) > LB[0]) | (u.sum(1) > LB[1]),
          'mix0.3': lambda u: np.log1p(0.3 * np.expm1(np.minimum(u, 600) / 2)).sum(1) > hm,
          'soft2': lambda u: np.maximum(u - 2, 0).sum(1) > hs,
          'LAR': lambda u: ((np.sort(u, 1)[:, ::-1].cumsum(1) - MU) / SD).max(1) > hl}
def run(d, delta, seed):
    rng = np.random.default_rng(seed); z = np.zeros((R, p)); out = {k: np.full(R, -1.0) for k in charts}; alive = {k: np.ones(R, bool) for k in charts}
    for t in range(1, TAU + TMAX + 1):
        x = rng.standard_normal((R, p)) + (delta * d if t > TAU else 0); z = lam * x + (1 - lam) * z; u = cst * z * z
        for k, f in charts.items():
            al = f(u)
            if t <= TAU: alive[k] &= ~al
            else: out[k][al & alive[k] & (out[k] < 0)] = t - TAU
        if t > TAU + 100 and all(((out[k] > 0) | ~alive[k]).all() for k in charts): break
    res = {}
    for k in charts:
        v = out[k][alive[k] & (out[k] > 0)]; res[k] = (float(v.mean()), float(v.std() / np.sqrt(len(v))))
    return res
res = {}
res['ss_arl0'] = json.load(open('pr_general.json'))['ss_arl0']
res['structured'] = []
for k in (1, 2, 3, 5, 8, 10, 15, 30):
    d = np.zeros(p); d[:k] = 1 / np.sqrt(k); r = run(d, 1.0, 1800 + k); res['structured'].append({'pr': float(k), 'arl': r}); print(k, {a: round(b[0], 2) for a, b in r.items()}, flush=True)
    json.dump(res, open('pr_general_d1.json', 'w'), indent=1)
rng = np.random.default_rng(2026); res['random'] = []
for i in range(48):
    k = int(np.clip(np.round(np.exp(rng.uniform(0, np.log(30)))), 1, 30)); alpha = rng.choice([0.2, 0.5, 1.0, 3.0])
    d = np.zeros(p); idx = rng.choice(p, k, replace=False); d[idx] = rng.gamma(alpha, 1.0, k) + 1e-6; d /= np.linalg.norm(d)
    r = run(d, 1.0, 19000 + i); res['random'].append({'pr': float(1 / (d ** 4).sum()), 'arl': r}); print(i, round(1 / (d ** 4).sum(), 2), flush=True)
    json.dump(res, open('pr_general_d1.json', 'w'), indent=1)
print('DONE')
