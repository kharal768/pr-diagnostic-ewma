"""Prediction errors of shape summaries (participation ratio, largest loading, Hill numbers) for
the four charts of pr_general.py on the 48 random directions.
Inputs: pr_general.json
Outputs: shape_general.json
Paper: Table 1; Supplement S14.
"""
import json, numpy as np
D = json.load(open('pr_general.json')); p = 30; E = 0.37; a0 = {k: v[0] for k, v in D['ss_arl0'].items()}
rng = np.random.default_rng(2026); dirs = []
for i in range(48):
    k = int(np.clip(np.round(np.exp(rng.uniform(0, np.log(30)))), 1, 30)); alpha = rng.choice([0.2, 0.5, 1.0, 3.0])
    d = np.zeros(p); idx = rng.choice(p, k, replace=False); d[idx] = rng.gamma(alpha, 1.0, k) + 1e-6; d /= np.linalg.norm(d); dirs.append(d)
for d, r in zip(dirs, D['random']): assert abs(1 / (d ** 4).sum() - r['pr']) < 1e-6
def summ(d):
    w = d ** 2; w = w[w > 0]
    return {'PR': 1 / (w ** 2).sum(), 'max loading': 1 / w.max(), 'Hill-1 (entropy)': np.exp(-(w * np.log(w)).sum()), 'l1/l2': (np.abs(d).sum()) ** 2}
S = [summ(d) for d in dirs]; names = list(S[0])
def gain(r, k): return 100 * (1 - r[k][0] * (a0['MEWMA'] / a0[k]) ** E / r['MEWMA'][0])
out = {'pr_quantiles': np.quantile([s['PR'] for s in S], [0, .25, .5, .75, 1]).tolist(), 'n_pr_2_8': int(sum(2 <= s['PR'] <= 8 for s in S))}
for ch in ['BAS', 'mix0.3', 'soft2', 'LAR']:
    xs = np.log([s['pr'] for s in D['structured']]); ys = [gain(s['arl'], ch) for s in D['structured']]; b = np.polyfit(xs, ys, 3)
    y = np.array([gain(r['arl'], ch) for r in D['random']]); res = {}
    for n in names:
        x = np.log([s[n] for s in S]); pred = np.polyval(b, x); mid = np.array([2 <= s['PR'] <= 8 for s in S])
        res[n] = {'rmse': float(np.sqrt(((y - pred) ** 2).mean())), 'sign': int((np.sign(y[mid]) == np.sign(pred[mid])).sum())}
    be = [r.real for r in np.roots(b) if abs(r.imag) < 1e-9 and 0 <= r.real <= np.log(30)][0]
    res['rule PR<break-even'] = {'sign': int((np.sign(y[mid]) == np.sign(be - np.log([s['PR'] for s in S])[mid])).sum())}
    res['always gain sign of majority'] = {'sign': int(max((y[mid] > 0).sum(), (y[mid] <= 0).sum()))}
    out[ch] = res
# sensitivity to elasticity
for e in (0.2, 0.5):
    out[f'mean_e{e}'] = {ch: float(np.mean([s['arl'][ch][0] * (a0['MEWMA'] / a0[ch]) ** e for s in D['structured']])) for ch in ['MEWMA', 'BAS', 'mix0.3', 'soft2', 'LAR']}
# paired SE mixture vs MEWMA, BAS (independence-across-levels approx using per-level SEs)
for c2 in ['MEWMA', 'BAS']:
    dif = [s['arl']['mix0.3'][0] - s['arl'][c2][0] for s in D['structured']]; se = np.sqrt(sum(s['arl']['mix0.3'][1] ** 2 + s['arl'][c2][1] ** 2 for s in D['structured'])) / len(dif)
    out[f'mix_minus_{c2}'] = (float(np.mean(dif)), float(se))
print(json.dumps(out, indent=1)); json.dump(out, open('shape_general.json', 'w'), indent=1)
