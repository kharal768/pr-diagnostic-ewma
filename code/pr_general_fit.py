"""Cubic fits in log PR for the four charts of pr_general.py, adjusted to MEWMA's steady-state
in-control ARL (elasticity 0.37), and their break-even points.
Inputs: pr_general.json
Outputs: pr_general_fit.json
Paper: Section 4.2 (chart-specific break-even); Supplement S9.
"""
import json, numpy as np
D = json.load(open('pr_general.json')); ch = ['BAS', 'mix0.3', 'soft2', 'LAR']; E = 0.37
a0 = {k: D['ss_arl0'][k][0] for k in D['ss_arl0']}
def gain(r, k, adj): 
    v = r[k][0] * ((a0['MEWMA'] / a0[k]) ** E if adj else 1); return 100 * (1 - v / r['MEWMA'][0])
out = {'ss_arl0': a0, 'E': E}
for adj in (False, True):
    for k in ch:
        xs = np.log([s['pr'] for s in D['structured']]); ys = np.array([gain(s['arl'], k, adj) for s in D['structured']])
        b = np.polyfit(xs, ys, 3); xr = np.log([s['pr'] for s in D['random']]); yr = np.array([gain(s['arl'], k, adj) for s in D['random']])
        e = yr - np.polyval(b, xr); be_ = np.exp([r for r in np.roots(b) if abs(r.imag) < 1e-9 and 0 <= r.real <= np.log(30)][0].real) if any(abs(r.imag) < 1e-9 and 0 <= r.real <= np.log(30) for r in np.roots(b)) else None
        mid = (np.exp(xr) >= 2) & (np.exp(xr) <= 8); sign = float((np.sign(yr[mid]) == np.sign(np.polyval(b, xr[mid]))).mean())
        se = np.mean([100 * np.hypot(s['arl'][k][1] / s['arl'][k][0], s['arl']['MEWMA'][1] / s['arl']['MEWMA'][0]) for s in D['random']])
        st = np.array([gain(s['arl'], k, adj) for s in D['structured']])
        out[f'{k}|adj={adj}'] = {'rmse_oos': float(np.sqrt((e ** 2).mean())), 'mc_se': float(se), 'breakeven': be_, 'sign_2to8': sign, 'n_2to8': int(mid.sum()),
                                 'mean_struct': float(np.mean([s['arl'][k][0] * ((a0['MEWMA'] / a0[k]) ** E if adj else 1) for s in D['structured']])), 'worst_struct': float(max(s['arl'][k][0] * ((a0['MEWMA'] / a0[k]) ** E if adj else 1) for s in D['structured']))}
out['MEWMA'] = {'mean_struct': float(np.mean([s['arl']['MEWMA'][0] for s in D['structured']])), 'worst_struct': float(max(s['arl']['MEWMA'][0] for s in D['structured']))}
print(json.dumps(out, indent=1)); json.dump(out, open('pr_general_fit.json', 'w'), indent=1)
