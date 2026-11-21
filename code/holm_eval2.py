"""Holm adjustment over the paired tests of te_eval2.py (raw and residuals, two charts, 20 faults).
Inputs: te_eval2_raw.json, te_eval2_res.json
Outputs: holm2.json
Paper: Supplement S5.
"""
import json, numpy as np
from scipy.stats import norm
A = {'raw': json.load(open('te_eval2_raw.json')), 'res': json.load(open('te_eval2_res.json'))}
tests = []
for d, J in A.items():
    for n in ('G-BAS std', 'BAS-EWMA'):
        for k, t in enumerate(J[f'{n}|paper']['t']): tests.append((d, n, k + 1, t, 2 * norm.sf(abs(t))))
m = len(tests); o = np.argsort([x[4] for x in tests]); padj = np.empty(m); run = 0
for r, i in enumerate(o): run = max(run, min(1, (m - r) * tests[i][4])); padj[i] = run
out = {}
for d, J in A.items():
    for n in ('G-BAS std', 'BAS-EWMA'):
        idx = [i for i in range(m) if tests[i][0] == d and tests[i][1] == n]; P = J[f'{n}|paper']
        out[f'{d}|{n}'] = {'mean_pct': 100 * (1 - P['mean_ratio']), 'worst_pct': 100 * (P['worst'] - 1),
            'faster': [tests[i][2] for i in idx if padj[i] < .05 and tests[i][3] < 0], 'slower': [tests[i][2] for i in idx if padj[i] < .05 and tests[i][3] > 0],
            'holm_sig': [bool(padj[i] < .05) for i in idx]}
        print(d, n, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out[f'{d}|{n}'].items() if k != 'holm_sig'})
json.dump(out, open('holm2.json', 'w'), indent=1)
