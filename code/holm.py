"""Holm adjustment over the paired Tennessee Eastman tests under the matched protocol.
Inputs: rieth_pr.json, rieth_res_test_matched.json
Outputs: holm.json
Paper: Supplement S5.
"""
import json, numpy as np
from scipy.stats import norm, binomtest
tests = []
for src, lab in (('rieth_test.json', 'raw'), ('rieth_res_test_matched.json', 'res')):
    d = json.load(open(src))
    for k, row in d.items():
        for n in ('G-BAS', 'BAS-EWMA'):
            t = row[f'{n}|cond']['t']; tests.append((lab, n, int(k), t, 2 * norm.sf(abs(t))))
m = len(tests); order = np.argsort([x[4] for x in tests]); padj = np.empty(m); run = 0
for r, i in enumerate(order):
    run = max(run, min(1, (m - r) * tests[i][4])); padj[i] = run
out = {'m': m, 'fwer_|t|>3_bound': float(1 - (1 - 2 * norm.sf(3)) ** m), 'sig': {}}
for lab in ('raw', 'res'):
    for n in ('G-BAS', 'BAS-EWMA'):
        f = [tests[i][2] for i in range(m) if tests[i][0] == lab and tests[i][1] == n and padj[i] < 0.05 and tests[i][3] < 0]
        s = [tests[i][2] for i in range(m) if tests[i][0] == lab and tests[i][1] == n and padj[i] < 0.05 and tests[i][3] > 0]
        out['sig'][f'{lab}|{n}'] = {'faster': sorted(f), 'slower': sorted(s)}
# per-fault sign rule vs majority baseline (BAS-EWMA residuals)
pr = json.load(open('rieth_pr.json')) if True else None
out['binom_13of20_onesided_p'] = float(binomtest(13, 20, 0.5, alternative='greater').pvalue)
d = json.load(open('rieth_res_test_matched.json'))
out['BAS_res_faults_faster'] = int(sum(d[k]['BAS-EWMA|cond']['ratio'] < 1 for k in d))
print(json.dumps(out, indent=1)); json.dump(out, open('holm.json', 'w'), indent=1)
