"""Summary of the random partitions of te_random_groups_eval2.py: mean gain, percentiles, Holm-
significant gains and losses, and the percentile of the engineering grouping.
Inputs: te_eval2_res.json, tg4_a.json, tg4_b.json
Outputs: te_groups4_holm.json
Paper: Supplement S11.
"""
import json, numpy as np
from scipy.stats import norm
R = json.load(open('tg4_a.json'))['random'] + json.load(open('tg4_b.json'))['random']
def holm(t):
    p = 2 * norm.sf(np.abs(t)); o = np.argsort(p); a = np.empty(len(p)); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (len(p) - r) * p[i])); a[i] = run
    return a < .05
E = json.load(open('te_eval2_res.json')); te = np.array(E['G-BAS std|paper']['t']); eg = 100 * (1 - E['G-BAS std|paper']['mean_ratio'])
eh = holm(te); out = {'n': len(R)}
for key in ('std', 'raw'):
    g = np.array([100 * (1 - r[key]['mean_ratio']) for r in R]); sl = np.array([int((holm(np.array(r[key]['t'])) & (np.array(r[key]['t']) > 0)).sum()) for r in R])
    fa = np.array([int((holm(np.array(r[key]['t'])) & (np.array(r[key]['t']) < 0)).sum()) for r in R])
    out[key] = {'mean': g.mean(), 'q05': np.quantile(g, .05), 'q95': np.quantile(g, .95), 'no_loss': 100 * (sl == 0).mean(), 'median_gains': float(np.median(fa))}
out['eng_std'] = {'gain': eg, 'pct': 100 * (np.array([100 * (1 - r['std']['mean_ratio']) for r in R]) <= eg).mean(), 'losses': int((eh & (te > 0)).sum()), 'gains': int((eh & (te < 0)).sum())}
o = out['std']; e = out['eng_std']
out['text'] = (f"{out['n']} random partitions with the engineering group sizes, standardised maximum, Holm adjustment over the 20 faults within each partition: mean gain {o['mean']:.1f}\\% (5--95\\%: {o['q05']:.1f} to {o['q95']:.1f}); {o['no_loss']:.0f}\\% of partitions have no significant loss; median number of significant gains {o['median_gains']:.0f}. "
               f"Engineering grouping: {e['gain']:.1f}\\% ({e['pct']:.0f}th percentile), {e['gains']} significant gains, {e['losses']} losses. Raw maximum, same rule: mean {out['raw']['mean']:.1f}\\%, {out['raw']['no_loss']:.0f}\\% without loss (\\texttt{{tg4\\_a.json}}, \\texttt{{tg4\\_b.json}}).")
out = json.loads(json.dumps(out, default=float)); print(json.dumps(out, indent=1)); json.dump(out, open('te_groups4_holm.json', 'w'), indent=1)
