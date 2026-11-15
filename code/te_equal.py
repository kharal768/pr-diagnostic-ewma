"""Primary Tennessee Eastman protocol: every chart's limits set to an in-control ARL of exactly
200 on the fault-free testing runs; delay of the first alarm at or after the fault over all 500
runs; paired t-tests on log delays with Holm adjustment over 80 comparisons.
Usage: python te_equal.py <cache folder> <output.json>
Inputs: cache folder (data/te_cache or data/te_cache_var10) given on the command line.
Outputs: <output.json> (command-line argument)
Paper: Fig. 5, Table 3 and Section 5.2; Supplement S8.
"""
import json, sys, numpy as np, importlib.util
from scipy.stats import norm
from te_lib import *
spec = importlib.util.spec_from_file_location('rieth', 'rieth_res.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
BASE, OUTF = sys.argv[1], sys.argv[2]; Rcor = np.load(f'{BASE}/Rcor.npy'); ENG = [list(g) for g in R.GROUPS.values()]
tot = {s: np.asarray(load(BASE, s, 'tot'), np.float64) for s in SETS}
first = {'G-BAS': {s: gstat(BASE, s, ENG, Rcor, True) for s in SETS}, 'BAS-EWMA': {s: np.asarray(load(BASE, s, 'umax'), np.float64) for s in SETS}}
W1 = {'G-BAS': 0.3, 'BAS-EWMA': 0.4}
lim = {'MEWMA': calibrate(None, tot['fftest'], None)}
for n in first: lim[n] = calibrate(first[n]['fftest'], tot['fftest'], W1[n])
def d_all(f, t, l, on=160):
    post = alarm(f, t, l)[:, on:]; return np.where(post.any(1), np.argmax(post, 1) + 1, post.shape[1] + 1).astype(float)
def d_cond(f, t, l, on=160):
    al = alarm(f, t, l); post = al[:, on:]; return np.where(post.any(1), np.argmax(post, 1) + 1, post.shape[1] + 1).astype(float), ~al[:, :on].any(1)
out = {'arl0_test': {n: float(first_alarm(alarm(None if n == 'MEWMA' else first[n]['fftest'], tot['fftest'], lim[n])).mean()) for n in lim}, 'rows': {}, 'curves': {}}
for k in list(range(1, 21)) + ['ff']:
    s = f'f{k}' if k != 'ff' else 'fftest'; dm = d_all(None, tot[s], lim['MEWMA']); dmc, cm = d_cond(None, tot[s], lim['MEWMA'])
    for n in first:
        dx = d_all(first[n][s], tot[s], lim[n]); ld = np.log(dx) - np.log(dm)
        dxc, cx = d_cond(first[n][s], tot[s], lim[n]); m = cm & cx; ldc = np.log(dxc[m]) - np.log(dmc[m])
        out['rows'][f'{k}|{n}'] = {'ratio_all': float(dx.mean() / dm.mean()), 'geo_all': float(np.exp(ld.mean())), 't_log_all': float(ld.mean() / (ld.std(ddof=1) / np.sqrt(len(ld)))) if ld.std() > 0 else 0.0,
                                   'ratio_cond': float(dxc[m].mean() / dmc[m].mean()), 't_log_cond': float(ldc.mean() / (ldc.std(ddof=1) / np.sqrt(m.sum()))) if ldc.std() > 0 else 0.0, 'n_cond': int(m.sum())}
    if k in (3, 9, 15, 'ff'):
        for n, f in (('MEWMA', None), ('G-BAS', first['G-BAS'][s]), ('BAS-EWMA', first['BAS-EWMA'][s])):
            d = d_all(f, tot[s], lim[n]); out['curves'][f'{k}|{n}'] = [float((d <= L).mean()) for L in (5, 10, 20, 50, 100, 200, 400)]
keys = [(k, n) for k in range(1, 21) for n in first]; t = np.array([out['rows'][f'{k}|{n}']['t_log_all'] for k, n in keys]); pv = 2 * norm.sf(np.abs(t))
o = np.argsort(pv); padj = np.empty(len(pv)); run = 0
for r, i in enumerate(o): run = max(run, min(1, (len(pv) - r) * pv[i])); padj[i] = run
for j, (k, n) in enumerate(keys): out['rows'][f'{k}|{n}']['holm40'] = float(padj[j])
for n in first:
    g = np.array([out['rows'][f'{k}|{n}']['ratio_all'] for k in range(1, 21)]); gc = np.array([out['rows'][f'{k}|{n}']['ratio_cond'] for k in range(1, 21)])
    out[f'summary|{n}'] = {'gain_all_pct': float(100 * (1 - np.exp(np.log(g).mean()))), 'gain_cond_pct': float(100 * (1 - np.exp(np.log(gc).mean()))),
        'faster': [k for k in range(1, 21) if out['rows'][f'{k}|{n}']['holm40'] < .05 and out['rows'][f'{k}|{n}']['t_log_all'] < 0],
        'slower': [k for k in range(1, 21) if out['rows'][f'{k}|{n}']['holm40'] < .05 and out['rows'][f'{k}|{n}']['t_log_all'] > 0], 'worst_pct': float(100 * (g.max() - 1))}
    print(n, out[f'summary|{n}'], flush=True)
print('ARL0', out['arl0_test'], flush=True); json.dump(out, open(OUTF, 'w'), indent=1); print('DONE')
