"""Robustness of the residual comparison under matched limits: all runs, log delays and the
Wilcoxon signed-rank test, Holm over faults (raw group maximum).
Inputs: rieth_res_cal.json, rieth_res_matched.json, data/te_cache/
Outputs: te_robust.json
Paper: Supplement S5.
"""
import json, numpy as np, importlib.util
from scipy.stats import wilcoxon, norm
from te_lib import *
spec = importlib.util.spec_from_file_location('rieth', 'rieth_res.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
BASE = '../data/te_cache'; Rcor = np.load(f'{BASE}/Rcor.npy'); ENG = [list(g) for g in R.GROUPS.values()]
cal = json.load(open('rieth_res_cal.json'))['designs']; M = json.load(open('rieth_res_matched.json'))
limM = cal['MEWMA lam=0.1']['lim']; LG = M['G-BAS lam=0.1 w1=0.3']['matched_lim']; LB = M['BAS-EWMA lam=0.1 w1=0.4']['matched_lim']
def delays(first, tot, lim, on=160):
    al = alarm(first, tot, lim); post = al[:, on:]
    return np.where(post.any(1), np.argmax(post, 1) + 1, al.shape[1] - on + 1).astype(float), ~al[:, :on].any(1)
out = {}
for k in range(1, 21):
    s = f'f{k}'; tot = np.asarray(load(BASE, s, 'tot'), np.float64); dm, cm = delays(None, tot, limM)
    for name, first, lim in (('G-BAS', gstat(BASE, s, ENG, Rcor, False), LG), ('BAS-EWMA', np.asarray(load(BASE, s, 'umax'), np.float64), LB)):
        dx, cx = delays(first, tot, lim); m = cm & cx; ld = np.log(dx[m]) - np.log(dm[m])
        w = wilcoxon(dx[m], dm[m]) if np.any(dx[m] != dm[m]) else None
        da = dx - dm
        out[f'{k}|{name}'] = {'ratio_cond': float(dx[m].mean() / dm[m].mean()), 'ratio_any': float(dx.mean() / dm.mean()),
            't_any': float(da.mean() / (da.std(ddof=1) / np.sqrt(500))) if da.std() > 0 else 0.0,
            'geo_ratio_cond': float(np.exp(ld.mean())), 't_log': float(ld.mean() / (ld.std(ddof=1) / np.sqrt(m.sum()))) if ld.std() > 0 else 0.0,
            'p_wilcoxon': float(w.pvalue) if w else 1.0, 'sign_faster': int(np.sign(ld.mean()) < 0)}
    print(k, {n: round(out[f'{k}|{n}']['ratio_any'], 3) for n in ('G-BAS', 'BAS-EWMA')}, flush=True)
for name in ('G-BAS', 'BAS-EWMA'):
    keys = [f'{k}|{name}' for k in range(1, 21)]
    ra = np.array([out[k]['ratio_any'] for k in keys]); ta = np.array([out[k]['t_any'] for k in keys]); tl = np.array([out[k]['t_log'] for k in keys])
    pw = np.array([out[k]['p_wilcoxon'] for k in keys]); o = np.argsort(pw); padj = np.empty(20); run = 0
    for r, i in enumerate(o): run = max(run, min(1, (20 - r) * pw[i])); padj[i] = run
    out[f'summary|{name}'] = {'geo_mean_ratio_any': float(np.exp(np.log(ra).mean())), 'sig_any_faster': [k + 1 for k in range(20) if ta[k] < -3], 'sig_any_slower': [k + 1 for k in range(20) if ta[k] > 3],
        'log_t_faster': [k + 1 for k in range(20) if tl[k] < -3], 'log_t_slower': [k + 1 for k in range(20) if tl[k] > 3],
        'wilcoxon_holm_sig': [k + 1 for k in range(20) if padj[k] < 0.05]}
    print(name, out[f'summary|{name}'], flush=True)
json.dump(out, open('te_robust.json', 'w'), indent=1)
