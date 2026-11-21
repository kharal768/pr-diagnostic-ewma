"""Per-fault 95% intervals for the delay ratios on raw observations (paired delta method).
Inputs: te_eval2_raw.json, data/te_cache_raw/
Outputs: fig5_ci.json
Paper: Fig. 5.
"""
import json, numpy as np, importlib.util
from te_lib import *
spec = importlib.util.spec_from_file_location('rieth', 'rieth_res.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
BASE = '../data/te_cache_raw'; Rcor = np.load(f'{BASE}/Rcor.npy'); ENG = [list(g) for g in R.GROUPS.values()]
E = json.load(open('te_eval2_raw.json')); limM = E['MEWMA']['lim']
def delays(first, t, lim, on=160):
    al = alarm(first, t, lim); post = al[:, on:]
    return np.where(post.any(1), np.argmax(post, 1) + 1, al.shape[1] - on + 1).astype(float), ~al[:, :on].any(1)
out = {}
for k in range(1, 21):
    s = f'f{k}'; tot = np.asarray(load(BASE, s, 'tot'), np.float64); dm, cm = delays(None, tot, limM)
    for n, first, lim in (('G-BAS std', gstat(BASE, s, ENG, Rcor, True), E['G-BAS std|paper']['lim']), ('BAS-EWMA', np.asarray(load(BASE, s, 'umax'), np.float64), E['BAS-EWMA|paper']['lim'])):
        dx, cx = delays(first, tot, lim); m = cm & cx; a, b = dm[m], dx[m]; N = m.sum(); r = b.mean() / a.mean()
        v = r ** 2 * (b.var() / N / b.mean() ** 2 + a.var() / N / a.mean() ** 2 - 2 * np.cov(a, b)[0, 1] / N / a.mean() / b.mean())
        out[f'{k}|{n}'] = [float(r), float(r - 1.96 * np.sqrt(v)), float(r + 1.96 * np.sqrt(v))]
json.dump(out, open('fig5_ci.json', 'w'), indent=1); print('DONE')
