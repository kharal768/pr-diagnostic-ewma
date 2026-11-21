"""Residual comparison with the VAR(1) filter and Phase I fitted on half of the training runs.
Inputs: data/te_cache_split/
Outputs: te_split.json
Paper: Section 5.2 (half-sample filter); Supplement S15.
"""
import json, numpy as np, importlib.util
from te_lib import *
spec = importlib.util.spec_from_file_location('rieth', 'rieth_res.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
BASE = '../data/te_cache_split'; Rcor = np.load(f'{BASE}/Rcor.npy'); ENG = [list(g) for g in R.GROUPS.values()]
tot = {s: np.asarray(load(BASE, s, 'tot'), np.float64) for s in SETS}; um = {s: np.asarray(load(BASE, s, 'umax'), np.float64) for s in SETS}
gm = {s: gstat(BASE, s, ENG, Rcor, False) for s in SETS}
limM = calibrate(None, tot['fftest'], None); baseM = first_alarm(alarm(None, tot['fftrain'], limM)).astype(float).mean()
out = {'MEWMA': {'lim': limM, 'arl0_test': float(first_alarm(alarm(None, tot['fftest'], limM)).mean()), 'mean_fa_train': baseM}}
for name, first, w1 in (('G-BAS', gm, 0.3), ('BAS-EWMA', um, 0.4)):
    l0 = calibrate(first['fftest'], tot['fftest'], w1); lim = match(first['fftrain'], tot['fftrain'], l0, baseM)
    rows = {k: cond_test(None, tot[f'f{k}'], limM, first[f'f{k}'], tot[f'f{k}'], lim) for k in range(1, 21)}
    r = np.array([rows[k]['ratio'] for k in rows]); t = np.array([rows[k]['t'] for k in rows])
    out[name] = {'lim': lim, 'scale': float(lim[0] / l0[0]), 'mean_ratio': float(np.exp(np.log(r).mean())), 'faster': [k for k in rows if rows[k]['t'] < -3],
                 'slower': [k for k in rows if rows[k]['t'] > 3], 'worst': float(r.max()), 'rows': rows}
    print(name, {a: b for a, b in out[name].items() if a != 'rows'}, flush=True)
json.dump(out, open('te_split.json', 'w'), indent=1); print('DONE')
