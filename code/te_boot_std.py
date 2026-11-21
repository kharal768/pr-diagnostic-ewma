"""As te_boot.py with the chi-square-standardised group maximum.
Inputs: data/te_cache/
Outputs: te_boot_std.json
Paper: Supplement S5.
"""
import json, numpy as np, importlib.util
from te_lib import *
spec = importlib.util.spec_from_file_location('rieth', 'rieth_res.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
BASE = '../data/te_cache'; Rcor = np.load(f'{BASE}/Rcor.npy'); ENG = [list(g) for g in R.GROUPS.values()]
tot = {s: np.asarray(load(BASE, s, 'tot'), np.float32) for s in SETS}; um = {s: np.asarray(load(BASE, s, 'umax'), np.float32) for s in SETS}
gm = {s: gstat(BASE, s, ENG, Rcor, True).astype(np.float32) for s in SETS}
def delays(first, t, lim, on=160):
    al = alarm(first, t, lim); post = al[:, on:]
    return np.where(post.any(1), np.argmax(post, 1) + 1, al.shape[1] - on + 1).astype(float), ~al[:, :on].any(1)
def one(rng):
    it = rng.integers(0, 500, 500); ir = rng.integers(0, 500, 500)
    T = tot['fftest'][it]; Tr = tot['fftrain'][ir]
    limM = calibrate(None, T, None); baseM = first_alarm(alarm(None, Tr, limM)).astype(float).mean(); res = {}
    for name, first, w1 in (('G-BAS', gm, 0.3), ('BAS-EWMA', um, 0.4)):
        lim = match(first['fftrain'][ir], Tr, calibrate(first['fftest'][it], T, w1), baseM); r = []
        for k in range(1, 21):
            jf = rng.integers(0, 500, 500); s = f'f{k}'; dm, cm = delays(None, tot[s][jf], limM); dx, cx = delays(first[s][jf], tot[s][jf], lim); m = cm & cx
            r.append(dx[m].mean() / dm[m].mean())
        res[name] = np.array(r)
    return res
rng = np.random.default_rng(5); B = 200; acc = {'G-BAS': [], 'BAS-EWMA': []}
for b in range(B):
    r = one(rng)
    for n in acc: acc[n].append(r[n])
    if b % 20 == 0: print(b, flush=True)
out = {}
for n, v in acc.items():
    v = np.array(v); gmr = np.exp(np.log(v).mean(1)); lo, hi = np.quantile(v, [.025, .975], 0)
    out[n] = {'gain_pct_ci95': [float(100 * (1 - np.quantile(gmr, .975))), float(100 * (1 - np.quantile(gmr, .025)))],
              'faults_ci_below1': [k + 1 for k in range(20) if hi[k] < 1], 'faults_ci_above1': [k + 1 for k in range(20) if lo[k] > 1]}
    print(n, out[n], flush=True)
json.dump(out, open('te_boot_std.json', 'w'), indent=1); print('DONE')
