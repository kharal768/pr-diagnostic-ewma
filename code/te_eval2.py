"""Tennessee Eastman comparison with limits matched on the training runs or on the pooled
fault-free runs (conditional delays).
Usage: python te_eval2.py <cache folder> <output.json>
Inputs: cache folder (data/te_cache or data/te_cache_var10) given on the command line.
Outputs: <output.json> (command-line argument)
Paper: Supplement S5 and S8.
"""
import json, sys, numpy as np, importlib.util
from te_lib import *
spec = importlib.util.spec_from_file_location('rieth', 'rieth_res.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
BASE, OUTF = sys.argv[1], sys.argv[2]; Rcor = np.load(f'{BASE}/Rcor.npy'); ENG = [list(g) for g in R.GROUPS.values()]
tot = {s: np.asarray(load(BASE, s, 'tot'), np.float64) for s in SETS}
first = {'BAS-EWMA': {s: np.asarray(load(BASE, s, 'umax'), np.float64) for s in SETS},
         'G-BAS raw': {s: gstat(BASE, s, ENG, Rcor, False) for s in SETS}, 'G-BAS std': {s: gstat(BASE, s, ENG, Rcor, True) for s in SETS}}
W1 = {'BAS-EWMA': 0.4, 'G-BAS raw': 0.3, 'G-BAS std': 0.3}
limM = calibrate(None, tot['fftest'], None); fa = lambda f, t, l: first_alarm(alarm(f, t, l)).astype(float)
baseM = fa(None, tot['fftrain'], limM).mean()
pool = lambda d: (np.concatenate([d['fftest'][:, :500], d['fftrain']]))
baseMp = fa(None, pool(tot), limM).mean()
rng = np.random.default_rng(3)
def boot(x, B=2000): idx = rng.integers(0, len(x), (B, len(x))); m = x[idx].mean(1); return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]
out = {'MEWMA': {'lim': limM, 'arl0_test': float(fa(None, tot['fftest'], limM).mean()), 'arl0_test_ci': boot(fa(None, tot['fftest'], limM)),
                 'fa_train': float(baseM), 'fa_train_ci': boot(fa(None, tot['fftrain'], limM))}}
for n, f in first.items():
    l0 = calibrate(f['fftest'], tot['fftest'], W1[n])
    for prot, lim in (('paper', match(f['fftrain'], tot['fftrain'], l0, baseM)), ('pooled', match(pool(f), pool(tot), l0, baseMp))):
        rows = {k: cond_test(None, tot[f'f{k}'], limM, f[f'f{k}'], tot[f'f{k}'], lim) for k in range(1, 21)}
        neg = cond_test(None, tot['fftest'], limM, f['fftest'], tot['fftest'], lim)
        r = np.array([rows[k]['ratio'] for k in rows])
        out[f'{n}|{prot}'] = {'lim': lim, 'mean_ratio': float(np.exp(np.log(r).mean())), 't': [rows[k]['t'] for k in rows], 'ratios': r.tolist(), 'worst': float(r.max()),
            'arl0_test': float(fa(f['fftest'], tot['fftest'], lim).mean()), 'arl0_test_ci': boot(fa(f['fftest'], tot['fftest'], lim)),
            'fa_train': float(fa(f['fftrain'], tot['fftrain'], lim).mean()), 'fa_train_ci': boot(fa(f['fftrain'], tot['fftrain'], lim)), 'negctl': neg}
        print(n, prot, round(out[f'{n}|{prot}']['mean_ratio'], 4), 'neg', round(neg['ratio'], 3), 'arl0', round(out[f'{n}|{prot}']['arl0_test'], 1), 'train', round(out[f'{n}|{prot}']['fa_train'], 1), flush=True)
json.dump(out, open(OUTF, 'w'), indent=1); print('DONE')
