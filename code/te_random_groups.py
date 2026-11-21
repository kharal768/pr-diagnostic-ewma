"""G-BAS with the chi-square-standardised group maximum for the engineering groups and for 100
random partitions with the same group sizes (raw and standardised maxima), matched protocol.
Usage: python te_random_groups.py [number of partitions]
Inputs: data/te_cache/
Outputs: te_groups.json
Paper: Supplement S5.
"""
import json, sys, numpy as np, importlib.util
from te_lib import *
spec = importlib.util.spec_from_file_location('rieth', 'rieth_res.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
BASE = '../data/te_cache'; Rcor = np.load(f'{BASE}/Rcor.npy'); ENG = [list(g) for g in R.GROUPS.values()]; sizes = [len(g) for g in ENG]
tot = {s: np.asarray(load(BASE, s, 'tot'), np.float64) for s in SETS}
limM = calibrate(None, tot['fftest'], None); baseM = first_alarm(alarm(None, tot['fftrain'], limM)).astype(float).mean()
def evaluate(groups, std):
    g = {s: gstat(BASE, s, groups, Rcor, std) for s in SETS}
    lim = match(g['fftrain'], tot['fftrain'], calibrate(g['fftest'], tot['fftest'], 0.3), baseM)
    rows = {k: cond_test(None, tot[f'f{k}'], limM, g[f'f{k}'], tot[f'f{k}'], lim) for k in range(1, 21)}
    r = np.array([rows[k]['ratio'] for k in rows]); t = np.array([rows[k]['t'] for k in rows])
    return {'mean_ratio': float(np.exp(np.log(r).mean())), 'n_faster': int((t < -3).sum()), 'n_slower': int((t > 3).sum()),
            'faster': [k for k in rows if rows[k]['t'] < -3], 'slower': [k for k in rows if rows[k]['t'] > 3], 'worst': float(r.max()), 'rows': rows}
out = {'eng_raw': evaluate(ENG, False), 'eng_std': evaluate(ENG, True), 'random': []}
for k in ('eng_raw', 'eng_std'): print(k, {a: b for a, b in out[k].items() if a != 'rows'}, flush=True)
json.dump(out, open('te_groups.json', 'w'), indent=1)
NR = int(sys.argv[1]) if len(sys.argv) > 1 else 100
for sd in range(NR):
    rng = np.random.default_rng(1000 + sd); perm = rng.permutation(52); gr = []; i = 0
    for s in sizes: gr.append(sorted(perm[i:i + s].tolist())); i += s
    rec = {'seed': 1000 + sd}
    for std in (False, True):
        e = evaluate(gr, std); rec['std' if std else 'raw'] = {a: b for a, b in e.items() if a != 'rows'}
    out['random'].append(rec); print(sd, rec['raw']['mean_ratio'], rec['std']['mean_ratio'], flush=True)
    json.dump(out, open('te_groups.json', 'w'), indent=1)
print('DONE', flush=True)
