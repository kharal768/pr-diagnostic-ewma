"""Random partitions of the variables under the matched protocol, evaluated in batches of seeds
(raw and standardised maxima).
Usage: python te_random_groups_batch.py first_seed last_seed <output.json>
Inputs: te_groups.json, data/te_cache/
Outputs: <output.json> (command-line argument)
Paper: Supplement S5.
"""
import json, sys, numpy as np, importlib.util
from scipy.stats import chi2
from te_lib import *
spec = importlib.util.spec_from_file_location('rieth', 'rieth_res.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
BASE = '../data/te_cache'; Rcor = np.load(f'{BASE}/Rcor.npy'); sizes = [len(g) for g in R.GROUPS.values()]
tot = {s: np.asarray(load(BASE, s, 'tot'), np.float64) for s in SETS}
limM = calibrate(None, tot['fftest'], None); baseM = first_alarm(alarm(None, tot['fftrain'], limM)).astype(float).mean()
out = json.load(open('te_groups.json')); done = {r['seed'] for r in out['random']}
def both(groups):
    Gi = [np.linalg.inv(Rcor[np.ix_(g, g)]).astype(np.float32) for g in groups]; raw = {}; std = {}
    for s in SETS:
        Z = np.load(f'{BASE}/{s}_Z.npy'); r_ = None; s_ = None
        for g, G in zip(groups, Gi):
            Zg = Z[..., g]; q = (C * np.einsum('rti,ij,rtj->rt', Zg, G, Zg)).astype(np.float64); qs = -chi2.logsf(q, len(g))
            r_ = q if r_ is None else np.maximum(r_, q); s_ = qs if s_ is None else np.maximum(s_, qs)
        raw[s], std[s] = r_, s_
    return raw, std
def evaluate(g):
    lim = match(g['fftrain'], tot['fftrain'], calibrate(g['fftest'], tot['fftest'], 0.3), baseM)
    rows = {k: cond_test(None, tot[f'f{k}'], limM, g[f'f{k}'], tot[f'f{k}'], lim) for k in range(1, 21)}
    r = np.array([rows[k]['ratio'] for k in rows]); t = np.array([rows[k]['t'] for k in rows])
    return {'mean_ratio': float(np.exp(np.log(r).mean())), 'n_faster': int((t < -3).sum()), 'n_slower': int((t > 3).sum()),
            'faster': [k for k in rows if rows[k]['t'] < -3], 'slower': [k for k in rows if rows[k]['t'] > 3], 'worst': float(r.max())}
A, B, OUTF = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
out = {'random': [r for r in out['random'] if A <= r['seed'] - 1000 < B]}; done = {r['seed'] for r in out['random']}
for sd in range(A, B):
    if 1000 + sd in done: continue
    rng = np.random.default_rng(1000 + sd); perm = rng.permutation(52); gr = []; i = 0
    for s in sizes: gr.append(sorted(perm[i:i + s].tolist())); i += s
    raw, std = both(gr); rec = {'seed': 1000 + sd, 'raw': evaluate(raw), 'std': evaluate(std)}
    out['random'].append(rec); print(sd, rec['raw']['mean_ratio'], rec['std']['mean_ratio'], flush=True)
    json.dump(out, open(OUTF, 'w'), indent=1)
print('DONE', flush=True)
