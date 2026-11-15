"""Random partitions of the variables under the primary protocol of te_equal.py, in batches.
Usage: python te_random_groups_equal.py first_seed last_seed <output.json>
Inputs: data/te_cache/
Outputs: <output.json> (command-line argument)
Paper: Section 5.2 (random partitions); Supplement S11.
"""
import json, sys, numpy as np, importlib.util
from scipy.stats import chi2
from te_lib import *
spec = importlib.util.spec_from_file_location('rieth', 'rieth_res.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
BASE = '../data/te_cache'; Rcor = np.load(f'{BASE}/Rcor.npy'); sizes = [len(g) for g in R.GROUPS.values()]
tot = {s: np.asarray(load(BASE, s, 'tot'), np.float64) for s in SETS}
limM = calibrate(None, tot['fftest'], None); baseM = first_alarm(alarm(None, tot['fftrain'], limM)).astype(float).mean()
out = {'random': []}; done = set()
def both(groups):
    Gi = [np.linalg.inv(Rcor[np.ix_(g, g)]).astype(np.float32) for g in groups]; raw = {}; std = {}
    for s in SETS:
        Z = np.load(f'{BASE}/{s}_Z.npy'); r_ = None; s_ = None
        for g, G in zip(groups, Gi):
            Zg = Z[..., g]; q = (C * np.einsum('rti,ij,rtj->rt', Zg, G, Zg)).astype(np.float64); qs = -chi2.logsf(q, len(g))
            r_ = q if r_ is None else np.maximum(r_, q); s_ = qs if s_ is None else np.maximum(s_, qs)
        raw[s], std[s] = r_, s_
    return raw, std
def d_all(f, t, l, on=160):
    post = alarm(f, t, l)[:, on:]; return np.where(post.any(1), np.argmax(post, 1) + 1, post.shape[1] + 1).astype(float)
DM = None
def evaluate(g):
    global DM
    if DM is None: DM = {k: d_all(None, tot[f'f{k}'], limM) for k in range(1, 21)}
    lim = calibrate(g['fftest'], tot['fftest'], 0.3); r = []; t = []
    for k in range(1, 21):
        dx = d_all(g[f'f{k}'], tot[f'f{k}'], lim); ld = np.log(dx) - np.log(DM[k]); r.append(dx.mean() / DM[k].mean()); t.append(ld.mean() / (ld.std(ddof=1) / np.sqrt(len(ld))) if ld.std() > 0 else 0.0)
    r = np.array(r); t = np.array(t)
    return {'t': t.tolist(), 'ratios': r.tolist(), 'mean_ratio': float(np.exp(np.log(r).mean())), 'n_faster': int((t < -3).sum()), 'n_slower': int((t > 3).sum()),
            'faster': [k + 1 for k in range(20) if t[k] < -3], 'slower': [k + 1 for k in range(20) if t[k] > 3], 'worst': float(r.max())}
A, B, OUTF = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

for sd in range(A, B):
    if 1000 + sd in done: continue
    rng = np.random.default_rng(1000 + sd); perm = rng.permutation(52); gr = []; i = 0
    for s in sizes: gr.append(sorted(perm[i:i + s].tolist())); i += s
    raw, std = both(gr); rec = {'seed': 1000 + sd, 'raw': evaluate(raw), 'std': evaluate(std)}
    out['random'].append(rec); print(sd, rec['raw']['mean_ratio'], rec['std']['mean_ratio'], flush=True)
    json.dump(out, open(OUTF, 'w'), indent=1)
print('DONE', flush=True)
