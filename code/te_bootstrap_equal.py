"""Bootstrap confidence intervals for the residual gains under the primary protocol (equal
in-control ARL on the testing runs, all runs). Each replicate resamples the fault-free testing
runs used for the limits and, per fault, the 500 faulty runs.
Usage: python te_bootstrap_equal.py seed replicates <output.json>
Inputs: data/te_cache/
Outputs: <output.json> (command-line argument)
Paper: Section 5.2 (bootstrap intervals); Supplement S15.
"""
import json, numpy as np, importlib.util
from te_lib import *
spec = importlib.util.spec_from_file_location('rieth', 'rieth_res.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
BASE = '../data/te_cache'; Rcor = np.load(f'{BASE}/Rcor.npy'); ENG = [list(g) for g in R.GROUPS.values()]
tot = {s: np.asarray(load(BASE, s, 'tot'), np.float32) for s in SETS}; um = {s: np.asarray(load(BASE, s, 'umax'), np.float32) for s in SETS}
gm = {s: gstat(BASE, s, ENG, Rcor, True).astype(np.float32) for s in SETS}
def d_all(f, t, l, on=160):
    post = alarm(f, t, l)[:, on:]; return np.where(post.any(1), np.argmax(post, 1) + 1, post.shape[1] + 1).astype(float)
def one(rng):
    it = rng.integers(0, 500, 500); T = tot['fftest'][it]; limM = calibrate(None, T, None); res = {}
    lims = {n: calibrate(f['fftest'][it], T, w1) for n, f, w1 in (('G-BAS', gm, 0.3), ('BAS-EWMA', um, 0.4))}
    for k in range(1, 21):
        jf = rng.integers(0, 500, 500); s = f'f{k}'; dm = d_all(None, tot[s][jf], limM)
        for n, f in (('G-BAS', gm), ('BAS-EWMA', um)): res.setdefault(n, []).append(d_all(f[s][jf], tot[s][jf], lims[n]).mean() / dm.mean())
    return {n: np.array(v) for n, v in res.items()}
import sys; rng = np.random.default_rng(int(sys.argv[1])); B = int(sys.argv[2]); acc = {'G-BAS': [], 'BAS-EWMA': []}; out = {}
for b in range(B):
    r = one(rng)
    for n in acc: acc[n].append(r[n])
    if b % 10 == 0: print(b, flush=True)
for n, v in acc.items():
    v = np.array(v); g = 100 * (1 - np.exp(np.log(v).mean(1))); lo, hi = np.quantile(v, [.025, .975], 0)
    out[n] = {'raw': np.array(v).tolist(), 'gain_ci': [float(np.quantile(g, .025)), float(np.quantile(g, .975))], 'gain_mean': float(g.mean()),
              'faster_ci': [k + 1 for k in range(20) if hi[k] < 1], 'slower_ci': [k + 1 for k in range(20) if lo[k] > 1]}
print(out); json.dump(out, open(sys.argv[3], 'w'), indent=1)
