"""Sensitivity of G-BAS on VAR(1) residuals to the grouping: engineering groups against random
partitions with the same group sizes, matched protocol.
Usage: python rieth_randgroups.py [seed ...]
Inputs: rieth_res_cal.json, data/rieth2017/
Outputs: rieth_randgroups.json
Paper: Supplement S5.
"""
import numpy as np, json, importlib.util, sys
spec = importlib.util.spec_from_file_location('rieth', 'rieth_res.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
D = '../data/rieth2017/ftest'; X = np.load(f'{D}/X.npy', mmap_mode='r'); F = np.load(f'{D}/fault.npy')
cal = json.load(open('rieth_res_cal.json'))['designs']; mew = cal['MEWMA lam=0.1']['lim']
sizes = [len(g) for g in R.GROUPS.values()]
FT = R.runs('ff_test', 960); TR = R.runs('ff_train', 500)
def set_groups(groups):
    R.GROUPS = groups; R.Ginv = {k: np.linalg.inv(R.Rcor[np.ix_(g, g)]) for k, g in groups.items()}
out = {}
seeds = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
fault_tt = {}
for sd in seeds:
    rng = np.random.default_rng(sd); perm = rng.permutation(52); groups = {}; i = 0
    for k, s in enumerate(sizes): groups[k] = sorted(perm[i:i + s].tolist()); i += s
    set_groups(groups)
    lim, a0 = R.calibrate('G-BAS', 0.1, 0.3, FT)
    tt = R.terms(TR, 0.1)
    fa = lambda L: R.first_alarm(R.alarm('G-BAS', tt, L, 0.3)).astype(float).mean()
    base = R.first_alarm(R.alarm('MEWMA', tt, mew, None)).astype(float).mean()
    lo, hi = 0.9, 1.15
    for _ in range(25):
        s = 0.5 * (lo + hi); lo, hi = (s, hi) if fa(np.array(lim) * s) < base else (lo, s)
    L = (np.array(lim) * 0.5 * (lo + hi)).tolist()
    ratios, ts = [], []
    for k in range(1, 21):
        Xk = np.asarray(X[np.flatnonzero(F == k)], np.float64).reshape(500, 960, 52); t3 = R.terms(Xk, 0.1)
        def fa_after(al):
            post = al[:, 160:]; return np.where(post.any(1), np.argmax(post, 1) + 1, 801).astype(float), ~al[:, :160].any(1)
        dm, cm = fa_after(R.alarm('MEWMA', t3, mew, None)); dg, cg = fa_after(R.alarm('G-BAS', t3, L, 0.3))
        m = cm & cg; d = dg[m] - dm[m]; se = d.std(ddof=1) / np.sqrt(m.sum()) if d.std() > 0 else np.inf
        ratios.append(float(dg[m].mean() / dm[m].mean())); ts.append(float(d.mean() / se) if np.isfinite(se) else 0.0)
    r = np.array(ratios); t = np.array(ts)
    out[sd] = {'groups': groups, 'arl0': a0, 'mean_ratio': float(np.exp(np.log(r).mean())), 'sig_faster': [k + 1 for k in range(20) if t[k] < -3],
               'sig_slower': [k + 1 for k in range(20) if t[k] > 3], 'worst': float(r.max()), 'ratios': ratios}
    print(f"seed {sd}: ARL0 {a0:.1f}  mean ratio {out[sd]['mean_ratio']:.4f}  faster {out[sd]['sig_faster']}  slower {out[sd]['sig_slower']}  worst x{r.max():.3f}", flush=True)
    json.dump(out, open('rieth_randgroups.json', 'w'), indent=1)
print('DONE', flush=True)
