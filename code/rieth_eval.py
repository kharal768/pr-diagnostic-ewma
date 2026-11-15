"""Charts on the faulty training runs, where each fault starts at sample 21 (zero-state effect).
Inputs: rieth_cal.json, data/rieth2017/
Outputs: rieth_eval.json
Paper: Section 5.2 (faulty training runs).
"""
import numpy as np, json, importlib.util
spec = importlib.util.spec_from_file_location('rieth', 'rieth.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
D = '../data/rieth2017/ftrain'
X = np.load(f'{D}/X.npy', mmap_mode='r'); F = np.load(f'{D}/fault.npy')
cal = json.load(open('rieth_cal.json'))['designs']
ONSET = 20
out = {n: {} for n in cal}
conc = {}
for k in range(1, 21):
    idx = np.flatnonzero(F == k); Xk = np.asarray(X[idx], np.float64).reshape(500, 500, 52)
    for lam in (0.05, 0.1, 0.2):
        tt = R.terms(Xk, lam)
        for n, d in cal.items():
            if d['lam'] != lam:
                continue
            al = R.alarm(d['kind'], tt, d['lim'], d['w1'])
            clean = ~al[:, :ONSET].any(1)
            post = al[clean, ONSET:]
            has = post.any(1); delay = np.where(has, np.argmax(post, 1) + 1, 481)
            out[n][k] = {'cond_arl': float(delay.mean()), 'undetected': float(1 - has.mean()),
                         'early_false': float(1 - clean.mean()), 'n_clean': int(clean.sum())}
        if lam == 0.1:   # concentration diagnostic (group energy excess over samples 21-120)
            Z = np.zeros_like(Xk); z = np.zeros((500, 52)); Xs = R.Zs(Xk)
            for t in range(500):
                z = lam * Xs[:, t] + (1 - lam) * z; Z[:, t] = z
            c = (2 - lam) / lam
            E = np.stack([c * np.einsum('rti,ij,rtj->rt', Z[:, ONSET:ONSET + 100][..., g], R.Ginv[n_], Z[:, ONSET:ONSET + 100][..., g])
                          for n_, g in R.GROUPS.items()], -1).mean((0, 1))
            base = np.array([len(g) for g in R.GROUPS.values()]) * 2.8   # in-control level is ~2.8x nominal here
            w = np.clip(E - base, 0, None); conc[k] = float(w.max() / w.sum()) if w.sum() > 0 else None
    m, g = out['MEWMA lam=0.1'][k], out['G-BAS lam=0.1 w1=0.3'][k]
    print(f"fault {k:2d}: MEWMA {m['cond_arl']:7.2f}  G-BAS {g['cond_arl']:7.2f}  ratio {g['cond_arl']/m['cond_arl']:.3f}   undetected {m['undetected']:.2f}  conc {conc[k] if conc[k] is None else round(conc[k],2)}", flush=True)
json.dump({'designs': out, 'concentration': conc}, open('rieth_eval.json', 'w'), indent=1)
print('DONE', flush=True)
