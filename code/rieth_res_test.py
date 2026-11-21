"""As rieth_test.py on VAR(1) residuals.
Inputs: rieth_res_cal.json, data/rieth2017/
Outputs: rieth_res_test.json
Paper: Supplement S5.
"""
import numpy as np, json, importlib.util
spec = importlib.util.spec_from_file_location('rieth', 'rieth_res.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
D = '../data/rieth2017/ftest'
X = np.load(f'{D}/X.npy', mmap_mode='r'); F = np.load(f'{D}/fault.npy')
cal = json.load(open('rieth_res_cal.json'))['designs']; adj = {k: v['lim'] for k, v in cal.items()}
D_ = {'MEWMA': (cal['MEWMA lam=0.1'], cal['MEWMA lam=0.1']['lim']),
      'G-BAS': (cal['G-BAS lam=0.1 w1=0.3'], adj['G-BAS lam=0.1 w1=0.3']),
      'BAS-EWMA': (cal['BAS-EWMA lam=0.1 w1=0.4'], adj['BAS-EWMA lam=0.1 w1=0.4'])}
ON = 160
def first_after(al):
    post = al[:, ON:]; return np.where(post.any(1), np.argmax(post, 1) + 1, 801).astype(float), ~al[:, :ON].any(1)
out = {}
for k in range(1, 21):
    Xk = np.asarray(X[np.flatnonzero(F == k)], np.float64).reshape(500, 960, 52); tt = R.terms(Xk, 0.1)
    res = {n: first_after(R.alarm(d['kind'], tt, lim, d['w1'])) for n, (d, lim) in D_.items()}
    dm, cm = res['MEWMA']; row = {}
    for n in ('G-BAS', 'BAS-EWMA'):
        dx, cx = res[n]
        for meas, mask in (('cond', cm & cx), ('any', np.ones(500, bool))):
            d = dx[mask] - dm[mask]; se = d.std(ddof=1) / np.sqrt(mask.sum()) if d.std() > 0 else np.inf
            row[f'{n}|{meas}'] = {'mewma': float(dm[mask].mean()), 'chart': float(dx[mask].mean()),
                                  'ratio': float(dx[mask].mean() / dm[mask].mean()), 't': float(d.mean() / se) if np.isfinite(se) else 0.0,
                                  'n': int(mask.sum())}
    row['pre_alarm_share_mewma'] = float(1 - cm.mean())
    out[k] = row
    g, b = row['G-BAS|cond'], row['BAS-EWMA|cond']
    print(f"fault {k:2d} (cond n={g['n']:3d}): MEWMA {g['mewma']:7.2f} | G-BAS x{g['ratio']:.3f} t={g['t']:+6.1f} | BAS-EWMA x{b['ratio']:.3f} t={b['t']:+6.1f}"
          f"   [any: G x{row['G-BAS|any']['ratio']:.3f} t={row['G-BAS|any']['t']:+5.1f}, B x{row['BAS-EWMA|any']['ratio']:.3f} t={row['BAS-EWMA|any']['t']:+5.1f}]", flush=True)
json.dump(out, open('rieth_res_test.json', 'w'), indent=1)
for n in ('G-BAS', 'BAS-EWMA'):
    for meas in ('cond', 'any'):
        r = np.array([out[k][f'{n}|{meas}']['ratio'] for k in out]); t = np.array([out[k][f'{n}|{meas}']['t'] for k in out])
        print(f"{n:8s} {meas:4s}: mean ratio {np.exp(np.log(r).mean()):.4f}  sig faster {[k for k in out if out[k][f'{n}|{meas}']['t'] < -3]}  sig slower {[k for k in out if out[k][f'{n}|{meas}']['t'] > 3]}  worst x{r.max():.3f}")
print('DONE', flush=True)
