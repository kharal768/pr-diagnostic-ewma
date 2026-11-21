"""Shared helpers for the Tennessee Eastman analyses on the cached EWMA arrays.

Loading of cached statistics, limit setting to a target in-control ARL on the fault-free
testing runs, first-alarm delays, paired t-tests on log delays and Holm adjustment.
Library module imported by the analysis scripts; writes no files.
Paper: Sections 5.1-5.2; Supplement S8, S11, S15.
"""
import numpy as np
from scipy.stats import chi2
SETS = ['fftest', 'fftrain'] + [f'f{k}' for k in range(1, 21)]
lam = 0.1; C = (2 - lam) / lam
def load(base, name, what): return np.load(f'{base}/{name}_{what}.npy', mmap_mode='r')
def gstat(base, name, groups, Rcor, std):
    Z = load(base, name, 'Z'); out = None
    for g in groups:
        Gi = np.linalg.inv(Rcor[np.ix_(g, g)]); Zg = np.asarray(Z[..., g], np.float64)
        q = C * np.einsum('rti,ij,rtj->rt', Zg, Gi, Zg)
        if std: q = -chi2.logsf(q, len(g))
        out = q if out is None else np.maximum(out, q)
    return out
def first_alarm(al):
    return np.where(al.any(1), np.argmax(al, 1) + 1, al.shape[1] + 1)
def alarm(first, tot, lim): return tot > lim[0] if first is None else (first > lim[0]) | (tot > lim[1])
def calibrate(first, tot, w1, target=200.0):
    def lims(a): return [np.quantile(tot, 1 - a)] if first is None else [np.quantile(first, 1 - a * w1), np.quantile(tot, 1 - a * (1 - w1))]
    lo, hi = 1e-6, 0.6
    for _ in range(40):
        a = np.sqrt(lo * hi); lo, hi = (lo, a) if first_alarm(alarm(first, tot, lims(a))).mean() < target else (a, hi)
    return lims(np.sqrt(lo * hi))
def match(first, tot, lim, base):
    lo, hi = 0.85, 1.2
    for _ in range(30):
        s = 0.5 * (lo + hi); fa = first_alarm(alarm(first, tot, np.array(lim) * s)).astype(float).mean()
        lo, hi = (s, hi) if fa < base else (lo, s)
    return (np.array(lim) * 0.5 * (lo + hi)).tolist()
def cond_test(fm, tm, limm, fx, tx, limx, on=160):
    def fa(first, tot, lim):
        al = alarm(first, tot, lim); post = al[:, on:]
        return np.where(post.any(1), np.argmax(post, 1) + 1, al.shape[1] - on + 1).astype(float), ~al[:, :on].any(1)
    dm, cm = fa(fm, tm, limm); dx, cx = fa(fx, tx, limx); m = cm & cx; d = dx[m] - dm[m]
    se = d.std(ddof=1) / np.sqrt(m.sum()); dany = dx - dm
    return {'ratio': float(dx[m].mean() / dm[m].mean()), 't': float(d.mean() / se) if se > 0 else 0.0, 'n': int(m.sum()),
            'ratio_any': float(dx.mean() / dm.mean()), 't_any': float(dany.mean() / (dany.std(ddof=1) / np.sqrt(len(dany))))}
