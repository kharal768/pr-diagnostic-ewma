"""Tennessee Eastman, Rieth et al. (2017) data on raw observations: Phase I from the 500 fault-free
training runs, standardisation, Ledoit-Wolf covariance, calibration of MEWMA, G-BAS and
BAS-EWMA (lambda = 0.1).
Inputs: data/rieth2017/, data/tep_braatz/
Outputs: rieth_cal.json
Paper: Section 5.1.
"""
import numpy as np, json, sys, os, importlib.util
from sklearn.covariance import LedoitWolf
DR = '../data/rieth2017'
spec = importlib.util.spec_from_file_location('tep', 'tep.py')
sys_argv = sys.argv; sys.argv = ['x', '../data/tep_braatz']
tep = importlib.util.module_from_spec(spec); spec.loader.exec_module(tep); sys.argv = sys_argv
GROUPS = tep.GROUPS

def runs(name, n_per):
    X = np.load(f'{DR}/{name}.npy').astype(np.float64); r = np.load(f'{DR}/{name}_run.npy')
    R = len(np.unique(r)); return X.reshape(R, n_per, 52)

TR = runs('ff_train', 500).reshape(-1, 52)
mu, sd = TR.mean(0), TR.std(0, ddof=1)
Zs = lambda X: (X - mu) / sd
lw = LedoitWolf().fit(Zs(TR)); Rcor = lw.covariance_
Rinv = np.linalg.inv(Rcor); Linv = np.linalg.inv(np.linalg.cholesky(Rcor))
Ginv = {k: np.linalg.inv(Rcor[np.ix_(g, g)]) for k, g in GROUPS.items()}

def terms(X, lam):
    """X: (runs, T, 52). Returns tot, gmax, umax each (runs, T)."""
    Z = np.zeros_like(X); z = np.zeros((X.shape[0], 52)); Xs = Zs(X)
    for t in range(X.shape[1]):
        z = lam * Xs[:, t] + (1 - lam) * z; Z[:, t] = z
    c = (2 - lam) / lam
    tot = c * np.einsum('rti,ij,rtj->rt', Z, Rinv, Z)
    gmax = np.max(np.stack([c * np.einsum('rti,ij,rtj->rt', Z[..., g], Ginv[k], Z[..., g]) for k, g in GROUPS.items()]), 0)
    umax = (c * (Z @ Linv.T) ** 2).max(-1)
    return tot, gmax, umax

def first_alarm(al):                       # al: (runs, T) bool -> 1-based first alarm, T+1 if none
    has = al.any(1); fa = np.argmax(al, 1) + 1
    return np.where(has, fa, al.shape[1] + 1)

def alarm(kind, tt, lim, w1):
    tot, gmax, umax = tt
    if kind == 'MEWMA':
        return tot > lim[0]
    first = gmax if kind == 'G-BAS' else umax
    return (first > lim[0]) | (tot > lim[1])

def calibrate(kind, lam, w1, FT, target=200.0):
    tt = terms(FT, lam); tot, gmax, umax = tt
    first = gmax if kind == 'G-BAS' else umax
    def lims(a):
        if kind == 'MEWMA':
            return [np.quantile(tot, 1 - a)]
        return [np.quantile(first, 1 - a * w1), np.quantile(tot, 1 - a * (1 - w1))]
    lo, hi = 1e-6, 0.6
    for _ in range(50):
        a = np.sqrt(lo * hi)
        arl0 = first_alarm(alarm(kind, tt, lims(a), w1)).mean()
        lo, hi = (lo, a) if arl0 < target else (a, hi)
    a = np.sqrt(lo * hi); L = lims(a)
    return [float(x) for x in L], float(first_alarm(alarm(kind, tt, L, w1)).mean())

if __name__ == '__main__':
    FT = runs('ff_test', 960)
    out = {'lw_shrinkage': float(lw.shrinkage_), 'designs': {}}
    grid = [('MEWMA', l, None) for l in (0.05, 0.1, 0.2)] + \
           [('G-BAS', l, w) for l in (0.05, 0.1, 0.2) for w in (0.1, 0.3, 0.5)] + \
           [('BAS-EWMA', l, 0.4) for l in (0.05, 0.1, 0.2)]
    for kind, lam, w1 in grid:
        name = f'{kind} lam={lam}' + ('' if w1 is None else f' w1={w1}')
        lim, a0 = calibrate(kind, lam, w1, FT)
        out['designs'][name] = {'kind': kind, 'lam': lam, 'w1': w1, 'lim': lim, 'arl0': a0}
        print(f'{name:24s} in-control ARL {a0:6.1f}  limits {np.round(lim, 2)}', flush=True)
    json.dump(out, open('rieth_cal.json', 'w'), indent=1)
