"""Tennessee Eastman (Braatz data set): variable list, process-unit groups and loading.

Defines the 52 variables (XMEAS 1-41, XMV 1-11) and the eight engineering groups used by
G-BAS, and runs the charts on the 21 faults of the Braatz data.
Usage: python tep.py [data folder]  (default: ../data/tep_braatz)
Inputs: data/tep_braatz/
Outputs: tep.json
Paper: variable groups of Section 5.1 (assignment in Supplement S5).
"""
import numpy as np, json, sys
from sklearn.covariance import LedoitWolf

D = sys.argv[1] if len(sys.argv) > 1 else '../data/tep_braatz'
X0 = np.loadtxt(f'{D}/d00.dat').T            # (500, 52)
CAL = np.loadtxt(f'{D}/d00_te.dat')          # (960, 52)
FAULTS = {k: np.loadtxt(f'{D}/d{k:02d}_te.dat') for k in range(1, 22)}
assert X0.shape[1] == 52 and CAL.shape == (960, 52)

M = lambda i: i - 1                 # XMEAS(i) -> column
V = lambda i: 41 + i - 1            # XMV(i)  -> column
GROUPS = {
    'feeds':              [M(1), M(2), M(3), M(4), V(1), V(2), V(3), V(4)],
    'reactor':            [M(6), M(7), M(8), M(9), M(21), V(10)],
    'separator':          [M(11), M(12), M(13), M(14), M(22), V(7), V(11)],
    'stripper':           [M(15), M(16), M(17), M(18), M(19), V(8), V(9)],
    'compressor/purge':   [M(5), M(10), M(20), V(5), V(6)],
    'reactor feed anal.': [M(i) for i in range(23, 29)],
    'purge gas anal.':    [M(i) for i in range(29, 37)],
    'product anal.':      [M(i) for i in range(37, 42)],
}
allidx = sorted(i for g in GROUPS.values() for i in g)
assert allidx == list(range(52)), 'groups must partition the 52 variables'

mu, sd = X0.mean(0), X0.std(0, ddof=1)
Zs = lambda X: (X - mu) / sd
Rcor = LedoitWolf().fit(Zs(X0)).covariance_
shrink = LedoitWolf().fit(Zs(X0)).shrinkage_
Rinv = np.linalg.inv(Rcor)
L = np.linalg.cholesky(Rcor); Linv = np.linalg.inv(L)        # regression adjustment (original order)
Ginv = {k: np.linalg.inv(Rcor[np.ix_(g, g)]) for k, g in GROUPS.items()}

def ewma(X, lam):
    Z = np.zeros_like(X); z = np.zeros(X.shape[1])
    for t in range(X.shape[0]):
        z = lam * X[t] + (1 - lam) * z; Z[t] = z
    return Z

def stats(X, lam):
    """Return per-sample terms: total (MEWMA), max group energy, max regression-adjusted coordinate."""
    Z = ewma(Zs(X), lam); c = (2 - lam) / lam
    tot = c * np.einsum('ti,ij,tj->t', Z, Rinv, Z)
    grp = np.stack([c * np.einsum('ti,ij,tj->t', Z[:, g], Ginv[k], Z[:, g]) for k, g in GROUPS.items()], 1)
    U = c * (Z @ Linv.T) ** 2
    return tot, grp.max(1), U.max(1)

ALPHA = 0.01
def limits(kind, lam, w1):
    tot, gmax, umax = stats(CAL, lam)
    if kind == 'MEWMA':
        return {'tot': np.quantile(tot, 1 - ALPHA)}
    first = gmax if kind == 'G-BAS' else umax
    lo, hi = 0.0, 3.0                    # scale a on the per-term tail budgets so the joint rate = ALPHA
    for _ in range(60):
        a = 0.5 * (lo + hi)
        h1 = np.quantile(first, 1 - min(0.999, ALPHA * a * w1)); h2 = np.quantile(tot, 1 - min(0.999, ALPHA * a * (1 - w1)))
        rate = np.mean((first > h1) | (tot > h2))
        lo, hi = (a, hi) if rate < ALPHA else (lo, a)
    return {'first': h1, 'tot': h2}

def alarms(X, kind, lam, lim):
    tot, gmax, umax = stats(X, lam)
    if kind == 'MEWMA':
        return tot > lim['tot']
    first = gmax if kind == 'G-BAS' else umax
    return (first > lim['first']) | (tot > lim['tot'])

def evaluate(kind, lam, w1):
    lim = limits(kind, lam, w1)
    cal_rate = alarms(CAL, kind, lam, lim).mean()
    pre, out = [], {}
    for k, X in FAULTS.items():
        a = alarms(X, kind, lam, lim)
        pre.append(a[:160]); post = a[160:]
        first = np.argmax(post) + 1 if post.any() else None
        out[k] = {'delay': None if first is None else int(first), 'rate': float(post.mean())}
    return {'cal_rate': float(cal_rate), 'pre_rate': float(np.concatenate(pre).mean()), 'faults': out,
            'lim': {k: float(v) for k, v in lim.items()}}

if __name__ == '__main__':
    print('Ledoit-Wolf shrinkage intensity:', round(shrink, 3))
    res = {'shrinkage': float(shrink), 'groups': {k: [int(i) for i in g] for k, g in GROUPS.items()}, 'designs': {}}
    grid = [('MEWMA', l, None) for l in (0.05, 0.1, 0.2)] + \
           [('G-BAS', l, w) for l in (0.05, 0.1, 0.2) for w in (0.1, 0.3, 0.5)] + \
           [('BAS-EWMA', l, w) for l in (0.05, 0.1, 0.2) for w in (0.2, 0.4)]
    for kind, lam, w1 in grid:
        name = f'{kind} lam={lam}' + ('' if w1 is None else f' w1={w1}')
        r = evaluate(kind, lam, w1)
        res['designs'][name] = r
        print(f"{name:26s} cal {r['cal_rate']:.4f}  pre-fault {r['pre_rate']:.4f} | delays " +
              ' '.join('  -' if r['faults'][k]['delay'] is None else f"{r['faults'][k]['delay']:3d}" for k in range(1, 22)), flush=True)
    json.dump(res, open('tep.json', 'w'), indent=1)
