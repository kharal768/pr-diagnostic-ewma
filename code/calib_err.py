"""Confidence intervals for paired ARL differences that include the calibration error of each
design's control limit (elasticity of the out-of-control ARL with respect to the in-control ARL).
Inputs: front.json, group_bench.json
Outputs: calib_err.json
Paper: Sections 3 and 4.3; Supplement S3.
"""
import json, numpy as np
from engine import run_lengths, arl
P, TAU, MAXT, R = 30, 50, 3000, 20000
F = json.load(open('front.json')); G = json.load(open('group_bench.json'))
def dv(idx): v = np.zeros(P); v[list(idx)] = 1.0; return v / np.linalg.norm(v)
D = {'MEWMA': (0, [0.05], F['MEWMA lam=0.05']['lim'], 12000), 'BAS w1=0.40': (6, [0.05, 1, P], F['BAS-EWMA lam=0.05 w1=0.4']['lim'], 12000),
     'G-BAS w1=0.1': (12, [0.05, 5], G['G-BAS w1=0.1']['lim'], 12000)}
def rls(name, s, delta, d, seed):
    k, prm, lim, _ = D[name]; return run_lengths(k, np.asarray(prm, float), np.asarray(lim, float) * s, P, delta, TAU if delta > 0 else 0, MAXT, R, seed, np.asarray(d, float))
def elasticity(name, cells):
    a0 = {s: rls(name, s, 0.0, np.zeros(P), 991) for s in (0.99, 1.01)}
    l0 = {s: np.log(a0[s][a0[s] > 0].mean()) for s in a0}
    e = {}
    for c, (delta, d) in cells.items():
        l1 = {s: np.log(rls(name, s, delta, d, 4321)[lambda x: x > 0].mean()) if False else None for s in ()}
        r = {s: rls(name, s, delta, d, 4321) for s in (0.99, 1.01)}; l1 = {s: np.log(r[s][r[s] > 0].mean()) for s in r}
        e[c] = float((l1[1.01] - l1[0.99]) / (l0[1.01] - l0[0.99]))
    base = rls(name, 1.0, 0.0, np.zeros(P), 991); base = base[base > 0]; rel = float(base.std() / base.mean() / np.sqrt(D[name][3]))
    return e, rel
def compare(A, B, cells):
    eA, rA = elasticity(A, cells); eB, rB = elasticity(B, cells); out = {}
    for c, (delta, d) in cells.items():
        a = rls(A, 1.0, delta, d, 4321); b = rls(B, 1.0, delta, d, 4321); m = (a > 0) & (b > 0)
        la, lb = np.log(a[m].mean()), np.log(b[m].mean())
        # delta-method MC SE of log ratio of paired means
        n = m.sum(); va = a[m].var() / n / a[m].mean() ** 2; vb = b[m].var() / n / b[m].mean() ** 2; cab = np.cov(a[m], b[m])[0, 1] / n / a[m].mean() / b[m].mean()
        se_mc = np.sqrt(va + vb - 2 * cab); se_cal = np.sqrt((eA[c] * rA) ** 2 + (eB[c] * rB) ** 2)
        out[c] = {'gain_pct': float(100 * (1 - np.exp(lb - la))), 'se_mc_pct': float(100 * se_mc), 'se_cal_pct': float(100 * se_cal),
                  'se_total_pct': float(100 * np.hypot(se_mc, se_cal)), 'elast': [eA[c], eB[c]]}
        print(A, B, c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[c].items()}, flush=True)
    return out
res = {}
res['BAS vs MEWMA, delta=0.75'] = compare('MEWMA', 'BAS w1=0.40', {f'nz={n}': (0.75, dv(range(n))) for n in (1, 2, 3, 5, 10, 30)})
res['G-BAS w1=0.1 vs MEWMA'] = compare('MEWMA', 'G-BAS w1=0.1', {'all 30, d=0.75': (0.75, dv(range(30))), '1 coord, d=0.75': (0.75, dv([0])), 'whole group, d=0.75': (0.75, dv(range(5)))})
json.dump(res, open('calib_err.json', 'w'), indent=1); print('DONE')
