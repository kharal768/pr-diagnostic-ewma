"""Monte Carlo standard errors of the gains of equal-PR directions with different shapes
(paired, delta in {0.75, 1.0}).
Inputs: final.json
Outputs: shape_se.json
Paper: Section 4.2 (direct shape test); Supplement S1.
"""
import json, numpy as np
from scipy.optimize import brentq
from engine import run_lengths
p, R, TAU, MAXT = 30, 15000, 50, 3000; F = json.load(open('final.json'))['p30']
CH = {'MEWMA': (0, [0.05], [F['bench']['MEWMA l=0.05']['lim']]), 'BAS w1=0.5': (6, F['bas']['w1=0.5']['prm'], F['bas']['w1=0.5']['lim']), 'BAS w1=0.25': (6, F['bas']['w1=0.25']['prm'], F['bas']['w1=0.25']['lim'])}
def tail_dir(t):
    f = lambda a: 1 / (a ** 4 + (1 - a ** 2) ** 2 / 29) - t; a = brentq(f, 1 / np.sqrt(30) + 1e-9, 1 - 1e-9); d = np.full(p, np.sqrt((1 - a * a) / 29)); d[0] = a; return d
def eq(k): d = np.zeros(p); d[:k] = 1 / np.sqrt(k); return d
out = []
for i, (lab, d) in enumerate([(f'{k} equal', eq(k)) for k in (2, 3, 4, 5)] + [(f'dominant+tail PR={k}', tail_dir(k)) for k in (2, 3, 4, 5)]):
    rec = {'dir': lab, 'pr': float(1 / np.sum(d ** 4))}; seed = 70_000_000 + i * 100_000
    rl = {n: [run_lengths(k_, np.asarray(pr, float), np.asarray(l, float), p, dl, TAU, MAXT, R, seed + j * 50_000, d) for j, dl in enumerate((0.75, 1.0))] for n, (k_, pr, l) in CH.items()}
    for n in ('BAS w1=0.5', 'BAS w1=0.25'):
        g = []; v = []
        for j in range(2):
            a, b = rl['MEWMA'][j], rl[n][j]; m = (a > 0) & (b > 0); N = m.sum(); ma, mb = a[m].mean(), b[m].mean()
            g.append(1 - mb / ma); va = a[m].var() / N / ma ** 2; vb = b[m].var() / N / mb ** 2; cab = np.cov(a[m], b[m])[0, 1] / N / ma / mb
            v.append((mb / ma) ** 2 * (va + vb - 2 * cab))
        rec[n] = float(100 * np.mean(g)); rec[n + ' se'] = float(100 * np.sqrt(np.sum(v)) / 2)
    out.append(rec); print(rec, flush=True)
json.dump(out, open('shape_se.json', 'w'), indent=1)
