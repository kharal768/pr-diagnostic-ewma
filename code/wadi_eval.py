"""WaDi, step 2: charts calibrated to an in-control ARL of 200 samples on the attack-free periods,
pseudo-attack false-alarm check, detection delay per attack and participation ratio of each
attack's shift.
Usage: python wadi_eval.py [mode]  (default: attackday)
Inputs: data/wadi/
Outputs: wadi_res_{MODE}.json
Paper: Section 5.3; Supplement S18.
"""
import numpy as np, json
from scipy.stats import chi2
from te_lib import first_alarm, alarm
D = np.load('../data/wadi/base.npz', allow_pickle=True); cols = list(D['cols']); p = len(cols); LAM = 0.05; cst = (2 - LAM) / LAM
pref = [c.split('_')[0] + (c.split('_')[2][0] if len(c.split('_')) > 2 and c.split('_')[2][:1].isdigit() else '') for c in cols]; G = [[j for j in range(p) if pref[j] == g] for g in sorted(set(pref))]
def ewma(E):
    z = np.zeros(p); U = np.empty_like(E)
    for t in range(len(E)): z = LAM * E[t] + (1 - LAM) * z; U[t] = cst * z * z
    return U
def stats(U):
    tot = U.sum(1); gq = np.stack([U[:, g].sum(1) for g in G], 1)
    gstd = np.max(np.nan_to_num(-chi2.logsf(gq, [len(g) for g in G]), nan=1e3, posinf=1e3), 1)
    return {'MEWMA': (None, tot), 'BAS-EWMA': (U.max(1), tot), 'G-BAS': (gstd, tot),
            'mixture': (None, np.log1p((3 / p) * np.expm1(np.minimum(U, 600) / 2)).sum(1)), 'soft-threshold': (None, np.maximum(U - 2, 0).sum(1))}
W1 = {'BAS-EWMA': 0.4, 'G-BAS': 0.3}
def runs(x, T=1500, step=150): return None if x is None else np.stack([x[i:i + T] for i in range(0, len(x) - T, step)])
def calib(first, tot, w1, target=200.0):
    def lims(a): return [np.quantile(tot, 1 - a)] if first is None else [np.quantile(first, 1 - a * w1), np.quantile(tot, 1 - a * (1 - w1))]
    lo, hi = 1e-7, 0.5
    for _ in range(45):
        a = np.sqrt(lo * hi); lo, hi = (lo, a) if first_alarm(alarm(runs(first), runs(tot), lims(a))).mean() < target else (a, hi)
    return lims(np.sqrt(lo * hi))
S2 = stats(ewma(D['E2'])); SA = stats(ewma(D['EA'])); lab = D['lab']
import sys; MODE = sys.argv[1] if len(sys.argv) > 1 else 'attackday'
if MODE == 'heldout':
    lim = {k: calib(f, t, W1.get(k)) for k, (f, t) in S2.items()}
a = (lab == -1).astype(int); d = np.diff(np.r_[0, a, 0]); st, en = np.where(d == 1)[0], np.where(d == -1)[0]
far = {}
quiet = np.ones(len(lab), bool)
for s, e in zip(st, en): quiet[max(0, s - 360):e + 360] = False
if MODE == 'attackday':
    T = 600; idx = [i for i in range(0, len(lab) - T, 60) if quiet[i:i + T].all()]
    def runs(x, T=T, step=None): return None if x is None else np.stack([x[i:i + T] for i in idx])
    lim = {k: calib(f, t, W1.get(k)) for k, (f, t) in SA.items()}
    print('calibration windows', len(idx))
for k, (f, t) in SA.items():
    al = alarm(f, t, lim[k]); far[k] = float(al[quiet].mean())
res = {'lim': {k: list(map(float, v)) for k, v in lim.items()}, 'groups': [len(g) for g in G], 'false_alarm_rate_attack_day_quiet': far,
       'nominal_rate_heldout': {k: float(alarm(f, t, lim[k]).mean()) for k, (f, t) in S2.items()}, 'attacks': []}
ZA = D['ZA']; base = ZA[quiet].mean(0); EA = D['EA']; WU = 150; H = 360
def delays(s0):
    seg = EA[s0 - WU:s0 + H]; out = {}
    for k, (f, t) in stats(ewma(seg)).items():
        al = alarm(f, t, lim[k]); pre = al[WU - 30:WU].any(); post = al[WU:]
        out[k] = (int(np.argmax(post)) + 1 if post.any() else H + 1, bool(pre))
    return out
for i, (s0, e) in enumerate(zip(st, en)):
    shift = ZA[s0:e].mean(0) - base; dW = D['W'] @ shift; dn = dW / np.linalg.norm(dW); pr = 1 / np.sum(dn ** 4)
    dl = delays(s0)
    row = {'start': int(s0), 'len': int(e - s0), 'pr': float(pr), 'size': float(np.linalg.norm(dW)), 'top_vars': [str(cols[j]) for j in np.argsort(-np.abs(dW))[:3]],
           'delay': {k: v[0] for k, v in dl.items()}, 'alarm_before': {k: v[1] for k, v in dl.items()}}
    res['attacks'].append(row); print(i, row['len'], round(pr, 1), row['delay'], [k for k, v in row['alarm_before'].items() if v], flush=True)
rng = np.random.default_rng(1); qs = [q for q in np.where(quiet)[0] if q > WU and q + H < len(lab) and quiet[q:q + H].all()]
pick = rng.choice(qs, 200, replace=False); ps = {k: [] for k in SA}
for q in pick:
    for k, v in delays(int(q)).items(): ps[k].append(v[0])
res['pseudo_attack'] = {k: {'mean_delay': float(np.mean(v)), 'frac_within_30': float(np.mean(np.array(v) <= 30))} for k, v in ps.items()}
print('pseudo-attack', res['pseudo_attack'])
json.dump(res, open(f'wadi_res_{MODE}.json', 'w'), indent=1)
