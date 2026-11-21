"""Out-of-sample comparison of shape summaries: cubic trained on the 54 structured directions,
tested on the 48 random directions.
Inputs: collapse.json, diverse_dirs.json
Outputs: shape_summaries.json
Paper: Section 4.2; Supplement S1.
"""
import json, numpy as np
C = json.load(open('collapse.json')); src = open('collapse.py').read(); src = src[:src.index('F = json.load')]; ns = {}; exec(src, ns)
Dtr = [d / np.linalg.norm(d) for _, _, d in ns['directions']()]; Dte = [np.array(r['d']) for r in json.load(open('diverse_dirs.json'))]
Yte = {c: np.array([r[c] for r in json.load(open('diverse_dirs.json'))]) for c in ('BAS w1=0.50', 'BAS w1=0.25')}
Ytr = {c: np.array([r[c] for r in C]) for c in Yte}
def feats(d):
    w = d ** 2; w = w / w.sum(); nz = w[w > 0]
    return {'Hill q=2 (PR)': 1 / np.sum(w ** 2), 'Hill q=1 (exp entropy)': np.exp(-np.sum(nz * np.log(nz))), 'Hill q=inf (1/max loading)': 1 / w.max(),
            'l1/l2 ratio squared': np.sum(np.abs(d)) ** 2 / np.sum(d ** 2), 'top-3 share (inverse)': 1 / np.sort(w)[::-1][:3].sum()}
Ftr = [feats(d) for d in Dtr]; Fte = [feats(d) for d in Dte]; out = {}
for c in Yte:
    for f in Ftr[0]:
        xtr = np.log([r[f] for r in Ftr]); xte = np.log([r[f] for r in Fte]); b = np.polyfit(xtr, Ytr[c], 3)
        rm = float(np.sqrt(np.mean((np.polyval(b, xte) - Yte[c]) ** 2))); rin = float(np.sqrt(np.mean((np.polyval(b, xtr) - Ytr[c]) ** 2)))
        out[f'{c}|{f}'] = {'rmse_train_pp': rin, 'rmse_test_pp': rm}
        print(c, f, round(rin, 2), round(rm, 2))
pr = np.array([r['Hill q=2 (PR)'] for r in Fte]); out['test_pr_range'] = [float(pr.min()), float(pr.max())]; out['test_gain_range'] = [float(Yte['BAS w1=0.50'].min()), float(Yte['BAS w1=0.50'].max())]
json.dump(out, open('shape_summaries.json', 'w'), indent=1)
