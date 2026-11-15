"""Leave-one-covariance-structure-out cross-validation of the cubic fit and its residual
standard error (percentage points).
Inputs: collapse.json
Outputs: cv_struct.json
Paper: Section 4.2; Supplement S1.
"""
import json, numpy as np
C = json.load(open('collapse.json')); out = {}
for ch in ('BAS w1=0.50', 'BAS w1=0.25', 'LEWMA l=0.05'):
    y = np.array([c[ch] for c in C]); st = np.array([c['structure'] for c in C])
    for feat, x in (('logPR', np.log([c['pr'] for c in C])), ('maxd2', None)):
        if x is None: continue
        X = np.vander(x, 4); b = np.linalg.lstsq(X, y, rcond=None)[0]; res = y - X @ b
        pred = np.empty_like(y)
        for s in np.unique(st):
            m = st == s; bb = np.linalg.lstsq(X[~m], y[~m], rcond=None)[0]; pred[m] = X[m] @ bb
        out[f'{ch}|{feat}'] = {'R2': float(1 - (res**2).sum() / ((y - y.mean())**2).sum()), 'rse_pp': float(np.sqrt((res**2).sum() / (len(y) - 4))),
                               'R2_LOSO': float(1 - ((y - pred)**2).sum() / ((y - y.mean())**2).sum()), 'rmse_LOSO_pp': float(np.sqrt(((y - pred)**2).mean())), 'n_struct': int(len(np.unique(st)))}
print(json.dumps(out, indent=1)); json.dump(out, open('cv_struct.json', 'w'), indent=1)
