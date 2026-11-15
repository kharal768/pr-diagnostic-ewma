"""Negative control: fault-free testing runs treated as a pseudo-fault, same delay measure.
Inputs: rieth_res_cal.json, rieth_res_matched.json
Outputs: negctl.json
Paper: Section 5.2 (pseudo-fault); Supplement S5.
"""
import numpy as np, json, importlib.util
spec = importlib.util.spec_from_file_location('rieth', 'rieth_res.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
cal = json.load(open('rieth_res_cal.json'))['designs']; M = json.load(open('rieth_res_matched.json'))
adj = {k: (M[k]['matched_lim'] if k in M else v['lim']) for k, v in cal.items()}
D_ = {'MEWMA': (cal['MEWMA lam=0.1'], cal['MEWMA lam=0.1']['lim']), 'G-BAS': (cal['G-BAS lam=0.1 w1=0.3'], adj['G-BAS lam=0.1 w1=0.3']),
      'BAS-EWMA': (cal['BAS-EWMA lam=0.1 w1=0.4'], adj['BAS-EWMA lam=0.1 w1=0.4'])}
FT = R.runs('ff_test', 960); tt = R.terms(FT, 0.1); ON = 160; out = {}
res = {}
for n, (d, lim) in D_.items():
    al = R.alarm(d['kind'], tt, lim, d['w1']); post = al[:, ON:]
    res[n] = (np.where(post.any(1), np.argmax(post, 1) + 1, 801).astype(float), ~al[:, :ON].any(1))
dm, cm = res['MEWMA']
for n in ('G-BAS', 'BAS-EWMA'):
    dx, cx = res[n]; m = cm & cx; d = dx[m] - dm[m]
    out[n] = {'mewma': float(dm[m].mean()), 'chart': float(dx[m].mean()), 'ratio': float(dx[m].mean() / dm[m].mean()),
              't': float(d.mean() / (d.std(ddof=1) / np.sqrt(m.sum()))), 'n': int(m.sum())}
    print(n, out[n], flush=True)
json.dump(out, open('negctl.json', 'w'), indent=1)
