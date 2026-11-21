"""False-alarm matching of the residual charts on the 500 fault-free training runs: common limit
scale that matches MEWMA's mean censored first-alarm time.
Inputs: rieth_res_cal.json
Outputs: rieth_res_matched.json
Paper: Supplement S5 and S8.
"""
import numpy as np, json, importlib.util
spec = importlib.util.spec_from_file_location('rieth', 'rieth_res.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
cal = json.load(open('rieth_res_cal.json'))['designs']
TR = R.runs('ff_train', 500); tt = R.terms(TR, 0.1)
def fa(name, lim): d = cal[name]; return R.first_alarm(R.alarm(d['kind'], tt, lim, d['w1'])).astype(float)
base = fa('MEWMA lam=0.1', cal['MEWMA lam=0.1']['lim']); out = {'MEWMA': float(base.mean())}
for name in ('G-BAS lam=0.1 w1=0.3', 'BAS-EWMA lam=0.1 w1=0.4'):
    L = np.array(cal[name]['lim']); m0 = fa(name, L).mean()
    lo, hi = 0.9, 1.15
    for _ in range(30):
        s = 0.5 * (lo + hi); m = fa(name, L * s).mean()
        lo, hi = (s, hi) if m < base.mean() else (lo, s)
    s = 0.5 * (lo + hi)
    out[name] = {'mean_first_alarm_as_calibrated': float(m0), 'ratio_to_mewma': float(m0 / base.mean()),
                 'matching_scale': float(s), 'matched_lim': (L * s).tolist(), 'matched_mean': float(fa(name, L * s).mean())}
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name].items()}, flush=True)
print('MEWMA mean censored first false alarm', round(out['MEWMA'], 1))
json.dump(out, open('rieth_res_matched.json', 'w'), indent=1)
