"""Participation ratio of the mean post-fault signal for each fault, on observations and on VAR(1)
residuals, and its correlation with the gain.
Inputs: data/rieth2017/
Outputs: rieth_pr.json
Paper: Section 5.2 (diagnostic fault by fault).
"""
import numpy as np, json, importlib.util
def load(f):
    s = importlib.util.spec_from_file_location('m'+f, f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
Raw, Res = load('rieth.py'), load('rieth_res.py')
D = '../data/rieth2017/ftest'; X = np.load(f'{D}/X.npy', mmap_mode='r'); F = np.load(f'{D}/fault.npy')
def pr(v): d = v / np.linalg.norm(v); return float(1 / (d ** 4).sum())
out = {}
for k in range(1, 21):
    Xk = np.asarray(X[np.flatnonzero(F == k)], np.float64).reshape(500, 960, 52)
    a = ((Raw.Zs(Xk[:, 160:180]) @ Raw.Linv.T)).mean((0, 1))
    E = Res.resid(Xk)[:, 160:180]; b = (Res.Zs(E) @ Res.Linv.T).mean((0, 1))
    out[k] = {'raw': pr(a), 'resid': pr(b)}; print(k, round(out[k]['raw'], 1), round(out[k]['resid'], 1), flush=True)
json.dump(out, open('rieth_pr.json', 'w'), indent=1)
r = np.array([[v['raw'], v['resid']] for v in out.values()]); print('median PR raw', np.median(r[:, 0]), 'resid', np.median(r[:, 1]), 'resid lower in', int((r[:, 1] < r[:, 0]).sum()), 'of 20')
