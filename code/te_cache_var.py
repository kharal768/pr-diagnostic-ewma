"""Caches residual EWMA arrays for a VAR(k) filter fitted on all fault-free training runs.
Usage: python te_cache_var.py k  (VAR order)
Inputs: data/rieth2017/, data/te_cache_raw/, data/te_cache_var{k}/
Outputs: {OUT}/Rcor.npy, {OUT}/{name}_Z.npy, {OUT}/{name}_tot.npy, {OUT}/{name}_umax.npy
Paper: Section 5.2 (VAR(10) check); Supplement S15.
"""
import numpy as np, sys, os, importlib.util
from sklearn.covariance import LedoitWolf
k = int(sys.argv[1]); spec = importlib.util.spec_from_file_location('rieth', 'rieth_res.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
OUT = f'../data/te_cache_var{k}' if k else '../data/te_cache_raw'; os.makedirs(OUT, exist_ok=True)
TR3 = R.runs('ff_train', 500); m0, s0 = TR3.reshape(-1, 52).mean(0), TR3.reshape(-1, 52).std(0, ddof=1)
def lagX(S):
    T = S.shape[1]; return np.concatenate([S[:, k - j:T - j] for j in range(1, k + 1)], 2)
S = (TR3 - m0) / s0
if k:
    Y = S[:, k:].reshape(-1, 52); X = lagX(S).reshape(-1, 52 * k); X = np.c_[X, np.ones(len(X))]
    B = np.linalg.lstsq(X, Y, rcond=None)[0]; del X, Y
def resid(Xr):
    if not k: return (Xr - m0) / s0
    S = (Xr - m0) / s0; E = np.zeros_like(S); L = lagX(S); E[:, k:] = S[:, k:] - (L @ B[:-1] + B[-1]); return E
TRr = resid(TR3)[:, k:].reshape(-1, 52); mu, sd = TRr.mean(0), TRr.std(0, ddof=1); Rcor = LedoitWolf().fit((TRr - mu) / sd).covariance_; np.save(f'{OUT}/Rcor.npy', Rcor)
Rinv = np.linalg.inv(Rcor); Linv = np.linalg.inv(np.linalg.cholesky(Rcor)); lam = 0.1; c = (2 - lam) / lam
def cache(name, X):
    E = (resid(X) - mu) / sd
    if k: E[:, :k] = 0
    Z = np.empty(E.shape, np.float32); z = np.zeros((E.shape[0], 52))
    for t in range(E.shape[1]): z = lam * E[:, t] + (1 - lam) * z; Z[:, t] = z
    np.save(f'{OUT}/{name}_Z.npy', Z); np.save(f'{OUT}/{name}_tot.npy', (c * np.einsum('rti,ij,rtj->rt', Z, Rinv, Z)).astype(np.float32))
    np.save(f'{OUT}/{name}_umax.npy', (c * (Z @ Linv.T) ** 2).max(-1).astype(np.float32)); print(name, flush=True)
cache('fftest', R.runs('ff_test', 960)); cache('fftrain', TR3)
D = '../data/rieth2017/ftest'; Xd = np.load(f'{D}/X.npy', mmap_mode='r'); F = np.load(f'{D}/fault.npy')
for f in range(1, 21): cache(f'f{f}', np.asarray(Xd[np.flatnonzero(F == f)], np.float64).reshape(500, 960, 52))
print('DONE')
