"""Caches the EWMA vectors (lambda = 0.1) of raw observations and VAR(1) residuals for all
Tennessee Eastman run sets ('split': VAR and Phase I fitted on training runs 1-250 only).
Usage: python te_cache.py [split]
Inputs: data/rieth2017/, data/te_cache/
Outputs: {OUT}/Rcor.npy, {OUT}/{name}_Z.npy, {OUT}/{name}_tot.npy, {OUT}/{name}_umax.npy
Paper: Sections 5.1-5.2.
"""
import numpy as np, json, sys, os, importlib.util
from sklearn.covariance import LedoitWolf
SPLIT = len(sys.argv) > 1 and sys.argv[1] == 'split'
spec = importlib.util.spec_from_file_location('rieth', 'rieth_res.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
OUT = '../data/te_cache' + ('_split' if SPLIT else ''); os.makedirs(OUT, exist_ok=True)
TR3 = R.runs('ff_train', 500)
if SPLIT:  # refit filter + Phase I on first half of training runs
    A = TR3[:250]; m0, s0 = A.reshape(-1, 52).mean(0), A.reshape(-1, 52).std(0, ddof=1)
    Y = ((A[:, 1:] - m0) / s0).reshape(-1, 52); Xl = ((A[:, :-1] - m0) / s0).reshape(-1, 52); Xa = np.c_[Xl, np.ones(len(Xl))]
    B = np.linalg.lstsq(Xa, Y, rcond=None)[0]
    def resid(X):
        S = (X - m0) / s0; E = np.zeros_like(S); E[:, 1:] = S[:, 1:] - (S[:, :-1] @ B[:52] + B[52]); return E
    TRr = resid(A)[:, 1:].reshape(-1, 52); mu, sd = TRr.mean(0), TRr.std(0, ddof=1)
    Rcor = LedoitWolf().fit((TRr - mu) / sd).covariance_
else:
    resid, mu, sd, Rcor = R.resid, R.mu, R.sd, R.Rcor
np.save(f'{OUT}/Rcor.npy', Rcor)
Rinv = np.linalg.inv(Rcor); Linv = np.linalg.inv(np.linalg.cholesky(Rcor)); lam = 0.1; c = (2 - lam) / lam
def cache(name, X):
    E = (resid(X) - mu) / sd; Z = np.empty(E.shape, np.float32); z = np.zeros((E.shape[0], 52))
    for t in range(E.shape[1]): z = lam * E[:, t] + (1 - lam) * z; Z[:, t] = z
    np.save(f'{OUT}/{name}_Z.npy', Z)
    np.save(f'{OUT}/{name}_tot.npy', (c * np.einsum('rti,ij,rtj->rt', Z, Rinv, Z)).astype(np.float32))
    np.save(f'{OUT}/{name}_umax.npy', (c * (Z @ Linv.T) ** 2).max(-1).astype(np.float32)); print(name, flush=True)
cache('fftest', R.runs('ff_test', 960))
cache('fftrain', TR3[250:] if SPLIT else TR3)
D = '../data/rieth2017/ftest'; X = np.load(f'{D}/X.npy', mmap_mode='r'); F = np.load(f'{D}/fault.npy')
for k in range(1, 21): cache(f'f{k}', np.asarray(X[np.flatnonzero(F == k)], np.float64).reshape(500, 960, 52))
print('DONE')
