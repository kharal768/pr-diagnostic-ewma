"""WaDi, step 1: cleaning, Phase I (standardisation, VAR(1) filter, shrinkage whitening) on the
normal days, control-loop groups and EWMA statistics.
Inputs: data/wadi/
Outputs: data/wadi/base.npz
Paper: Section 5.3; Supplement S18.
"""
import numpy as np, pandas as pd, json
from sklearn.covariance import LedoitWolf
N = pd.read_parquet('../data/wadi/normal10.parquet'); A = pd.read_parquet('../data/wadi/attack10.parquet')
lab = A.pop('Attack LABLE (1:No Attack, -1:Attack)').values
for D in (N, A): D.drop(columns=['Date', 'Time'], inplace=True)
cols = [c for c in N.columns if not any(k in c for k in ('STATUS', '_AL', '_AH', 'SPEED', '_AIT_')) and c in A.columns]
N, A = N[cols], A[cols]
ok = (N.isna().mean() < 0.05) & (A.isna().mean() < 0.05)
N, A = N.loc[:, ok].ffill().bfill(), A.loc[:, ok].ffill().bfill()
n0, n1 = int(0.3 * len(N)), int(0.75 * len(N)); P1 = N.iloc[n0:n1].values.astype(float)
sd = P1.std(0); keep = (sd > 1e-6) & (A.values.std(0) > 1e-6) & (np.unique(np.round(P1, 4), axis=0).shape[0] > 0)
nun = np.array([len(np.unique(np.round(P1[:, j], 3))) for j in range(P1.shape[1])]); keep &= nun > 20
cols = [c for c, k in zip(N.columns, keep) if k]; print('variables kept', len(cols))
X1, X2, XA = (D[cols].values.astype(float) for D in (N.iloc[n0:n1], N.iloc[n1:], A))
mu, s = X1.mean(0), X1.std(0)
Z1, Z2, ZA = ((X - mu) / s for X in (X1, X2, XA))
# VAR(1) with ridge, fitted on Phase I
Y, Xl = Z1[1:], Z1[:-1]; B = np.linalg.solve(Xl.T @ Xl + 0.1 * np.eye(Xl.shape[1]), Xl.T @ Y)
def res(Z): return Z[1:] - Z[:-1] @ B
E1 = res(Z1); C = LedoitWolf().fit(E1).covariance_; w, V = np.linalg.eigh(C); W = V @ np.diag(w ** -0.5) @ V.T
np.savez_compressed('../data/wadi/base.npz', E1=res(Z1) @ W, E2=res(Z2) @ W, EA=res(ZA) @ W, lab=lab[1:],
                    ZA=ZA[1:] , W=W, cols=np.array(cols))
print('shapes', E1.shape, res(Z2).shape, res(ZA).shape, 'lag-1 autocorr of whitened residuals', float(np.mean([np.corrcoef((res(Z1) @ W)[:-1, j], (res(Z1) @ W)[1:, j])[0, 1] for j in range(len(cols))])))
