"""VAR order selected by BIC on the fault-free training runs (orders 1-10).
Inputs: none (simulation).
Outputs: var_order.json
Paper: Section 5.2 (VAR(10) check); Supplement S15.
"""
import numpy as np, json, importlib.util
spec = importlib.util.spec_from_file_location('rieth', 'rieth_res.py'); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
TR = R.runs('ff_train', 500); m0, s0 = TR.reshape(-1, 52).mean(0), TR.reshape(-1, 52).std(0, ddof=1); S = (TR - m0) / s0
out = {}; K = 10
Y = S[:, K:].reshape(-1, 52); n = len(Y)
for k in range(1, K + 1):
    X = np.concatenate([S[:, K - j:-j or None].reshape(-1, 52) for j in range(1, k + 1)] + [np.ones((n, 1))], 1)
    B = np.linalg.lstsq(X, Y, rcond=None)[0]; E = Y - X @ B; Sig = E.T @ E / n
    ld = np.linalg.slogdet(Sig)[1]; npar = B.size; out[k] = {'bic': float(ld + np.log(n) * npar / n), 'aic': float(ld + 2 * npar / n)}
    print(k, out[k], flush=True)
json.dump(out, open('var_order.json', 'w'), indent=1)
