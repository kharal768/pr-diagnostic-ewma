"""Ablation of the balancing step: the two BAS-EWMA terms with one limit on their chi-square-
standardised scale (as in VS-MEWMA) against the balanced limits. p = 30, lambda = 0.05.
Inputs: none (simulation).
Outputs: ablation_clean.json
Paper: Section 4.3 (ablation).
"""
import json, numpy as np
from engine import arl
from calib import calibrate
P = 30; prm = np.array([0.05, 1.0, float(P)])
lf = lambda c: np.array([1 + c * np.sqrt(2), P + c * np.sqrt(2 * P)])
c, a0 = calibrate(6, prm, lf, 1.0, 30.0, p=P, R=20000, seed0=1, iters=26, maxT=20000)
row = {nz: float(arl(6, prm, lf(c), p=P, delta=0.75, tau=50, maxT=3000, R=20000, seed0=4321, nz=nz)[0]) for nz in (1, 2, 3, 5, 10, 30)}
out = {'c': c, 'arl0': a0, 'lim': lf(c).tolist(), 'ss_0.75': row}; print(out, flush=True)
json.dump(out, open('ablation_clean.json', 'w'), indent=1)
