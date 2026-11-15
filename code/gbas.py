"""G-BAS chart: chi-square-standardised largest group energy paired with the MEWMA total.

In-control steady-state sampling of the two terms and balanced calibration for a given
partition of the variables into groups.
Library module imported by the analysis scripts; writes no files.
Paper: Sections 2.3 and 2.4.
"""
import numpy as np
from calib import calibrate

def ic_samples(lam, g, p, R=6000, seed=777):
    """Steady-state in-control draws of (max-group Q, total Q), vectorised over replications."""
    rng = np.random.default_rng(seed)
    burn = int(10 / lam); gap = max(1, int(1 / lam)); takes = 30
    Z = np.zeros((R, p)); cst = (2 - lam) / lam
    mx, tot = [], []
    for t in range(burn + gap * takes):
        Z = lam * rng.standard_normal((R, p)) + (1 - lam) * Z
        if t >= burn and (t - burn) % gap == 0:
            Q = cst * (Z ** 2).reshape(R, p // g, g).sum(2)
            mx.append(Q.max(1)); tot.append(Q.sum(1))
    return np.sort(np.concatenate(mx)), np.sort(np.concatenate(tot))

def limfun_for(lam, g, p, w1):
    smx, stot = ic_samples(lam, g, p)
    def lf(c):
        a = 10.0 ** (-c)
        return [float(np.quantile(smx, 1 - a * w1 * 2)),
                float(np.quantile(stot, 1 - a * (1 - w1) * 2))]
    return lf

def calibrate_gbas(lam, g, p, w1, R=12000, maxT=3000):
    lf = limfun_for(lam, g, p, w1)
    c, a = calibrate(12, [lam, g], lf, 0.5, 4.5, p=p, R=R, iters=20, maxT=maxT)
    return lf(c), a
