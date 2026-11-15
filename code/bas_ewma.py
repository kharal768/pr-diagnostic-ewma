"""BAS-EWMA chart: balanced calibration of the largest-coordinate term and the MEWMA total.

Splits the false-alarm budget between the two terms in the ratio w1:(1-w1) on their marginal
steady-state tail probabilities and scales both limits to the target in-control ARL.
Library module imported by the analysis scripts; writes no files.
Paper: Sections 2.2 and 2.4.
"""
import numpy as np
from engine import topsum_samples
from calib import calibrate

def design(lam, svals, p, R=4000, seed0=777, w=None):
    svals = np.asarray(svals, float)
    smp = topsum_samples(lam, svals, p, R, int(10 / lam), max(1, int(1 / lam)), 50, seed0)
    sorted_cols = [np.sort(smp[:, j]) for j in range(len(svals))]
    def limfun(c):  # c = -log10(tail prob) of each term's marginal steady-state exceedance
        a = 10.0 ** (-c)
        ww = np.ones(len(sorted_cols)) / len(sorted_cols) if w is None else np.asarray(w)
        return [float(np.quantile(col, 1 - a * ww[j] * len(ww))) for j, col in enumerate(sorted_cols)]
    return limfun

def calibrate_bas(lam, svals, p, R=20000, w=None, maxT=20000):
    lf = design(lam, svals, p, w=w)
    prm = [lam] + list(svals)
    c, a = calibrate(6, prm, lf, 0.5, 4.5, p=p, R=R, iters=24, maxT=maxT)
    return prm, lf(c), a
