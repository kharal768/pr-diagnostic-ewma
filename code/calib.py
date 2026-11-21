"""Control-limit calibration helpers.

`calibrate` finds the common limit scale giving a target zero-state in-control ARL (common
random numbers, bisection); `ss_profile` returns conditional steady-state out-of-control ARLs
over a grid of shift sizes and sparsity levels.
Library module imported by the analysis scripts; writes no files.
Paper: Section 2.4 (balanced calibration, Algorithm 1); used by most simulation scripts.
"""
import numpy as np
from engine import arl

ARL0 = 200.0

def calibrate(kind, prm, limfun, lo, hi, p=2, R=20000, seed0=1, iters=30, target=ARL0, maxT=20000):
    """Find scalar c so that zero-state in-control ARL(limfun(c)) = target (common random numbers)."""
    for _ in range(iters):
        c = 0.5 * (lo + hi)
        a = arl(kind, prm, limfun(c), p=p, R=R, seed0=seed0, maxT=maxT)[0]
        if a < target:
            lo = c
        else:
            hi = c
    c = 0.5 * (lo + hi)
    return c, arl(kind, prm, limfun(c), p=p, R=R, seed0=seed0, maxT=maxT)[0]

def ss_profile(kind, prm, lim, deltas, p=2, tau=50, R=20000, seed0=10_000_000, nz=1, dvec=None):
    return np.array([arl(kind, prm, lim, p=p, delta=d, tau=tau, R=R, seed0=seed0, nz=nz, dvec=dvec)[0]
                     for d in deltas])
