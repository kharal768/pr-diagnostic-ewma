"""Practitioner tool: gain of BAS-EWMA over MEWMA, 1 - ARL/ARL_MEWMA (positive = faster), as a
function of the participation ratio for a user-specified configuration (p, lambda, w1, delta,
in-control ARL), and the break-even participation ratio.
Usage: python gain_curve.py p lambda w1
Inputs: none (simulation).
Outputs: printed to the console.
Paper: Algorithm 2 (procedure for using the diagnostic).
"""
import numpy as np
from engine import arl
from calib import calibrate
from bas_ewma import design

def gain_curve(p, lam, w1, deltas=(0.75, 1.0), arl0=200.0, prs=None, R_cal=12000, R=20000, tau=50, maxT=3000, seed=4321):
    import calib; calib.ARL0 = arl0
    c, _ = calibrate(0, [lam], lambda h: [h], 1.0, 20.0 * p, p=p, R=R_cal, seed0=7, iters=24, target=arl0, maxT=20000)
    lim_m = [c]
    lf = design(lam, [1, p], p, w=[w1, 1 - w1]); prm = [lam, 1, p]
    cb, _ = calibrate(6, prm, lf, 0.5, 4.5, p=p, R=R_cal, seed0=7, iters=24, target=arl0, maxT=20000); lim_b = lf(cb)
    prs = prs or sorted(set([1, 2, 3, 5, 8, 12, 20, p]) & set(range(1, p + 1)))
    gain = {}
    for k in prs:
        a = [arl(0, [lam], lim_m, p=p, delta=d, tau=tau, maxT=maxT, R=R, seed0=seed, nz=k)[0] for d in deltas]
        b = [arl(6, prm, lim_b, p=p, delta=d, tau=tau, maxT=maxT, R=R, seed0=seed, nz=k)[0] for d in deltas]
        gain[k] = float(100 * np.mean([1 - y / x for x, y in zip(a, b)]))
    ks = sorted(gain); g = np.array([gain[k] for k in ks]); be = None
    for i in range(len(ks) - 1):
        if g[i] > 0 >= g[i + 1]:
            x0, x1 = np.log(ks[i]), np.log(ks[i + 1]); be = float(np.exp(x0 + g[i] * (x1 - x0) / (g[i] - g[i + 1]))); break
    return {'limits_mewma': lim_m, 'limits_bas': list(map(float, lim_b)), 'gain': gain, 'break_even': be}

if __name__ == '__main__':
    import sys, json
    p, lam, w1 = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    print(json.dumps(gain_curve(p, lam, w1), indent=1))
