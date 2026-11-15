"""Effect of serial correlation on calibration and on the diagnostic.

AR(1) observations with coefficient phi, charts with independent-data limits, and charts on
one-step residuals.
Usage: python serial.py phi [resid]
Inputs: none (simulation).
Outputs: serial.json
Paper: Section 3.1 (autocorrelation); Supplement S14.
"""
import json, os, sys, numpy as np

P, LAM, W1, ARL0 = 30, 0.05, 0.25, 200.0
D = [0.75, 1.0]
TAU, MAXT, RCAL, ROOC = 50, 1500, 4000, 5000
CST = (2 - LAM) / LAM
IID_LIMITS = (13.92, 48.33)      # limits calibrated for independent data (phi = 0)


def paths(phi, R, T, rng, delta=0.0, dvec=None, tau=0, resid=False):
    w = rng.standard_normal((R, P))          # start in the stationary distribution
    Z = np.zeros((R, P))
    mx = np.empty((T, R)); sm = np.empty((T, R))
    s = np.sqrt(1 - phi ** 2)
    shift = np.zeros(P) if dvec is None else delta * dvec
    Xprev = w.copy()
    for t in range(T):
        w = phi * w + s * rng.standard_normal((R, P))
        X = w + shift if t >= tau else w
        if resid:
            # monitor one-step-ahead residuals, rescaled to unit variance: iid by construction,
            # but a sustained shift is attenuated to (1 - phi) of its size
            obs = (X - phi * Xprev) / s
            Xprev = X
        else:
            obs = X
        Z = LAM * obs + (1 - LAM) * Z
        U = CST * Z * Z
        mx[t] = U.max(axis=1); sm[t] = U.sum(axis=1)
    return mx, sm


def first_pass(mx, sm, h1, hp, start=0):
    sig = (mx > h1) | (sm > hp)
    pre = sig[:start].any(axis=0) if start else np.zeros(sig.shape[1], bool)
    post = sig[start:] if start else sig
    has = post.any(axis=0)
    rl = np.where(has, post.argmax(axis=0) + 1, post.shape[0]).astype(float)
    rl[pre] = np.nan
    return rl


def calibrate(mx, sm, only=None):
    qm, qs = mx.ravel(), sm.ravel()
    def lims(c):
        a = 10.0 ** (-c)
        if only == "sum":
            return np.inf, float(np.quantile(qs, 1 - a))
        return (float(np.quantile(qm, 1 - a * W1 * 2)),
                float(np.quantile(qs, 1 - a * (1 - W1) * 2)))
    lo, hi = 0.3, 6.0
    for _ in range(28):
        c = 0.5 * (lo + hi)
        h1, hp = lims(c)
        lo, hi = (c, hi) if np.nanmean(first_pass(mx, sm, h1, hp)) < ARL0 else (lo, c)
    h1, hp = lims(0.5 * (lo + hi))
    return h1, hp, float(np.nanmean(first_pass(mx, sm, h1, hp)))


if __name__ == "__main__":
    phi = float(sys.argv[1])
    RES = len(sys.argv) > 2 and sys.argv[2] == 'resid'
    TAG = f"{phi}{'-resid' if RES else ''}"
    GRID = [1, 2, 3, 4, 5, 7]
    out = json.load(open("serial.json")) if os.path.exists("serial.json") else {}
    rng = np.random.default_rng(42000 + int(phi * 100))
    mx, sm = paths(phi, RCAL, MAXT, rng, resid=RES)
    arl_iid = float(np.nanmean(first_pass(mx, sm, *IID_LIMITS)))
    h1, hp, a_bas = calibrate(mx, sm)
    _, hp_only, a_mew = calibrate(mx, sm, only="sum")
    print(f"phi={TAG}: ARL0 with iid limits = {arl_iid:.1f} (nominal 200)", flush=True)
    print(f"   recalibrated BAS ({h1:.2f}, {hp:.2f}) ARL0={a_bas:.1f} | MEWMA {hp_only:.2f} ARL0={a_mew:.1f}", flush=True)
    g = []
    for nz in GRID:
        d = np.zeros(P); d[:nz] = 1 / np.sqrt(nz)
        r2 = np.random.default_rng(66000 + int(phi * 100) * 10 + nz)
        vals = []
        for dl in D:
            m2, s2 = paths(phi, ROOC, TAU + 300, r2, delta=dl, dvec=d, tau=TAU, resid=RES)
            b = np.nanmean(first_pass(m2, s2, np.inf, hp_only, start=TAU))
            a = np.nanmean(first_pass(m2, s2, h1, hp, start=TAU))
            vals.append(a / b)
        v = float(100 * (1 - np.mean(vals))); g.append(v)
        print(f"   PR={nz}  {v:6.2f}%", flush=True)
    gg, pr = np.array(g), np.array(GRID, float)
    neg = np.where(gg < 0)[0]
    be = float(np.exp(np.interp(0.0, [gg[neg[0]], gg[neg[0] - 1]],
                                [np.log(pr[neg[0]]), np.log(pr[neg[0] - 1])]))) if len(neg) and neg[0] > 0 else float("nan")
    out[TAG] = dict(phi=phi, resid=RES, arl0_iid_limits=arl_iid, h1=h1, hp=hp, hp_only=hp_only,
                         grid=GRID, gains=g, be=be)
    print(f"   -> phi={TAG} break-even {be:.2f}", flush=True)
    json.dump(out, open("serial.json", "w"), indent=1)
