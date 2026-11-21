"""Effect of Phase I estimation on the gain of BAS-EWMA over MEWMA.

Estimates mean and covariance from Phase I samples of size n, monitors with the estimated
regression adjustment, and records the gain curve against the participation ratio.
Usage: python phase1.py known | n  (Phase I sample size)
Inputs: none (simulation).
Outputs: phase1.json
Paper: Section 3.1 (estimated parameters).
"""
import json, sys, numpy as np

P, LAM, W1, ARL0 = 30, 0.05, 0.25, 200.0
D = [0.75, 1.0]
TAU, MAXT, RCAL, ROOC = 50, 1200, 3000, 4000
CST = (2 - LAM) / LAM


def transforms(n, R, rng):
    """Per-replication Phase I estimates: returns A (R,p,p) with A = L^{-1}, and mu_hat (R,p)."""
    if n is None:                                    # known parameters
        return np.broadcast_to(np.eye(P), (R, P, P)).copy(), np.zeros((R, P))
    Y = rng.standard_normal((R, n, P))
    mu = Y.mean(axis=1)
    Yc = Y - mu[:, None, :]
    S = np.einsum("rip,riq->rpq", Yc, Yc) / (n - 1)
    A = np.linalg.inv(np.linalg.cholesky(S))
    return A, mu


def stat_paths(A, mu, T, rng, delta=0.0, dvec=None, tau=0):
    """In-control or shifted statistic paths: returns (T,R) arrays of max and sum."""
    R = A.shape[0]
    Z = np.zeros((R, P))
    mx = np.empty((T, R), np.float32)
    sm = np.empty((T, R), np.float32)
    shift = np.zeros(P) if dvec is None else delta * dvec
    for t in range(T):
        X = rng.standard_normal((R, P))
        if t >= tau:
            X += shift          # in control until tau, shifted afterwards
        W = np.einsum("rpq,rq->rp", A, X - mu)
        Z = LAM * W + (1 - LAM) * Z
        U = CST * Z * Z
        mx[t] = U.max(axis=1)
        sm[t] = U.sum(axis=1)
    return mx, sm


def first_pass(mx, sm, h1, hp, start=0):
    sig = (mx > h1) | (sm > hp)
    if start:
        pre = sig[:start].any(axis=0)
        post = sig[start:]
    else:
        pre = np.zeros(sig.shape[1], bool)
        post = sig
    has = post.any(axis=0)
    rl = np.where(has, post.argmax(axis=0) + 1, post.shape[0]).astype(float)
    rl[pre] = np.nan
    return rl


def calibrate(mx, sm, w1, only=None):
    """Scale marginal quantiles until the unconditional in-control ARL hits ARL0.
    only='sum' gives a MEWMA-style chart (dense term alone)."""
    qm, qs = np.sort(mx.ravel()), np.sort(sm.ravel())
    def lims(c):
        a = 10.0 ** (-c)
        if only == "sum":
            return np.inf, float(np.quantile(qs, 1 - a))
        return (float(np.quantile(qm, 1 - a * w1 * 2)),
                float(np.quantile(qs, 1 - a * (1 - w1) * 2)))
    lo, hi = 0.3, 5.0
    for _ in range(26):
        c = 0.5 * (lo + hi)
        h1, hp = lims(c)
        lo, hi = (c, hi) if np.nanmean(first_pass(mx, sm, h1, hp)) < ARL0 else (lo, c)
    h1, hp = lims(0.5 * (lo + hi))
    return h1, hp, float(np.nanmean(first_pass(mx, sm, h1, hp)))


if __name__ == "__main__":
    import os
    arg = sys.argv[1]
    n = None if arg == "known" else int(arg)
    tag = "known" if n is None else f"n={n}"
    GRID = [1, 2, 3, 4, 5, 7]
    out = json.load(open("phase1.json")) if os.path.exists("phase1.json") else {}
    rng = np.random.default_rng(9000 + (n or 0))
    A, mu = transforms(n, RCAL, rng)
    mx, sm = stat_paths(A, mu, MAXT, rng)
    h1, hp, a_bas = calibrate(mx, sm, W1)
    _, hp_only, a_mew = calibrate(mx, sm, W1, only="sum")
    print(f"{tag:7s} BAS h=({h1:.2f},{hp:.2f}) ARL0={a_bas:.1f} | MEWMA h={hp_only:.2f} ARL0={a_mew:.1f}", flush=True)
    g = []
    for nz in GRID:
        d = np.zeros(P); d[:nz] = 1 / np.sqrt(nz)
        r2 = np.random.default_rng(77000 + (n or 0) * 10 + nz)
        A2, mu2 = transforms(n, ROOC, r2)
        vals = []
        for dl in D:
            m2, s2 = stat_paths(A2, mu2, TAU + 300, r2, delta=dl, dvec=d, tau=TAU)
            b = np.nanmean(first_pass(m2, s2, np.inf, hp_only, start=TAU))
            a = np.nanmean(first_pass(m2, s2, h1, hp, start=TAU))
            vals.append(a / b)
        v = float(100 * (1 - np.mean(vals))); g.append(v)
        print(f"    PR={nz}  {v:6.2f}%", flush=True)
    gg, pr = np.array(g), np.array(GRID, float)
    neg = np.where(gg < 0)[0]
    be = float(np.exp(np.interp(0.0, [gg[neg[0]], gg[neg[0] - 1]],
                                [np.log(pr[neg[0]]), np.log(pr[neg[0] - 1])]))) if len(neg) and neg[0] > 0 else float("nan")
    out[tag] = dict(h1=h1, hp=hp, hp_only=hp_only, arl0_bas=a_bas, arl0_mew=a_mew,
                    grid=GRID, gains=g, be=be)
    print(f"  -> {tag}: break-even {be:.2f}", flush=True)
    json.dump(out, open("phase1.json", "w"), indent=1)
