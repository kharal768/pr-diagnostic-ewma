"""Numba run-length engine for all multivariate mean-shift charts in the paper.

Simulates X_t ~ N_p(0, I) in the monitored (whitened) coordinates, with a mean shift of
Mahalanobis length delta after the change point tau, and returns run lengths for the chart
selected by an integer `kind` (MEWMA, MCUSUM, BAS-EWMA, G-BAS, VS-MEWMA, LEWMA, top-r CUSUM,
thresholded and mixture EWMA, LAR-EWMA, MSGL-EWMA).
Library module imported by the analysis scripts; writes no files.
Paper: used by every simulation in Sections 3-4 and the supplement.
"""
import numpy as np
from numba import njit, prange

@njit(cache=True)
def _one_run(kind, prm, lim, p, delta, tau, maxT, seed, dvec):
    np.random.seed(seed)
    m = lim.shape[0]
    x = np.zeros(p)
    # state
    nst = max(m, 1)
    if kind == 11 or kind == 14:
        nst = 2
    Z = np.zeros((nst, p))
    S = np.zeros((nst, p))
    mu_hat = np.zeros(p)
    acu = 0.0
    w = 1
    if kind == 3:
        w = int(prm[0])
    buf = np.zeros((w + 1, p))   # cumulative sums ring buffer for GLR
    csum = np.zeros(p)
    hist = np.zeros((w + 1, p))
    for t in range(1, tau + maxT + 1):
        for i in range(p):
            x[i] = np.random.standard_normal()
        if t > tau:
            for i in range(p):
                x[i] += delta * dvec[i]
        sig = False
        if kind == 0:
            for j in range(m):
                lam = prm[j]
                q = 0.0
                for i in range(p):
                    Z[j, i] = lam * x[i] + (1 - lam) * Z[j, i]
                    q += Z[j, i] * Z[j, i]
                if (2 - lam) / lam * q > lim[j]:
                    sig = True
        elif kind == 1:
            for j in range(m):
                k = prm[j]
                c = 0.0
                for i in range(p):
                    c += (S[j, i] + x[i]) ** 2
                c = np.sqrt(c)
                f = 0.0
                if c > k:
                    f = 1 - k / c
                q = 0.0
                for i in range(p):
                    S[j, i] = (S[j, i] + x[i]) * f
                    q += S[j, i] ** 2
                if np.sqrt(q) > lim[j]:
                    sig = True
        elif kind == 2 or kind == 4:
            lh = prm[0]; dmin = prm[1]
            nrm = 0.0
            for i in range(p):
                nrm += mu_hat[i] ** 2
            nrm = np.sqrt(nrm)
            # predicted shift direction/size from past data, size floored at dmin
            inc = 0.0
            if nrm > 1e-12:
                sc = max(nrm, dmin) / nrm
                d2 = 0.0
                for i in range(p):
                    di = mu_hat[i] * sc
                    inc += di * x[i]
                    d2 += di * di
                inc -= 0.5 * d2
            acu = max(0.0, acu + inc)
            if acu > lim[0]:
                sig = True
            for i in range(p):
                mu_hat[i] = lh * x[i] + (1 - lh) * mu_hat[i]
            if kind == 4:
                lam = prm[2]
                q = 0.0
                for i in range(p):
                    Z[0, i] = lam * x[i] + (1 - lam) * Z[0, i]
                    q += Z[0, i] * Z[0, i]
                if (2 - lam) / lam * q > lim[1]:
                    sig = True
        elif kind == 3:
            # store cumulative sums C_t in ring buffer of size w+1
            for i in range(p):
                csum[i] += x[i]
                hist[t % (w + 1), i] = csum[i]
            best = 0.0
            kmax = min(w, t)
            for n in range(1, kmax + 1):
                idx = (t - n) % (w + 1)
                q = 0.0
                for i in range(p):
                    if t - n == 0:
                        d = csum[i]
                    else:
                        d = csum[i] - hist[idx, i]
                    q += d * d
                v = q / (2.0 * n)
                if v > best:
                    best = v
            if best > lim[0]:
                sig = True
        elif kind == 5:
            # Adaptive-sparsity EWMA (AS-EWMA): per-variable EWMAs, standardized squares u_i,
            # statistic = max_s (sum of s largest u_i - s) / sqrt(2 s)
            lam = prm[0]
            u = np.empty(p)
            cst = (2 - lam) / lam
            for i in range(p):
                Z[0, i] = lam * x[i] + (1 - lam) * Z[0, i]
                u[i] = cst * Z[0, i] * Z[0, i]
            u = -np.sort(-u)
            cs = 0.0
            best = -1e18
            for s_ in range(p):
                cs += u[s_]
                v = (cs - (s_ + 1)) / np.sqrt(2.0 * (s_ + 1))
                if v > best:
                    best = v
            if best > lim[0]:
                sig = True
        elif kind == 10:
            # Restricted-selection variant: the sparse term maximises over the first mm
            # coordinates only, while the dense term pools all p coordinates.
            # prm = [lam, mm]; lim = [h_max, h_sum]
            lam = prm[0]
            mm = int(prm[1])
            cst = (2 - lam) / lam
            mx = -1.0e18
            tot = 0.0
            for i in range(p):
                Z[0, i] = lam * x[i] + (1 - lam) * Z[0, i]
                ui = cst * Z[0, i] * Z[0, i]
                tot += ui
                if i < mm and ui > mx:
                    mx = ui
            if mx > lim[0] or tot > lim[1]:
                sig = True
        elif kind == 6 and m == 2 and prm[1] == 1.0 and prm[2] == p:
            # Fast path for the common design S = {1, p}: the largest standardized squared
            # EWMA and their total, both obtainable without sorting. Identical to the
            # general branch below, O(p) instead of O(p log p).
            lam = prm[0]
            cst = (2 - lam) / lam
            mx = -1.0e18
            tot = 0.0
            for i in range(p):
                Z[0, i] = lam * x[i] + (1 - lam) * Z[0, i]
                ui = cst * Z[0, i] * Z[0, i]
                tot += ui
                if ui > mx:
                    mx = ui
            if mx > lim[0] or tot > lim[1]:
                sig = True
        elif kind == 6:
            # Balanced AS-EWMA: terms T_s = sum of s largest standardized squared EWMAs,
            # for s in the set prm[1:], each with its own limit lim[j]
            lam = prm[0]
            u = np.empty(p)
            cst = (2 - lam) / lam
            for i in range(p):
                Z[0, i] = lam * x[i] + (1 - lam) * Z[0, i]
                u[i] = cst * Z[0, i] * Z[0, i]
            u = -np.sort(-u)
            cs = 0.0
            j = 0
            for s_ in range(p):
                cs += u[s_]
                if j < m and s_ + 1 == int(prm[1 + j]):
                    if cs > lim[j]:
                        sig = True
                    j += 1
        elif kind == 7:
            # LEWMA-type chart (Zou & Qiu, 2009) with Sigma = I: adaptive-LASSO estimate of the
            # shift at penalty thresholds g = gamma/2 (closed form), statistic
            # max_g (V_g - E_g)/S_g with V_g = (u'm_g)^2 / m_g'm_g; prm = [lam, E_1..E_q, S_1..S_q, g_1..g_q]
            lam = prm[0]
            q = (prm.shape[0] - 1) // 3
            cst = np.sqrt((2 - lam) / lam)
            u = np.empty(p)
            for i in range(p):
                Z[0, i] = lam * x[i] + (1 - lam) * Z[0, i]
                u[i] = cst * Z[0, i]
            best = -1e18
            for j in range(q):
                g = prm[1 + 2 * q + j]
                num = 0.0
                den = 0.0
                for i in range(p):
                    a2 = u[i] * u[i]
                    if a2 > g:
                        mi = (a2 - g) / abs(u[i])
                        num += abs(u[i]) * mi
                        den += mi * mi
                v = 0.0
                if den > 0:
                    v = num * num / den
                st = (v - prm[1 + j]) / prm[1 + q + j]
                if st > best:
                    best = st
            if best > lim[0]:
                sig = True
        elif kind == 8:
            # LEWMA (Zou & Qiu, 2009), Sigma = I: adaptive-LASSO path breakpoints with k = 1..q active
            # components; with |u| sorted a_1>=a_2>=..., the k-active estimate shrinks a_j by a_{k+1}^2/a_j
            # (k = p: no shrinkage, MEWMA direction). V_k = (u'm)^2/(m'm), standardized by in-control
            # mean/sd; statistic = max_k. prm = [lam, E_1..E_q, S_1..S_q]
            lam = prm[0]
            q = (prm.shape[0] - 1) // 2
            cst = np.sqrt((2 - lam) / lam)
            a = np.empty(p)
            for i in range(p):
                Z[0, i] = lam * x[i] + (1 - lam) * Z[0, i]
                a[i] = abs(cst * Z[0, i])
            a = -np.sort(-a)
            best = -1e18
            for k in range(1, q + 1):
                g = 0.0
                if k < p:
                    g = a[k] * a[k]
                num = 0.0
                den = 0.0
                for i in range(k):
                    mi = a[i] - g / a[i]
                    num += a[i] * mi
                    den += mi * mi
                v = num * num / den if den > 0 else 0.0
                st = (v - prm[k]) / prm[q + k]
                if st > best:
                    best = st
            if best > lim[0]:
                sig = True
        elif kind == 11:
            # Top-r CUSUM (Mei 2010; Xie & Siegmund 2013). Per-coordinate two-sided CUSUM
            # for a reference shift d0, global statistic = sum of the r largest local CUSUMs.
            # prm = [d0, r]; lim = [h]
            d0 = prm[0]
            r = int(prm[1])
            u = np.empty(p)
            for i in range(p):
                a = S[0, i] + d0 * x[i] - 0.5 * d0 * d0
                b = S[1, i] - d0 * x[i] - 0.5 * d0 * d0
                if a < 0.0:
                    a = 0.0
                if b < 0.0:
                    b = 0.0
                S[0, i] = a
                S[1, i] = b
                u[i] = a if a > b else b
            if r >= p:
                tot = 0.0
                for i in range(p):
                    tot += u[i]
                if tot > lim[0]:
                    sig = True
            else:
                u = -np.sort(-u)
                tot = 0.0
                for i in range(r):
                    tot += u[i]
                if tot > lim[0]:
                    sig = True
        elif kind == 12:
            # Group-structured balanced EWMA (G-BAS). Contiguous equal groups of size g.
            # Q_k = (2-l)/l * sum_{i in group k} z_i^2 ; terms: max_k Q_k and sum_k Q_k (= MEWMA)
            # prm = [lam, g]; lim = [h_grp, h_all]
            lam = prm[0]
            g = int(prm[1])
            cst = (2 - lam) / lam
            mxq = -1.0e18
            tot = 0.0
            q = 0.0
            for i in range(p):
                Z[0, i] = lam * x[i] + (1 - lam) * Z[0, i]
                q += cst * Z[0, i] * Z[0, i]
                if (i + 1) % g == 0:
                    tot += q
                    if q > mxq:
                        mxq = q
                    q = 0.0
            if mxq > lim[0] or tot > lim[1]:
                sig = True
        elif kind == 20:
            # Group mixture EWMA: sum_k log(1 - p0 + p0 exp(Q_k / 2)); prm = [lam, g, p0]; lim = [h]
            lam = prm[0]
            g = int(prm[1])
            p0 = prm[2]
            cst = (2 - lam) / lam
            stt = 0.0
            q = 0.0
            for i in range(p):
                Z[0, i] = lam * x[i] + (1 - lam) * Z[0, i]
                q += cst * Z[0, i] * Z[0, i]
                if (i + 1) % g == 0:
                    stt += np.log1p(p0 * np.expm1(min(q, 600.0) / 2))
                    q = 0.0
            if stt > lim[0]:
                sig = True
        elif kind == 13:
            # MSGL-EWMA (Kim, Turkoz, Jeong & Elsayed, 2024), Sigma = I, equal contiguous groups.
            # u = standardized EWMA. Sparse-group-lasso prox in closed form (orthonormal design):
            #   s_n = S(u_n, a)  (coordinatewise soft threshold),  mu_n = (1 - b/||s_n||)_+ s_n.
            # Tuning as in Kim et al. (2024): kappa2* is the smallest group penalty zeroing all groups
            # (b* = max_n ||u_n||, equal sizes); kappa1* the smallest l1 penalty that, with the
            # group penalty at c2*kappa2*, zeroes everything (a* solves max_n ||S(u_n,a)|| = c2 b*).
            # Applied: a = c1 a*, b = c2 b*.  Statistic Q = 2 u'mu - mu'mu.
            # prm = [r, g, c1, c2]; lim = [H]
            r = prm[0]
            g = int(prm[1])
            c1 = prm[2]
            c2 = prm[3]
            ng = p // g
            sc = np.sqrt((2 - r) / r)
            u = np.empty(p)
            umax = 0.0
            for i in range(p):
                Z[0, i] = r * x[i] + (1 - r) * Z[0, i]
                u[i] = sc * Z[0, i]
                if abs(u[i]) > umax:
                    umax = abs(u[i])
            M = 0.0
            for k in range(ng):
                nn = 0.0
                for i in range(k * g, (k + 1) * g):
                    nn += u[i] * u[i]
                nn = np.sqrt(nn)
                if nn > M:
                    M = nn
            target = c2 * M
            lo_a = 0.0
            hi_a = umax
            for _it in range(40):
                mid = 0.5 * (lo_a + hi_a)
                f = 0.0
                for k in range(ng):
                    nn = 0.0
                    for i in range(k * g, (k + 1) * g):
                        v = abs(u[i]) - mid
                        if v > 0.0:
                            nn += v * v
                    nn = np.sqrt(nn)
                    if nn > f:
                        f = nn
                if f > target:
                    lo_a = mid
                else:
                    hi_a = mid
            a = c1 * 0.5 * (lo_a + hi_a)
            b = c2 * M
            Qs = 0.0
            for k in range(ng):
                nn = 0.0
                for i in range(k * g, (k + 1) * g):
                    v = abs(u[i]) - a
                    if v > 0.0:
                        nn += v * v
                nn = np.sqrt(nn)
                if nn > b:
                    fac = 1.0 - b / nn
                    for i in range(k * g, (k + 1) * g):
                        v = abs(u[i]) - a
                        if v > 0.0:
                            m_i = fac * v
                            if u[i] < 0:
                                m_i = -m_i
                            Qs += 2.0 * u[i] * m_i - m_i * m_i
            if Qs > lim[0]:
                sig = True
        elif kind == 14:
            # MSGL-EWMA (Kim et al., 2024) for block-correlated groups. Data x are whitened block-wise
            # (x_w = R x, Sigma^-1 = R'R, R block diagonal with identical blocks R0). u = sc * EWMA(x_w).
            # Per group: min 0.5||u_n - A mu_n||^2 + a||mu_n||_1 + b||mu_n||_2, A = sc*R0.
            # kappa* from KKT at mu = 0 on G_n = A'u_n; solved by warm-started FISTA.
            # Q = sum_n 2 u_n'A mu_n - ||A mu_n||^2.  prm = [r, g, c1, c2, Lip(unscaled), R0 (g*g), R0'R0 (g*g)]
            r = prm[0]
            g = int(prm[1])
            c1 = prm[2]
            c2 = prm[3]
            Lip = prm[4] * (2 - r) / r
            ng = p // g
            sc = np.sqrt((2 - r) / r)
            sc2 = (2 - r) / r
            u = np.empty(p)
            G = np.empty(p)
            for i in range(p):
                Z[0, i] = r * x[i] + (1 - r) * Z[0, i]
                u[i] = sc * Z[0, i]
            gmax = 0.0
            M = 0.0
            for k in range(ng):
                nn = 0.0
                for i in range(g):
                    s_ = 0.0
                    for j in range(g):
                        s_ += sc * prm[5 + j * g + i] * u[k * g + j]
                    G[k * g + i] = s_
                    nn += s_ * s_
                    if abs(s_) > gmax:
                        gmax = abs(s_)
                nn = np.sqrt(nn)
                if nn > M:
                    M = nn
            target = c2 * M
            lo_a = 0.0
            hi_a = gmax
            for _it in range(40):
                mid = 0.5 * (lo_a + hi_a)
                f = 0.0
                for k in range(ng):
                    nn = 0.0
                    for i in range(k * g, (k + 1) * g):
                        v = abs(G[i]) - mid
                        if v > 0.0:
                            nn += v * v
                    nn = np.sqrt(nn)
                    if nn > f:
                        f = nn
                if f > target:
                    lo_a = mid
                else:
                    hi_a = mid
            a = c1 * 0.5 * (lo_a + hi_a)
            b = c2 * M
            mode = 0.0
            if len(prm) > 5 + 2 * g * g:
                mode = prm[5 + 2 * g * g]
            if mode > 1.5:
                # SLEP order (Liu, Ji & Ye 2009, sgLeastR with ratio inputs): l1 level first, as a
                # ratio of max|A'y|; group level as a ratio of the largest group norm after soft-thresholding.
                a = c1 * gmax
                M2 = 0.0
                for k in range(ng):
                    nn = 0.0
                    for i in range(k * g, (k + 1) * g):
                        v = abs(G[i]) - a
                        if v > 0.0:
                            nn += v * v
                    nn = np.sqrt(nn)
                    if nn > M2:
                        M2 = nn
                b = c2 * M2
            Qs = 0.0
            mu = np.empty(g)
            if mode > 3.5:
                # mode 4: sparse group LASSO on the unwhitened standardized EWMA (identity design),
                # SLEP-order thresholds; Sigma enters only through the statistic Q.
                R0m = np.empty((g, g))
                for i in range(g):
                    for j in range(g):
                        R0m[i, j] = prm[5 + i * g + j]
                R0i = np.linalg.inv(R0m)
                y = np.empty(p)
                ym = 0.0
                for k in range(ng):
                    for i in range(g):
                        s_ = 0.0
                        for j in range(g):
                            s_ += R0i[i, j] * u[k * g + j]
                        y[k * g + i] = s_ / sc
                        if abs(y[k * g + i]) > ym:
                            ym = abs(y[k * g + i])
                a4 = c1 * ym
                M4 = 0.0
                for k in range(ng):
                    nn = 0.0
                    for i in range(k * g, (k + 1) * g):
                        v = abs(y[i]) - a4
                        if v > 0.0:
                            nn += v * v
                    nn = np.sqrt(nn)
                    if nn > M4:
                        M4 = nn
                b4 = c2 * M4
                for k in range(ng):
                    nn = 0.0
                    for i in range(k * g, (k + 1) * g):
                        v = abs(y[i]) - a4
                        if v > 0.0:
                            nn += v * v
                    nn = np.sqrt(nn)
                    fac = 0.0
                    if nn > b4:
                        fac = 1.0 - b4 / nn
                    for i in range(g):
                        v = abs(y[k * g + i]) - a4
                        if v < 0.0:
                            v = 0.0
                        mu[i] = fac * (v if y[k * g + i] >= 0 else -v)
                    for i in range(g):
                        am = 0.0
                        for j in range(g):
                            am += sc * prm[5 + i * g + j] * mu[j]
                        Qs += 2.0 * u[k * g + i] * am - am * am
                if Qs > lim[0]:
                    sig = True
                if sig:
                    if t <= tau:
                        return -1
                    return t - tau
                continue
            yk = np.empty(g)
            grad = np.empty(g)
            for k in range(ng):
                nn = 0.0
                for i in range(k * g, (k + 1) * g):
                    v = abs(G[i]) - a
                    if v > 0.0:
                        nn += v * v
                if np.sqrt(nn) <= b:
                    for i in range(g):
                        Z[1, k * g + i] = 0.0
                    continue
                for i in range(g):
                    mu[i] = Z[1, k * g + i]
                    yk[i] = mu[i]
                tk = 1.0
                for _f in range(30):
                    for i in range(g):
                        s_ = 0.0
                        for j in range(g):
                            s_ += sc2 * prm[5 + g * g + i * g + j] * yk[j]
                        grad[i] = s_ - G[k * g + i]
                    nn = 0.0
                    newm = np.empty(g)
                    for i in range(g):
                        v = yk[i] - grad[i] / Lip
                        av = abs(v) - a / Lip
                        if av < 0.0:
                            av = 0.0
                        newm[i] = av if v >= 0 else -av
                        nn += newm[i] * newm[i]
                    nn = np.sqrt(nn)
                    fac = 0.0
                    if nn > b / Lip:
                        fac = 1.0 - (b / Lip) / nn
                    tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
                    for i in range(g):
                        nv = fac * newm[i]
                        yk[i] = nv + ((tk - 1.0) / tn) * (nv - mu[i])
                        mu[i] = nv
                    tk = tn
                for i in range(g):
                    Z[1, k * g + i] = mu[i]
                if mode > 0.5 and mode != 2.0:
                    # refit (mode 1 or 3): GLR on the selected support, Q_n = G_S' (sc2 * (R0'R0)_SS)^-1 G_S
                    ns = 0
                    for i in range(g):
                        if mu[i] != 0.0:
                            ns += 1
                    if ns > 0:
                        Ms = np.empty((ns, ns))
                        gs = np.empty(ns)
                        ii = 0
                        for i in range(g):
                            if mu[i] != 0.0:
                                jj = 0
                                for j in range(g):
                                    if mu[j] != 0.0:
                                        Ms[ii, jj] = sc2 * prm[5 + g * g + i * g + j]
                                        jj += 1
                                gs[ii] = G[k * g + i]
                                ii += 1
                        sol = np.linalg.solve(Ms, gs)
                        for i in range(ns):
                            Qs += gs[i] * sol[i]
                    continue
                for i in range(g):
                    am = 0.0
                    for j in range(g):
                        am += sc * prm[5 + i * g + j] * mu[j]
                    Qs += 2.0 * u[k * g + i] * am - am * am
            if Qs > lim[0]:
                sig = True
        if sig:
            if t <= tau:
                return -1  # false alarm before change: discarded (conditional steady state)
            return t - tau
    return maxT

@njit(parallel=True, cache=True)
def run_lengths(kind, prm, lim, p, delta, tau, maxT, R, seed0, dvec):
    out = np.empty(R, dtype=np.int64)
    for r in prange(R):
        out[r] = _one_run(kind, prm, lim, p, delta, tau, maxT, seed0 + r, dvec)
    return out

def arl(kind, prm, lim, p=2, delta=0.0, tau=0, maxT=20000, R=20000, seed0=1, nz=1, dvec=None):
    if dvec is None:
        dvec = np.zeros(p)
        dvec[:nz] = 1.0 / np.sqrt(nz)
    dvec = np.ascontiguousarray(np.asarray(dvec, float))
    rl = run_lengths(kind, np.asarray(prm, float), np.asarray(lim, float), p, delta, tau, maxT, R, seed0, dvec)
    rl = rl[rl > 0]
    return rl.mean(), rl.std() / np.sqrt(len(rl)), rl


@njit(parallel=True, cache=True)
def topsum_samples(lam, svals, p, R, burn, every, n_per, seed0):
    """In-control steady-state samples of the top-s sums (for balancing limits)."""
    m = svals.shape[0]
    out = np.empty((R * n_per, m))
    for r in prange(R):
        np.random.seed(seed0 + r)
        z = np.zeros(p)
        cst = (2 - lam) / lam
        k = 0
        for t in range(burn + every * n_per):
            for i in range(p):
                z[i] = lam * np.random.standard_normal() + (1 - lam) * z[i]
            if t >= burn and (t - burn) % every == 0 and k < n_per:
                u = -np.sort(-(cst * z * z))
                cs = 0.0
                j = 0
                for s_ in range(p):
                    cs += u[s_]
                    if j < m and s_ + 1 == int(svals[j]):
                        out[r * n_per + k, j] = cs
                        j += 1
                k += 1
    return out


@njit(parallel=True, cache=True)
def lewma_samples(lam, gs, p, R, burn, every, n_per, seed0):
    q = gs.shape[0]
    out = np.empty((R * n_per, q))
    for r in prange(R):
        np.random.seed(seed0 + r)
        z = np.zeros(p)
        cst = np.sqrt((2 - lam) / lam)
        k = 0
        for t in range(burn + every * n_per):
            for i in range(p):
                z[i] = lam * np.random.standard_normal() + (1 - lam) * z[i]
            if t >= burn and (t - burn) % every == 0 and k < n_per:
                for j in range(q):
                    num = 0.0
                    den = 0.0
                    for i in range(p):
                        ui = cst * z[i]
                        a2 = ui * ui
                        if a2 > gs[j]:
                            mi = (a2 - gs[j]) / abs(ui)
                            num += abs(ui) * mi
                            den += mi * mi
                    out[r * n_per + k, j] = num * num / den if den > 0 else 0.0
                k += 1
    return out


@njit(parallel=True, cache=True)
def lewma_path_samples(lam, q, p, R, burn, every, n_per, seed0):
    out = np.empty((R * n_per, q))
    for r in prange(R):
        np.random.seed(seed0 + r)
        z = np.zeros(p)
        cst = np.sqrt((2 - lam) / lam)
        kk = 0
        for t in range(burn + every * n_per):
            for i in range(p):
                z[i] = lam * np.random.standard_normal() + (1 - lam) * z[i]
            if t >= burn and (t - burn) % every == 0 and kk < n_per:
                a = -np.sort(-np.abs(cst * z))
                for k in range(1, q + 1):
                    g = a[k] * a[k] if k < p else 0.0
                    num = 0.0
                    den = 0.0
                    for i in range(k):
                        mi = a[i] - g / a[i]
                        num += a[i] * mi
                        den += mi * mi
                    out[r * n_per + kk, k - 1] = num * num / den if den > 0 else 0.0
                kk += 1
    return out


@njit(parallel=True, cache=True)
def maxm_sum_samples(lam, mm, p, R, burn, every, n_per, seed0):
    """In-control steady-state samples of (max over first mm coords, sum over all p)."""
    out = np.empty((R * n_per, 2))
    for r in prange(R):
        np.random.seed(seed0 + r)
        z = np.zeros(p)
        cst = (2 - lam) / lam
        k = 0
        for t in range(burn + every * n_per):
            mx = -1.0e18
            tot = 0.0
            for i in range(p):
                z[i] = lam * np.random.standard_normal() + (1 - lam) * z[i]
                ui = cst * z[i] * z[i]
                tot += ui
                if i < mm and ui > mx:
                    mx = ui
            if t >= burn and (t - burn) % every == 0 and k < n_per:
                out[r * n_per + k, 0] = mx
                out[r * n_per + k, 1] = tot
                k += 1
    return out
