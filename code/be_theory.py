"""Fixed-sample (one-look) power approximation of the break-even, treating each look as a test at
a fixed level.
Inputs: none (simulation).
Outputs: be_theory.json
Paper: Section 4.2; Supplement S13.
"""
import numpy as np, json
from scipy.stats import chi2, ncx2
from scipy.optimize import brentq
def powers(p, k, eta, alpha, w1):
    hM = chi2.isf(alpha, p); PM = ncx2.sf(hM, p, eta)
    a1 = w1 * alpha; h1 = chi2.isf(1 - (1 - a1) ** (1 / p), 1); h2 = chi2.isf((1 - w1) * alpha, p)
    nomax = chi2.cdf(h1, 1) ** (p - k) * ncx2.cdf(h1, 1, eta / k) ** k
    Pmax = 1 - nomax; Psum = ncx2.sf(h2, p, eta)
    return PM, 1 - (1 - Pmax) * (1 - Psum)   # independence approximation between the two terms
def breakeven(p, alpha, w1, pi):
    eta = brentq(lambda e: ncx2.sf(chi2.isf(alpha, p), p, e) - pi, 1e-6, 1e5)
    ks = np.arange(1, p + 1); d = np.array([np.subtract(*powers(p, k, eta, alpha, w1))[()] for k in ks])  # PM - PB
    i = np.where(d > 0)[0]
    if len(i) == 0: return float(p)
    i = i[0];
    if i == 0: return 1.0
    return float(np.exp(np.log(ks[i-1]) + (0 - d[i-1]) * (np.log(ks[i]) - np.log(ks[i-1])) / (d[i] - d[i-1])))
P = [10, 30, 50, 100, 200, 500, 1000]; obs = {10: 2.3, 30: 3.9, 50: None, 100: 6.1, 1000: 11.0}
out = {}
for alpha in (0.005, 0.02): 
    for pi in (0.1, 0.3, 0.5):
        r = [breakeven(p, alpha, 0.25, pi) for p in P]; out[f'a={alpha} pi={pi}'] = r
        lp = np.log(P); b = np.polyfit(lp, r, 1); res = np.array(r) - np.polyval(b, lp)
        bs = np.polyfit(np.sqrt(P), r, 1); rs = np.array(r) - np.polyval(bs, np.sqrt(P))
        print(alpha, pi, [round(x, 2) for x in r], 'ln-fit slope %.2f rms %.2f | sqrt-fit rms %.2f' % (b[0], np.sqrt((res**2).mean()), np.sqrt((rs**2).mean())))
json.dump(out, open('be_theory.json', 'w'), indent=1)
