"""Draws Figs. 1-5 of the paper from the result files.
Inputs: authors_bench.json, be_wls.json, group_bench.json, pr_general.json, robust2.json, te_equal_holm80.json, te_equal_raw.json
Outputs: ../figures/fig1-fig5 (PDF)
"""
import json, numpy as np, matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 9, 'legend.frameon': False, 'pdf.fonttype': 42})
OUT = '../figures'
import os; os.makedirs(OUT, exist_ok=True)
ACC, GREY, DARK = '#1f5fbf', '#9a9a9a', '#222222'
W = 6.3

# Figure 1: break-even vs dimension
f, ax = plt.subplots(figsize=(W, 2.9))
d05 = [10, 30, 50, 100, 200, 500, 1000]; b05 = [2.32, 3.88, 4.66, 6.13, 8.52, 9.10, 11.04]
d20 = [10, 100, 1000]; b20 = [2.39, 4.89, 5.75]
dd = np.array([10, 30, 50, 100, 200, 500, 1000])
ax.plot(dd, 0.7 * np.sqrt(dd), ls='--', color=GREY, label=r'$0.7\sqrt{p}$ (rejected)')
W_ = json.load(open('be_wls.json'))['points']; se05 = [W_[str(d)]['se'] if str(d) in W_ else 0 for d in d05]
ax.errorbar(d05, b05, yerr=[1.96 * e for e in se05], fmt='o-', color=ACC, capsize=2, label=r'$\lambda=0.05$ (slope 1.93; bars: 95% MC intervals)')
ax.annotate('p = 200', (200, b05[4]), textcoords='offset points', xytext=(-12, 8), fontsize=8.5, color=ACC)
ax.plot(d20, b20, 'o-', color=ACC, alpha=0.45, label=r'$\lambda=0.20$ (slope 0.73)')
ax.set_xscale('log'); ax.set_xlabel('number of monitored variables $p$ (log scale)'); ax.set_ylabel('break-even participation ratio')
ax.set_ylim(0, 23); ax.legend(loc='upper left')
f.tight_layout(); f.savefig(f'{OUT}/fig1_breakeven.pdf'); plt.close(f)

# Figure 2: gain vs sparsity, no dominance
r = json.load(open('robust2.json')); NZ = [1, 2, 3, 5, 10, 30]
m = [r['MEWMA lam=0.05'][f'0.75_{n}'] for n in NZ]
sel = [('BAS-EWMA lam=0.05 w1=0.4', r'BAS-EWMA $w_1=0.40$', ACC, 1.0, 2.2),
       ('BAS-EWMA lam=0.05 w1=0.65', r'BAS-EWMA $w_1=0.65$', ACC, 0.45, 1.4),
       ('LEWMA lam=0.05', 'LEWMA', DARK, 0.75, 1.1), ('VS-MEWMA lam=0.05', 'VS-MEWMA', DARK, 0.5, 1.1),
       ('top-r CUSUM r=1', 'top-1 CUSUM', GREY, 1.0, 1.1), ('top-r CUSUM r=3', 'top-3 CUSUM', GREY, 0.7, 1.1),
       ('top-r CUSUM r=5', 'top-5 CUSUM', GREY, 0.45, 1.1)]
f, ax = plt.subplots(figsize=(W, 3.3))
ax.axhline(0, color=DARK, lw=0.8); ax.text(1.05, 1.5, 'MEWMA', fontsize=8)
for k, lab, c, a, lw in sel:
    g = [(1 - r[k][f'0.75_{n}'] / mm) * 100 for n, mm in zip(NZ, m)]
    ax.plot(NZ, g, 'o-', color=c, alpha=a, lw=lw, ms=3, label=lab)
PG = json.load(open('pr_general.json')); a0 = {k: v[0] for k, v in PG['ss_arl0'].items()}
for ck, lab, ls in (('mix0.3', r'mixture $p_0=0.3$ (CRN)', '-'), ('soft2', r'soft-threshold $c=2$ (CRN)', '--')):
    ks = [s_['pr'] for s_ in PG['structured']]; gg = [100 * (1 - s_['arl'][ck][0] * (a0['MEWMA'] / a0[ck]) ** 0.37 / s_['arl']['MEWMA'][0]) for s_ in PG['structured']]
    ax.plot(ks, gg, ls, marker='s', color='#c0572b', alpha=1.0 if ck == 'mix0.3' else 0.6, lw=1.6, ms=3, label=lab)
ax.set_xscale('log'); ax.set_xticks(NZ); ax.set_xticklabels([str(n) for n in NZ])
ax.set_xlabel('coordinates carrying the shift (of 30, log scale)'); ax.set_ylabel('% faster than MEWMA')
ax.legend(ncol=2, fontsize=8.5, loc='lower left')
f.tight_layout(); f.savefig(f'{OUT}/fig2_no_dominance.pdf'); plt.close(f)

# Figure 3: G-BAS exchange rate
g = json.load(open('group_bench.json'))
S = [k.split('|')[0] for k in g['MEWMA'] if k.endswith('|0.75')]
inG = [s for s in S if s.split()[0] in ('A1', 'A2', 'A3', 'A5')]
def pt(n):
    gg = lambda s: (1 - g[n][f'{s}|0.75'] / g['MEWMA'][f'{s}|0.75']) * 100
    return -min(gg(s) for s in S), np.mean([gg(s) for s in inG])
f, ax = plt.subplots(figsize=(W, 3.0))
gb = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
P = [pt(f'G-BAS w1={w}') for w in gb]
ax.errorbar([0] + [p[0] for p in P], [0] + [p[1] for p in P], xerr=0.88, yerr=0.88, fmt='o-', color=ACC, capsize=2, elinewidth=0.8, label='G-BAS (labels: $w_1$; bars: approx. 95% intervals)')
for w, (x, y) in zip(gb, P):
    ax.annotate(f'{w}', (x, y), textcoords='offset points', xytext=(6, -10) if w < 0.3 else (-6, 4), ha='left' if w < 0.3 else 'right', fontsize=8.5, color=ACC)
for n, lab in (('BAS-EWMA w1=0.40', 'BAS-EWMA'), ('LEWMA', 'LEWMA'), ('top-r CUSUM r=5', 'top-5 CUSUM')):
    x, y = pt(n); ax.plot(x, y, 'o', color=GREY); ax.annotate(lab, (x, y), textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8.5)
ax.plot(0, 0, 'o', color=DARK); ax.annotate('MEWMA', (0, 0), textcoords='offset points', xytext=(10, 4), fontsize=8.5)
ax.set_xlabel('worst-case loss to MEWMA over all ten shift patterns (%)'); ax.set_ylabel('% faster than MEWMA, in-group shifts')
ax.set_xlim(-0.5, 22); ax.set_ylim(-0.5, 16); ax.legend(loc='lower right')
f.tight_layout(); f.savefig(f'{OUT}/fig3_gbas_exchange.pdf'); plt.close(f)

# Figure 4: authors' setting p = 25
A = json.load(open('authors_bench.json'))['p25']; mw = 'MEWMA lam=0.1'
cases = [c.split('|')[0] for c in A[mw] if c.endswith('|0.5')]
series = [('G-BAS lam=0.1 w1=0.5', 'G-BAS', ACC, 1.0), ('MSGLs lam=0.1 c1=0.8 c2=0.8', 'MSGL-EWMA, heavy penalty', DARK, 1.0),
          ('MSGLs lam=0.1 c1=0.2 c2=0.2', 'MSGL-EWMA, light penalty', DARK, 0.35)]
f, ax = plt.subplots(figsize=(W, 3.2)); y = np.arange(len(cases))[::-1]
for k, lab, c, a in series:
    ratio = [A[k][f'{cs}|0.5'] / A[mw][f'{cs}|0.5'] for cs in cases]
    ax.plot(ratio, y, 'o', color=c, alpha=a, label=lab, ms=5)
ax.axvline(1, color=DARK, lw=0.8); ax.set_xscale('log'); ax.set_xticks([0.8, 1, 2, 5, 10]); ax.set_xticklabels(['0.8', '1', '2', '5', '10'])
ax.set_yticks(y); ax.set_yticklabels(cases); ax.set_xlabel(r'detection delay relative to MEWMA (log scale)')
ax.legend(loc='upper right', fontsize=8.5)
f.tight_layout(); f.savefig(f'{OUT}/fig4_authors_setting.pdf'); plt.close(f)

# Figure 5: TEP steady state on observations, equal in-control ARL, all runs (Holm over 80 log-delay tests)
E = json.load(open('te_equal_raw.json')); H = json.load(open('te_equal_holm80.json')); K = list(range(1, 21))
f, ax = plt.subplots(figsize=(W, 3.9)); y = np.arange(20)[::-1]
for n, c, off in (('G-BAS', ACC, -0.15), ('BAS-EWMA', DARK, 0.15)):
    rr = np.array([E['rows'][f'{k}|{n}']['ratio_all'] for k in K]); g = np.array([E['rows'][f'{k}|{n}']['geo_all'] for k in K]); t = np.array([E['rows'][f'{k}|{n}']['t_log_all'] for k in K])
    se = np.where(np.abs(t) > 1e-9, np.abs(np.log(g)) / np.maximum(np.abs(t), 1e-9), 0.0); lo, hi = rr * np.exp(-1.96 * se), rr * np.exp(1.96 * se)
    sig = np.array(H[f'raw|{n}']['holm_sig']); yy = y + off
    ax.hlines(yy, lo, hi, color=c, lw=0.8, alpha=0.7)
    ax.scatter(rr[sig], yy[sig], color=c, s=22, label=f'{n} (Holm significant)', zorder=3)
    ax.scatter(rr[~sig], yy[~sig], facecolors='none', edgecolors=c, s=22, label=f'{n} (not significant)', zorder=3)
ax.axvline(1, color=DARK, lw=0.8); ax.set_yticks(y); ax.set_yticklabels([f'fault {k}' for k in K], fontsize=8.5)
ax.set_xlabel('average detection delay relative to MEWMA'); ax.legend(fontsize=7, loc='lower right')
f.tight_layout(); f.savefig(f'{OUT}/fig5_tep_steady.pdf'); plt.close(f)
print('ok')
