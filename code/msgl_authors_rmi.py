"""Relative mean index (RMI) and worst case of each chart family's best design in the setting of
Kim et al. (2024).
Inputs: authors_bench.json
Outputs: printed to the console.
Paper: Section 4.4 (RMI and worst case).
"""
import json,sys,numpy as np
d=json.load(open('authors_bench.json'))
def fam(n): return n.split(' lam')[0]
def analyse(tag, exclude=()):
    D={n:r for n,r in d[tag].items() if fam(n) not in exclude}
    cells=sorted(k for k in next(iter(D.values())) if '|' in k)
    cells=[c for c in cells if all(c in r for r in D.values())]
    best={c:min(r[c] for r in D.values()) for c in cells}
    out={}
    for n,r in D.items():
        ex=np.array([r[c]/best[c]-1 for c in cells])
        f=fam(n); m,w=ex.mean(),ex.max()
        if f not in out or m<out[f][1]: out[f]=(n,m,w)
    return out,len(cells)
for tag in ('p25','p10'):
    for excl in (('MSGLr',),()):
        o,nc=analyse(tag,excl)
        print(tag,'pool without refit' if excl else 'pool with refit',nc,'cells')
        for f,(n,m,w) in sorted(o.items(),key=lambda x:x[1][1]): print(f'   {f:9s} {n:32s} mean {100*m:5.1f}%  worst {100*w:5.1f}%')
