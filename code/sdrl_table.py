"""Run-length distribution (ARL, SDRL, median, 90th percentile) for the designs of Figs. 2 and 3
at in-control ARL 200, delta = 0.75.
Inputs: group_bench.json
Outputs: sdrl_table.json
Paper: Supplement S7.
"""
import numpy as np, json, os
from engine import arl
from robust2 import prm_of, SEL, F, P, TAU, MAXT
from group_bench import SCEN
OUT='sdrl_table.json'; res=json.load(open(OUT)) if os.path.exists(OUT) else {}
def stats(rl):
    rl=rl[rl>0]; return {'ARL':float(rl.mean()),'SDRL':float(rl.std()),'MRL':float(np.median(rl)),'q90':float(np.quantile(rl,0.9))}
for key in SEL:
    kind,prm=prm_of(key); lim=F[key]['lim']
    for nz in (1,2,3,5,10,30):
        k=f'sparsity|{key}|{nz}'
        if k in res: continue
        v=np.zeros(P); v[:nz]=1; v/=np.linalg.norm(v)
        res[k]=stats(arl(kind,prm,lim,p=P,delta=0.75,tau=TAU,maxT=MAXT,R=10000,seed0=4321,dvec=v)[2])
        json.dump(res,open(OUT,'w'),indent=1); print(k,{a:round(b,1) for a,b in res[k].items()},flush=True)
G=json.load(open('group_bench.json'))
for key,kind,prm in [('MEWMA',0,[0.05]),('G-BAS w1=0.1',12,[0.05,5]),('G-BAS w1=0.3',12,[0.05,5])]:
    lim=G[key]['lim']
    for s,v in SCEN.items():
        k=f'group|{key}|{s.strip()}'
        if k in res: continue
        res[k]=stats(arl(kind,prm,lim,p=P,delta=0.75,tau=TAU,maxT=MAXT,R=10000,seed0=4321,dvec=v)[2])
        json.dump(res,open(OUT,'w'),indent=1); print(k,{a:round(b,1) for a,b in res[k].items()},flush=True)
print('DONE',flush=True)
