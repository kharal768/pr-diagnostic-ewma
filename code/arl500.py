"""Recalibrates key designs to an in-control ARL of 500 and rechecks rankings and the break-even.
p=30, delta=0.75.
Inputs: none (simulation).
Outputs: arl500.json
Paper: Sections 4.2 and 4.3 (in-control ARL 500).
"""
import numpy as np, json, os
from engine import arl
from calib import calibrate
from bas_ewma import design
from gbas import limfun_for
from robust2 import F, P, TAU
from group_bench import SCEN
MAXT=8000; OUT='arl500.json'; res=json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(res,open(OUT,'w'),indent=1)
specs={'MEWMA':(0,[0.05],lambda h:[h],20.,150.),
       'BAS-EWMA w1=0.4':(6,[0.05,1,P],design(0.05,[1,P],P,w=[0.4,0.6]),0.5,6.),
       'top-r CUSUM r=1':(11,[0.5,1],lambda h:[h],0.5*F['top-r CUSUM r=1']['lim'][0],4*F['top-r CUSUM r=1']['lim'][0]),
       'top-r CUSUM r=5':(11,[0.5,5],lambda h:[h],0.5*F['top-r CUSUM r=5']['lim'][0],4*F['top-r CUSUM r=5']['lim'][0]),
       'G-BAS w1=0.1':(12,[0.05,5],limfun_for(0.05,5,P,0.1),0.5,6.),
       'G-BAS w1=0.3':(12,[0.05,5],limfun_for(0.05,5,P,0.3),0.5,6.)}
for name,(kind,prm,lf,lo,hi) in specs.items():
    if name in res: continue
    c,_=calibrate(kind,prm,lf,lo,hi,p=P,R=6000,seed0=7,iters=22,target=500.,maxT=MAXT); lim=[float(x) for x in np.atleast_1d(lf(c))]
    row={'lim':lim,'arl0':float(arl(kind,prm,lim,p=P,R=8000,seed0=991,maxT=MAXT)[0])}
    pats={f'nz{n}':n for n in (1,2,3,5,10,30)} if not name.startswith('G-BAS') else {}
    for tag,n in pats.items():
        v=np.zeros(P); v[:n]=1; v/=np.linalg.norm(v)
        row[tag]=float(arl(kind,prm,lim,p=P,delta=0.75,tau=TAU,maxT=MAXT,R=10000,seed0=4321,dvec=v)[0])
    if name=='MEWMA' or name.startswith('G-BAS'):
        for s,v in SCEN.items():
            row['grp '+s.strip()]=float(arl(kind,prm,lim,p=P,delta=0.75,tau=TAU,maxT=MAXT,R=10000,seed0=4321,dvec=v)[0])
    if name in ('MEWMA','BAS-EWMA w1=0.4'):
        for k in (2,3,4,5,6,8):
            v=np.zeros(P); v[:k]=1; v/=np.linalg.norm(v)
            row[f'pr{k}']=float(arl(kind,prm,lim,p=P,delta=0.75,tau=TAU,maxT=MAXT,R=10000,seed0=4321,dvec=v)[0])
    res[name]=row; save(); print(name,{a:(round(b,1) if not isinstance(b,list) else b) for a,b in row.items()},flush=True)
print('DONE',flush=True)
