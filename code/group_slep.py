"""MSGL-EWMA (reading L, SLEP penalty order) on the group benchmark (p = 30, six groups of five).
Inputs: none (simulation).
Outputs: group_bench.json
Paper: Section 4.4.
"""
import numpy as np, json
from engine import arl
from calib import calibrate
from group_bench import SCEN, P, TAU, MAXT, DELTAS
OUT='group_bench.json'; res=json.load(open(OUT))
g=5; I=np.eye(g)
for c1 in (0.2,0.5,0.8):
    for c2 in (0.2,0.5,0.8):
        name=f'MSGLs r=0.05 c1={c1} c2={c2}'
        if name in res: continue
        prm=[0.05,g,c1,c2,1.0]+I.ravel().tolist()+I.ravel().tolist()+[2.0]
        h,_=calibrate(14,prm,lambda h:[h],0.01,80.,p=P,R=6000,seed0=7,iters=20,maxT=MAXT)
        row={'lim':[float(h)],'arl0':float(arl(14,prm,[h],p=P,R=10000,seed0=991,maxT=MAXT)[0])}
        if abs(row['arl0']-200)>15: print(name,'CAL OFF',row['arl0'],flush=True); continue
        for s,v in SCEN.items():
            for dl in DELTAS:
                row[f'{s}|{dl}']=float(arl(14,prm,[h],p=P,delta=dl,tau=TAU,maxT=MAXT,R=10000,seed0=4321,dvec=v)[0])
        res[name]=row; json.dump(res,open(OUT,'w'),indent=1)
        print(name,round(row['arl0'],1),' '.join(f"{row[f'{s}|0.75']:.1f}" for s in SCEN),flush=True)
print('DONE',flush=True)
