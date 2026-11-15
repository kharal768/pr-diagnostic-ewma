"""Replication check of MSGL-EWMA reading A against Tables 2-4 of Kim et al. (2024): delta as
noncentrality, zero state, smoothing 0.2, in-control ARL 200.
Inputs: none (simulation).
Outputs: repro_kim2.json
Paper: Supplement S6.
"""
import numpy as np, json, sys
from engine import arl
from calib import calibrate
RHO, r = 0.75, 0.2
def setup(g, p):
    S=(1-RHO)*np.eye(g)+RHO*np.ones((g,g)); R0=np.linalg.inv(np.linalg.cholesky(S)); return R0, np.kron(np.eye(p//g),R0)
def msprm(R0,g,c1,c2): return [r,g,c1,c2,float(np.linalg.eigvalsh(R0.T@R0).max())]+R0.ravel().tolist()+(R0.T@R0).ravel().tolist()
out={}
jobs={'p10':(2,10,[(0.5,0.5),(0.7,0.3),(0.3,0.8)],{'C1':[0,2],'C2':[0,1]}),
      'p25':(5,25,[(0.0,0.8),(0.2,0.8),(0.4,0.8),(0.6,0.8),(0.8,0.8)],{'s1':[0],'s2':[0,1],'s3':[0,1,2]})}
for key,(g,p,cs,cases) in jobs.items():
    R0,R=setup(g,p)
    des={'MEWMA':(0,[r],2.,80.)}
    for c1,c2 in cs: des[f'{c1},{c2}']=(14,msprm(R0,g,c1,c2),0.01,60.)
    for name,(kind,prm,lo,hi) in des.items():
        h,_=calibrate(kind,prm,lambda h:[h],lo,hi,p=p,R=8000,seed0=7,iters=22,maxT=3000)
        row={'arl0':float(arl(kind,prm,[h],p=p,R=10000,seed0=991,maxT=3000)[0])}
        for cn,idx in cases.items():
            d=np.zeros(p); d[idx]=1.; w=R@d; w/=np.linalg.norm(w)
            for dl in (0.5,1.0,2.0):
                row[f'{cn}|{dl}']=float(arl(kind,prm,[h],p=p,delta=dl,tau=0,maxT=3000,R=10000,seed0=4321,dvec=w)[0])
        out[f'{key} {name}']=row; print(key,name,{k:round(v,2) for k,v in row.items()},flush=True)
json.dump(out,open('repro_kim2.json','w'),indent=1)
