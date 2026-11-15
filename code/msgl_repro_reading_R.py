"""Replication check of MSGL-EWMA reading R against Tables 2-4 of Kim et al. (2024).
Inputs: none (simulation).
Outputs: repro_kim3.json
Paper: Supplement S6.
"""
import numpy as np, json
from engine import arl
from calib import calibrate
RHO,r=0.75,0.2
def setup(g,p):
    S=(1-RHO)*np.eye(g)+RHO*np.ones((g,g)); R0=np.linalg.inv(np.linalg.cholesky(S)); return R0,np.kron(np.eye(p//g),R0)
out={}
jobs={'p10':(2,10,[(0.5,0.5),(0.7,0.3),(0.3,0.8)],{'C1':[0,2],'C2':[0,1]}),
      'p25':(5,25,[(0.0,0.8),(0.4,0.8),(0.8,0.8)],{'s1':[0],'s2':[0,1],'s3':[0,1,2]})}
for key,(g,p,cs,cases) in jobs.items():
    R0,R=setup(g,p)
    for c1,c2 in cs:
        prm=[r,g,c1,c2,float(np.linalg.eigvalsh(R0.T@R0).max())]+R0.ravel().tolist()+(R0.T@R0).ravel().tolist()+[1.0]
        h,_=calibrate(14,prm,lambda h:[h],0.01,80.,p=p,R=6000,seed0=7,iters=22,maxT=3000)
        row={'H':float(h),'arl0':float(arl(14,prm,[h],p=p,R=10000,seed0=991,maxT=3000)[0])}
        for cn,idx in cases.items():
            d=np.zeros(p); d[idx]=1.; w=R@d; w/=np.linalg.norm(w)
            for dl in (0.5,1.0,2.0):
                row[f'{cn}|{dl}']=float(arl(14,prm,[h],p=p,delta=dl,tau=0,maxT=3000,R=10000,seed0=4321,dvec=w)[0])
        out[f'{key} {c1},{c2}']=row; print(key,c1,c2,{k:round(v,2) for k,v in row.items()},flush=True)
        json.dump(out,open('repro_kim3.json','w'),indent=1)
