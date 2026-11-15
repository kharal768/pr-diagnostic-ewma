"""Builds the replication table of Supplement S6 from the three readings and the published values.
Inputs: repro_kim2.json, repro_kim3.json, repro_kim4.json
Outputs: repro_stats.json
Paper: Supplement S6.
"""
import json
P={('p10','MEWMA','C1'):(67.25,17.92,5.60),
 ('p10','0.5,0.5','C1'):(71.90,18.60,5.64),('p10','0.7,0.3','C1'):(70.06,18.18,5.55),('p10','0.3,0.8','C1'):(72.51,18.93,5.69),
 ('p10','0.5,0.5','C2'):(67.70,16.96,5.19),('p10','0.7,0.3','C2'):(70.02,17.59,5.36),('p10','0.3,0.8','C2'):(65.66,15.94,4.91),
 ('p25','MEWMA','s1'):(93.72,26.63,7.36)}
T4={'0.0,0.8':[(95.43,27.17,6.52),(97.92,28.11,6.58),(99.53,28.97,6.83)],
    '0.2,0.8':[(94.20,24.63,6.14),(95.07,24.21,6.11),(98.22,27.23,6.65)],
    '0.4,0.8':[(88.36,21.58,5.65),(89.97,22.41,5.84),(93.63,25.00,6.33)],
    '0.6,0.8':[(84.12,19.14,5.28),(92.83,22.69,5.89),(95.98,25.14,6.40)],
    '0.8,0.8':[(79.26,17.51,4.93),(98.22,24.70,6.32),(101.00,27.12,6.86)]}
for d,v in T4.items():
    for i,c in enumerate(('s1','s2','s3')): P[('p25',d,c)]=v[i]
L=json.load(open('repro_kim4.json')); A=json.load(open('repro_kim2.json')); R=json.load(open('repro_kim3.json'))
lab={'C1':'two groups','C2':'same group','s1':'1 variable','s2':'2 variables','s3':'3 variables'}
f=lambda v:'---' if v is None else (f'{v:.2f}' if v<10 else f'{v:.1f}')
def get(D,pp,des,case,dl):
    r=D.get(f'{pp} {des}'); return r[f'{case}|{dl}'] if r else None
rows=[];last=None;stat={}
for (pp,des,case),pv in P.items():
    if last and last!=pp: rows.append(r'\midrule')
    last=pp; cells=[]
    for i,dl in enumerate((0.5,1.0,2.0)):
        if des=='MEWMA':
            m=A[f'{pp} MEWMA'][f'{case}|{dl}']; cells+=[f(pv[i]),f(m),'','']; continue
        vals=[get(D,pp,des,case,dl) for D in (L,A,R)]
        for n,v in zip('LAR',vals):
            if v is not None:
                s=stat.setdefault((pp,n),[0,0,0]); e=abs(v/pv[i]-1); s[0]+=e<=0.05; s[1]+=1; s[2]=max(s[2],e)
        if dl==2.0: continue
        cells+=[f(pv[i])]+[f(v) for v in vals]
    if des=='MEWMA': cells=cells[:8]
    d='MEWMA' if des=='MEWMA' else f'({des.replace(",",", ")})'
    rows.append(f"${pp[0]}={pp[1:]}$ & {d} & {'any' if des=='MEWMA' else lab[case]} & "+' & '.join(cells)+r' \\')
for k,v in sorted(stat.items()): print(k, f'{v[0]} of {v[1]} within 5%, worst {100*v[2]:.0f}%')
json.dump({f'{k[0]} {k[1]}':v for k,v in stat.items()},open('repro_stats.json','w'))
open('../paper/appendix_table.tex','w').write('\n'.join(rows))
