"""Prepares the WaDi data: reads the normal-operation and attack files, averages to 10-second
samples and keeps the continuous signals.
Usage: python wadi_prep.py <zip containing WADI_14days_new.csv> <WADI_attackdataLABLE.csv>
Inputs: data/wadi/
Outputs: data/wadi/normal10.parquet, data/wadi/attack10.parquet
Paper: Section 5.3.
"""
import sys, zipfile, numpy as np, pandas as pd
STEP = 10
def shrink(fh, out, skip):
    parts = []
    for ch in pd.read_csv(fh, chunksize=200000, low_memory=False, skiprows=skip):
        ch.columns = [str(c).strip() for c in ch.columns]
        lab = None
        lc = [c for c in ch.columns if 'Attack' in c or 'LABLE' in c.upper() or 'Label' in c]
        st = ch[['Date', 'Time']].iloc[::STEP].reset_index(drop=True)
        num = ch.drop(columns=[c for c in ('Row', 'Date', 'Time') if c in ch.columns]).apply(pd.to_numeric, errors='coerce').astype('float32')
        g = num.groupby(np.arange(len(num)) // STEP)
        m = g.mean().reset_index(drop=True)
        if lc: m[lc[0]] = g[lc[0]].min().values  # -1 attack if any second in block is attack
        parts.append(pd.concat([st.iloc[:len(m)], m], axis=1))
    df = pd.concat(parts, ignore_index=True); df.to_parquet(out); print(out, df.shape, lc)
z = zipfile.ZipFile(sys.argv[1]); shrink(z.open('WADI_14days_new.csv'), '../data/wadi/normal10.parquet', None)
shrink(open(sys.argv[2]), '../data/wadi/attack10.parquet', [0])
