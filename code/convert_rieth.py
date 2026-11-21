"""Converts the Rieth et al. (2017) Tennessee Eastman .RData files to .npy arrays.
Usage: python convert_rieth.py <folder with the .RData files>
Inputs: none (simulation).
Outputs: X.npy, fault.npy, {stem}.npy, {stem}_run.npy
Paper: Section 5.1 (data).
"""
import os, sys
import numpy as np
import pyreadr

SRC = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'rieth2017')
os.makedirs(os.path.join(OUT, 'ftest'), exist_ok=True)
COLS = [f'xmeas_{i}' for i in range(1, 42)] + [f'xmv_{i}' for i in range(1, 12)]


def load(name):
    df = next(iter(pyreadr.read_r(os.path.join(SRC, name)).values()))
    assert list(df.columns[3:]) == COLS, 'unexpected columns'
    return df


for name, stem in (('TEP_FaultFree_Training.RData', 'ff_train'),
                   ('TEP_FaultFree_Testing.RData', 'ff_test')):
    df = load(name)
    np.save(os.path.join(OUT, f'{stem}.npy'), df[COLS].to_numpy(np.float32))
    np.save(os.path.join(OUT, f'{stem}_run.npy'), df['simulationRun'].to_numpy(np.int32))
    print(stem, df.shape)

df = load('TEP_Faulty_Testing.RData')
np.save(os.path.join(OUT, 'ftest', 'X.npy'), df[COLS].to_numpy(np.float32))
np.save(os.path.join(OUT, 'ftest', 'fault.npy'), df['faultNumber'].to_numpy(np.int16))
print('faulty testing', df.shape)
