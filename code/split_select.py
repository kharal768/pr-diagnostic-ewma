"""Design selection on a separate tuning grid against selection on the evaluation grid.
Inputs: final.json
Outputs: split_select.json
Paper: Section 4.1 (hold-out selection); Supplement S3 and S10.
"""
import json, numpy as np
F = json.load(open('final.json'))['p30']; D = F['deltas']; NZ = F['nz']
fam = {}
def arr(v):
    ss = v['ss']; return np.array([[ss[str(n)][j] for j in range(len(D))] for n in NZ]) if isinstance(ss, dict) else np.tile(ss, (len(NZ), 1))
for k, v in F['bench'].items(): fam.setdefault(k.split()[0], {})[k] = arr(v)
for k, v in F['bas'].items(): fam.setdefault('BAS-EWMA', {})['BAS ' + k] = arr(v)
for k, v in F['lewma'].items(): fam.setdefault('LEWMA', {})['LEWMA ' + k] = arr(v)
allA = np.stack([a for f in fam.values() for a in f.values()]); best = allA.min(0)
tune = [D.index(0.5), D.index(1.0)]; ev = [i for i in range(len(D)) if i not in tune]
ex = lambda a, cols: (a[:, cols] - best[:, cols]) / best[:, cols]
out = {'arl0_achieved': {}}
for f, ds in fam.items():
    pick_t = min(ds, key=lambda k: ex(ds[k], tune).mean()); pick_o = min(ds, key=lambda k: ex(ds[k], ev).mean())
    out[f] = {'n_designs': len(ds), 'tuned_pick': pick_t, 'oracle_pick': pick_o,
              'RMI_eval_tuned': float(100 * ex(ds[pick_t], ev).mean()), 'worst_eval_tuned': float(100 * ex(ds[pick_t], ev).max()),
              'RMI_eval_oracle': float(100 * ex(ds[pick_o], ev).mean()), 'worst_eval_oracle': float(100 * ex(ds[pick_o], ev).max())}
for grp in ('bench', 'bas', 'lewma'):
    for k, v in F[grp].items(): out['arl0_achieved'][k] = v['arl0']
print(json.dumps({k: v for k, v in out.items() if k != 'arl0_achieved'}, indent=1))
a = np.array(list(out['arl0_achieved'].values())); print('achieved ARL0 range', a.min(), a.max())
json.dump(out, open('split_select.json', 'w'), indent=1)
