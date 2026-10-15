"""Recompute saved test scores on the server without retraining."""
import json
import socket
import statistics
from pathlib import Path
import numpy as np
import torch
from hyperbolic_hierarchy_probe import distances, hashes

torch.set_default_dtype(torch.float64)
p = Path('outputs/hyperbolic-hierarchy-tangent-20261015')
r = json.loads((p / 'results.json').read_text())
assert r['completed'] and r['protected_source_unchanged']
a, b = np.array(r['pair_indices'])
n = len(r['parents'])
target = np.full((n,n), np.inf)
np.fill_diagonal(target,0)
for i, parent in enumerate(r['parents'][1:],1):
    target[i,parent] = target[parent,i] = 1
for k in range(n):
    target = np.minimum(target,target[:,k,None]+target[None,k,:])
idx = r['splits']['test']
assert len(set(r['splits']['train']) & set(idx)) == 0
for row in r['results']:
    z = torch.tensor(np.load(p / f"{row['geometry']}_{row['dim']}_{row['seed']}.npy"))
    actual = distances(z,row['geometry']).numpy()[a,b][idx]
    rmse = float(np.sqrt(np.mean((actual-target[a,b][idx])**2)))
    assert abs(rmse-row['test_rmse']) < 1e-10
print('SERVER:',socket.gethostname())
print('WORKSPACE:',Path.cwd())
print('VERIFIED: 18 saved embeddings; all test scores reproduced')
for dim in [2,4,8]:
    for geometry in ['euclidean','hyperbolic']:
        scores=[x['test_rmse'] for x in r['results'] if x['dim']==dim and x['geometry']==geometry]
        print(f'{dim}D {geometry}: RMSE={statistics.mean(scores):.6f} +/- {statistics.stdev(scores):.6f}')
print('SCOPE: skeleton tree distances only; no TriDi generation evaluation')
(p/'verification.txt').write_text('SERVER='+socket.gethostname()+'\n18 saved embeddings: test scores independently recomputed and matched.\n')
