"""Experiment 4 (D7): units and scale of the stored Aria positions. CPU only, seconds.

Run from the code directory on the cluster: python position_units_check.py [n_takes]
For a fixed random sample of training takes: dtype, finiteness, quaternion norm, total extent per axis, and the
3D step between consecutive stored frames, compared with the 3D step noise that per-frame Gaussian noise of
std sigma per coordinate adds (a difference of two noisy positions: sqrt(2) * sigma * sqrt(3) on average).
"""
import glob
import os
import random
import sys
import numpy as np

n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
root = os.path.join(os.environ.get('EGOEXO4D_PRETRAIN_TRAJ_DIR', os.path.expanduser(
    '~/dl4cv/data/camformer_data/data/egoexo4d_pretrain')), 'train_presr50_v2')
files = sorted(glob.glob(os.path.join(root, '*.npz')))
random.seed(0)
steps = []
print('take, entries, dtype, finite, quaternion norm, extent x/y/z (units), median step, p99 step')
for f in random.sample(files, n):
  z = np.load(f)
  for k in list(z.keys())[:1]:
    a = z[k]
    s = np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1)
    steps.append(np.median(s))
    print(os.path.basename(f), len(z.keys()), a.dtype, bool(np.isfinite(a).all()),
          round(float(np.linalg.norm(a[:, 3:], axis=1).mean()), 4), np.round(np.ptp(a[:, :3], 0), 2),
          round(float(np.median(s)), 4), round(float(np.percentile(s, 99)), 4))
steps = np.array(steps)
print(f'median step over takes: min {steps.min():.4f}, median {np.median(steps):.4f}, max {steps.max():.4f}')
for sigma in [0.005, 0.01, 0.02, 0.05]:
  noise = np.sqrt(2) * sigma * np.sqrt(3)  # typical 3D size of the noise in one step
  print(f'sigma {sigma}: step noise ~{noise:.4f}; takes whose median step is smaller: '
        f'{(steps < noise).sum()} of {len(steps)}')
