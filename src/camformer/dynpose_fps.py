"""Native frame rate of every DynPose-100K clip's ViPE poses, for resampling exo clips to the ego rate.

Usage (CPU, on the cluster, after scripts/deploy.sh):
    python dynpose_fps.py [--out data_files/dynpose100k_fps/vipe_fps_v1.csv.gz]

ViPE gives one pose per video frame (its `inds` are consecutive), at the video's own rate, which is not always 30 FPS
as the paper's supp. 2.2 states. The dataset's own poses index frames of a 12 FPS sequence (`frame_idxs`), with
some frames missing. A clip of T seconds therefore has about 12 T frames at 12 FPS and fps * T ViPE poses, so
    est = 12 * (number of ViPE poses N) / (last 12 FPS frame index + 1 = L).
Missing frames at the end of the 12 FPS sequence can only make `est` too high. Rounding N and L by one frame each
can make it too low by at most delta = 12 N / L^2 + 12 / L (0.6 to 0.9 here), so the true rate is at most
est + delta. Rule (`native_fps`):
- est in a tight group -> that rate: 14.5-15.6 -> 15; 23.0-24.5 -> 24; 24.5-25.5 -> 25; 29.5-30.5 -> 30;
- between 25.5 and 29.5 -> 30 if est + delta >= 30, else 25 (a 25 FPS clip with missing end frames);
- above 30.5, up to 40 -> 30 (a 30 FPS clip with missing end frames);
- anything else keeps its own estimate.
All but the first case are marked `ambiguous`.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from datasets.dynpose import _DEFAULT_DATA_DIR, _resolve_csv

SPLITS = ['metadata_train_v1.csv', 'metadata_val_v1.csv', 'metadata_samevideo_test5000.csv']


GROUPS = [(14.5, 15.6, 15.0), (23.0, 24.5, 24.0), (24.5, 25.5, 25.0), (29.5, 30.5, 30.0)]


def native_fps(n_vipe, last_index):
  """(estimate, assigned rate, ambiguous) for one clip; the rule is in the module docstring."""
  L = last_index + 1
  est = 12.0 * n_vipe / L
  delta = 12.0 * n_vipe / L**2 + 12.0 / L
  for lo, hi, rate in GROUPS:
    if lo <= est < hi:
      return est, rate, False
  if 25.5 <= est < 29.5:
    return est, (30.0 if est + delta >= 30.0 else 25.0), True
  if 30.5 <= est <= 40.0:
    return est, 30.0, True
  return est, round(est, 2), True


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    'data_files', 'dynpose100k_fps', 'vipe_fps_v1.csv.gz'))
  opts = parser.parse_args()
  df = pd.concat([pd.read_csv(_resolve_csv(f)) for f in SPLITS], ignore_index=True)
  df = df[df['vipe_pose_path'].notnull()].drop_duplicates('uid').reset_index(drop=True)
  rows = []
  for r in df.itertuples():
    with open(os.path.join(_DEFAULT_DATA_DIR, r.pose_path), 'rb') as f:
      last = int(np.asarray(pickle.load(f)['frame_idxs'])[-1])
    inds = np.load(os.path.join(_DEFAULT_DATA_DIR, r.vipe_pose_path))['inds']
    if len(inds) > 1 and not (np.diff(inds) == 1).all():
      raise ValueError(f'{r.uid}: ViPE poses are not one per consecutive frame')
    est, rate, ambiguous = native_fps(len(inds), last)
    rows.append((r.uid, len(inds), last, round(est, 3), rate, ambiguous))
  out = pd.DataFrame(rows, columns=['uid', 'vipe_poses', 'last_12fps_index', 'est', 'fps', 'ambiguous'])
  os.makedirs(os.path.dirname(opts.out), exist_ok=True)
  out.to_csv(opts.out, index=False)
  print(f'{len(out)} clips with ViPE poses written to {opts.out}')
  standard = out['fps'].isin([g[2] for g in GROUPS])
  print('assigned rate:', out.loc[standard, 'fps'].value_counts().sort_index().to_dict())
  print(f"ambiguous: {int(out['ambiguous'].sum())} ({100 * out['ambiguous'].mean():.1f}%), of which between 25 and 30"
        f" -> 25: {int((out.ambiguous & out.est.between(25.5, 29.5, inclusive='left') & (out.fps == 25)).sum())},"
        f" -> 30: {int((out.ambiguous & out.est.between(25.5, 29.5, inclusive='left') & (out.fps == 30)).sum())};"
        f" own estimate: {int((~standard).sum())}")
  print('est quantiles:', out['est'].quantile([0.001, 0.01, 0.5, 0.99, 0.999]).round(2).to_dict())


if __name__ == '__main__':
  main()
