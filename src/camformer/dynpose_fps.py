"""Native frame rate of every DynPose-100K clip's ViPE poses, for resampling exo clips to the ego rate.

Usage (CPU, on the cluster, after scripts/deploy.sh):
    python dynpose_fps.py [--out data_files/dynpose100k_fps/vipe_fps_v1.csv.gz]

ViPE gives one pose per video frame (its `inds` are consecutive), at the video's own rate, which is not always 30 FPS
as the paper's supp. 2.2 states. The dataset's own poses index frames of a 12 FPS sequence (`frame_idxs`), with
some frames missing. A clip of T seconds therefore has about 12 T frames at 12 FPS and fps * T ViPE poses, so
    est = 12 * (number of ViPE poses) / (last 12 FPS frame index + 1).
Missing frames at the end of the 12 FPS sequence can only make `est` too high, and rounding moves it by up to about
0.6. The estimates fall into tight groups, assigned as: 14.5 <= est < 15.6 -> 15; 23 <= est < 24.5 -> 24;
24.5 <= est < 27.5 -> 25; 27.5 <= est <= 40 -> 30. Clips between the groups (25.5 to 29.5, or above 30.5) are marked
`ambiguous`. The few clips outside every group (first table: 69 of 99,501, 3 of them in the test) keep their own
estimate as the rate and are marked `ambiguous` too, so no clip is dropped and the test set is unchanged.
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from datasets.dynpose import _DEFAULT_DATA_DIR, _resolve_csv

SPLITS = ['metadata_train_v1.csv', 'metadata_val_v1.csv', 'metadata_samevideo_test5000.csv']


def native_fps(est):
  if 14.5 <= est < 15.6:
    return 15.0
  if 23.0 <= est <= 40.0:
    return 24.0 if est < 24.5 else 25.0 if est < 27.5 else 30.0
  return round(est, 2)  # outside every group: the estimate itself (marked ambiguous)


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
    est = 12.0 * len(inds) / (last + 1)
    rows.append((r.uid, len(inds), last, round(est, 3), native_fps(est)))
  out = pd.DataFrame(rows, columns=['uid', 'vipe_poses', 'last_12fps_index', 'est', 'fps'])
  grouped = out['est'].between(14.5, 15.6, inclusive='left') | out['est'].between(23.0, 40.0)
  out['ambiguous'] = out['est'].between(25.5, 29.5) | (out['est'] > 30.5) | ~grouped
  os.makedirs(os.path.dirname(opts.out), exist_ok=True)
  out.to_csv(opts.out, index=False)
  print(f'{len(out)} clips with ViPE poses written to {opts.out}')
  print('native rate:', out.loc[grouped, 'fps'].value_counts().sort_index().to_dict())
  print(f"ambiguous (between or outside the groups): {int(out['ambiguous'].sum())}"
        f" ({100 * out['ambiguous'].mean():.1f}%), of which outside every group: {int((~grouped).sum())}")
  print('est quantiles:', out['est'].quantile([0.001, 0.01, 0.5, 0.99, 0.999]).round(2).to_dict())


if __name__ == '__main__':
  main()
