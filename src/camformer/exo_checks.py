"""CPU checks for experiment 2 (docs/ego_exo.md, the items marked Open). No model, no GPU.

Usage (from the code directory, with the data environment variables of the job scripts):
    python exo_checks.py [--n 2000] [--seed 0]

Prints, on a random sample of training clips of each domain:
  1. Exo (DynPose ViPE) clip length in frames and seconds, and the step of the stored frame indices
     (1 = every video frame, i.e. the native 30 FPS).
  2. Translation speed (metres per second, from consecutive camera positions) for ego windows and exo
     clips: median and 90th percentile of the per-clip median speed.
  3. Axes. Ego: how the Aria camera axes sit against the world up axis (z, the gravity used by
     get_gravity_in_camera). Exo: whether the first pose is the identity (world = first camera) and the
     direction of motion in the first camera's frame. Pi3: the first pose of a few files.
  4. The DynPose test set after the loader's filters (must stay 5,000 rows for the 5-way grouping), and
     the number of validation clips that are not in the test set.
"""

import argparse
import glob
import os
import random

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

CODE = os.path.dirname(os.path.abspath(__file__))
EGO_CSV = os.path.join(CODE, 'data_files', 'egoexo4d')
DYN_CSV = os.path.join(CODE, 'data_files', 'dynpose100k')
EGO_TRAJ = os.path.join(os.environ['EGOEXO4D_PRETRAIN_TRAJ_DIR'], 'train_presr50_v2')
DYN_DIR = os.environ['DYNPOSE_DATA_DIR']
EGO_FPS, EXO_FPS = 20.0, 30.0


def pct(x):
  x = np.asarray(x, float)
  return f'median {np.median(x):.3f}, p10 {np.percentile(x, 10):.3f}, p90 {np.percentile(x, 90):.3f}'


def axis_shares(dirs):
  """Share of unit vectors whose largest component is +x, -x, +y, -y, +z, -z."""
  dirs = np.asarray(dirs)
  i = np.abs(dirs).argmax(1)
  sign = np.sign(dirs[np.arange(len(dirs)), i])
  return ' '.join(f'{np.mean((i == a) & (sign == sg)):.2f}' for a in range(3) for sg in (1, -1))


def speeds(pos, fps):
  return np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--n', type=int, default=2000)
  parser.add_argument('--seed', type=int, default=0)
  opts = parser.parse_args()
  rng = random.Random(opts.seed)

  # Exo: lengths, index step, speed, first pose, motion direction.
  train = pd.read_csv(os.path.join(DYN_CSV, 'metadata_train_v1.csv'))
  train = train[train['vipe_pose_path'].notnull() & (train['vipe_pose_scale'] < 20)]
  rows = train.sample(n=min(opts.n, len(train)), random_state=opts.seed)
  lengths, steps, exo_speed, first_dev, dirs, cam_y_world = [], [], [], [], [], []
  for p in rows['vipe_pose_path']:
    z = np.load(os.path.join(DYN_DIR, p))
    T, inds = z['data'], z['inds']
    lengths.append(len(T))
    steps.extend(np.unique(np.diff(inds)).tolist())
    exo_speed.append(np.median(speeds(T[:, :3, 3], EXO_FPS)))
    first_dev.append(np.abs(T[0, :3, :] - np.eye(4)[:3, :]).max())
    d = T[-1, :3, 3] - T[0, :3, 3]
    if np.linalg.norm(d) > 1e-6:
      dirs.append(d / np.linalg.norm(d))
    cam_y_world.append(T[:, :3, 1].mean(0))
  lengths = np.array(lengths)
  print(f'[exo] {len(rows)} ViPE training clips')
  print(f'  length in frames: {pct(lengths)}; max {lengths.max()}')
  print(f'  length in seconds at 30 FPS: {pct(lengths / EXO_FPS)}')
  print(f'  frame index steps seen: {sorted(set(steps))[:10]}')
  print(f'  length at 20 FPS after resampling, frames: {pct(lengths * 20 / 30)}')
  print(f'  first pose differs from identity by at most: {pct(first_dev)}')
  dirs = np.array(dirs)
  print(f'  net motion direction in the first camera frame (mean of unit vectors): {dirs.mean(0).round(3)}')
  print(f'  dominant axis of the net motion, shares of +x -x +y -y +z -z: {axis_shares(dirs)}')
  print(f'  mean camera y axis in the first camera frame: {np.mean(cam_y_world, 0).round(3)}')

  # Ego: speed over the labelled windows, and the camera axes against world up.
  ego = pd.read_csv(os.path.join(EGO_CSV, 'train_v2.csv'))
  ego['take_name'] = ego['save_id'].str.rsplit('_', n=2).str[0]
  ego_rows = ego.sample(n=min(opts.n, len(ego)), random_state=opts.seed).sort_values('take_name')
  ego_speed, up_dots, ego_dirs = [], {0: [], 1: [], 2: []}, []
  cache = {}
  for _, r in ego_rows.iterrows():
    f = os.path.join(EGO_TRAJ, r['take_name'] + '.npz')
    if f not in cache:
      cache.clear()
      cache[f] = np.load(f)
    traj = cache[f][r['save_id']]  # (N, 7): t, quaternion (x, y, z, w)
    lo = max(0, int(r['timestamp_idx']) - 80)
    w = traj[lo: lo + 160]
    if len(w) < 3:
      continue
    ego_speed.append(np.median(speeds(w[:, :3], EGO_FPS)))
    R = Rotation.from_quat(w[:, 3:]).as_matrix()
    d = R[0].T @ (w[-1, :3] - w[0, :3])  # net motion in the first camera's frame (if quats are camera-to-world)
    if np.linalg.norm(d) > 0.3:
      ego_dirs.append(d / np.linalg.norm(d))
    for a in range(3):
      up_dots[a].append(R[:, 2, a].mean())  # camera axis a, its world-z component
  print(f'[ego] {len(ego_speed)} training windows (8 s around the narration)')
  print(f'  camera x / y / z axis, mean world-up component: '
        f'{np.mean(up_dots[0]):.3f} / {np.mean(up_dots[1]):.3f} / {np.mean(up_dots[2]):.3f}')
  ego_dirs = np.array(ego_dirs)
  print(f'  {len(ego_dirs)} windows moving > 0.3 m: net motion in the first camera frame, mean of unit vectors '
        f'{ego_dirs.mean(0).round(3)}')
  print(f'  dominant axis of that motion, shares of +x -x +y -y +z -z: {axis_shares(ego_dirs)}')

  ego_speed, exo_speed = np.array(ego_speed), np.array(exo_speed)
  print('[speed] per-clip median translation speed (units per second)')
  print(f'  ego: {pct(ego_speed)}')
  print(f'  exo: {pct(exo_speed)}')
  print(f'  ratio of medians ego/exo: {np.median(ego_speed) / np.median(exo_speed):.3f}')

  # Pi3: first pose of a few files.
  pi3 = sorted(glob.glob(os.path.join(os.path.dirname(os.environ['EGOEXO4D_PRETRAIN_TRAJ_DIR']), '..', 'pi3_preds', '*', '*.npy')))
  print(f'[pi3] {len(pi3)} files')
  for f in rng.sample(pi3, min(3, len(pi3))):
    a = np.load(f)
    print(f'  {os.path.relpath(f, os.path.dirname(f) + "/..")}: shape {a.shape}, first pose\n{np.round(a[0], 3)}')

  # Test and validation sets.
  test = pd.read_csv(os.path.join(DYN_CSV, 'metadata_samevideo_test5000.csv'))
  kept = test[test['vipe_pose_path'].notnull() & (test['vipe_pose_scale'] < 20)]
  val = pd.read_csv(os.path.join(DYN_CSV, 'metadata_val_v1.csv'))
  val = val[val['vipe_pose_path'].notnull() & (val['vipe_pose_scale'] < 20)]
  print(f'[sets] DynPose test rows kept by the ViPE filters: {len(kept)} of {len(test)}')
  print(f'  validation clips (ViPE filters) not in the test set: {(~val["uid"].isin(test["uid"])).sum()}')


if __name__ == '__main__':
  main()
