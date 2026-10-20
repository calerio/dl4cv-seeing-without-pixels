"""Experiment 3 preparation and checks, CPU only (docs/gaze.md, code change 6).

Steps (each prints its result; run all with no arguments on the cluster, see scripts/slurm/gaze_prep.sh):
  cache     per-take gaze cache <EGOEXO4D_GAZE_DIR>/<take>.npz from eye_gaze/general_eye_gaze.csv
  khz       1 kHz-rows check: per take, max(end_time) <= trajectory length from traj_t0.csv
  donors    build the donor map data_files/egoexo4d/gaze_donors_v1.csv.gz and check every donor
  coverage  share of valid gaze frames over the model's input windows, per split and activity type
  vor       alignment check: eye-in-head yaw rate against head yaw rate, cross-correlation by lag

Usage: python gaze_prep.py [step ...] --takes_dir ~/dl4cv/data/egoexo4d/takes --traj_t0 ~/dl4cv/gaze_check/traj_t0.csv
"""

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd

from datasets import gaze
from datasets.egoexo4d_pretrain import EgoExo4DCameraPoseLongSeqForPretraining, _resolve_csv

PROCEDURAL = {'Cooking', 'Health', 'Bike Repair'}
SPLITS = {'train': 'train_v2.csv', 'val': 'val_v2.csv', 'test': 'test_alltasks_mcqv0.csv'}


def take_of(save_id):
  return '_'.join(save_id.split('_')[:-2])


def split_frames():
  out = {}
  for split, fn in SPLITS.items():
    df = pd.read_csv(_resolve_csv(fn))
    df['take_name'] = df['save_id'].map(take_of)
    out[split] = df
  return out


def step_cache(opts, frames):
  t0 = pd.read_csv(opts.traj_t0).set_index('take')['traj_t0_us']
  takes = sorted(set().union(*[set(df['take_name']) for df in frames.values()]))
  os.makedirs(gaze.GAZE_DIR, exist_ok=True)
  done = missing = 0
  for take in takes:
    out = os.path.join(gaze.GAZE_DIR, take + '.npz')
    src = os.path.join(opts.takes_dir, take, 'eye_gaze', 'general_eye_gaze.csv')
    if not os.path.exists(src) or take not in t0.index or pd.isna(t0[take]):
      missing += 1
      continue
    g = pd.read_csv(src, usecols=['tracking_timestamp_us', 'left_yaw_rads_cpf', 'right_yaw_rads_cpf',
                                  'pitch_rads_cpf', 'depth_m'])
    yaw = (g['left_yaw_rads_cpf'] + g['right_yaw_rads_cpf']) / 2
    ok = np.isfinite(yaw) & np.isfinite(g['pitch_rads_cpf']) & np.isfinite(g['depth_m']) & (g['depth_m'] > 0)
    t = (g['tracking_timestamp_us'][ok].to_numpy() - int(t0[take])) / 1e6
    order = np.argsort(t)
    np.savez(out, t=t[order],
             yaw=yaw[ok].to_numpy(np.float32)[order],
             pitch=g['pitch_rads_cpf'][ok].to_numpy(np.float32)[order],
             log_depth=np.log(np.maximum(g['depth_m'][ok].to_numpy(np.float32), gaze.MIN_DEPTH_M))[order])
    done += 1
  print(f'[cache] {done} takes cached to {gaze.GAZE_DIR}; {missing} without gaze file or traj_t0')


def step_khz(opts, frames):
  t = pd.read_csv(opts.traj_t0).set_index('take')
  length = (t['traj_t1_us'] - t['traj_t0_us']) / 1e6
  allrows = pd.concat(frames.values())
  m = allrows.groupby('take_name')['end_time'].max()
  m = m[m.index.isin(length.index)]
  diff = m - length[m.index]
  # end_time = min(len(traj)/1e3, ts + duration): the last clip is cut at the take end when its window reaches it.
  print(f'[khz] {len(m)} takes. max(end_time) - trajectory length (s): max {diff.max():+.4f}, '
        f'takes over +0.005: {(diff > 0.005).sum()}; takes within 0.005 of the end: {(diff.abs() <= 0.005).sum()}')
  bad = diff[diff > 0.005]
  if len(bad):
    print('[khz] FAIL, e.g.', bad.head().to_dict())
  else:
    print('[khz] OK: no clip ends after its trajectory')


def step_donors(opts, frames):
  win = pd.concat([df[['save_id', 'take_name', 'start_time', 'end_time']] for df in frames.values()])
  win = win.drop_duplicates('save_id')
  with_gaze = {f[:-4] for f in os.listdir(gaze.GAZE_DIR) if f.endswith('.npz')}
  donors = gaze.build_donor_map(win, with_gaze)
  donors.to_csv(gaze.DONOR_FILE, index=False)
  # Checks: same take, different clip, no overlap in time, every clip present.
  d = donors[donors['donor_save_id'] != ''].merge(win, on='save_id').merge(
      win.add_prefix('d_'), left_on='donor_save_id', right_on='d_save_id')
  same_take = (d['take_name'] == d['d_take_name']).all()
  overlap = ((d['d_start_time'] < d['end_time']) & (d['d_end_time'] > d['start_time'])).sum()
  print(f'[donors] {len(donors)} clip windows, {len(d)} with a donor, {len(donors) - len(d)} without; '
        f'same take: {same_take}; overlapping in time: {overlap}; written to {gaze.DONOR_FILE}')
  assert same_take and overlap == 0 and (d['save_id'] != d['donor_save_id']).all()
  assert set(donors['save_id']) == set(win['save_id'])


def step_coverage(opts, frames):
  scen = pd.read_csv(_resolve_csv('scenario_cls.csv')).set_index('take_name')['task']
  random.seed(0)
  for split in SPLITS:
    a = argparse.Namespace(use_pi3_pose=False, pre_sample_rate=50, sample_rate=1, encode_pose=11, ego_visible=False,
                           sample_dur=True, scenario='all', take_duration=8, test_take_duration=4, start_ratio=0.2,
                           test_time_ratio=0.5, gaze='real')
    ds = EgoExo4DCameraPoseLongSeqForPretraining(a, 'test' if split == 'test' else split)
    if split == 'val':
      ds.mode = 'test'  # deterministic windows, as when validation clips are tested
    idx = list(range(len(ds)))
    if split == 'train' and len(idx) > opts.train_sample:
      idx = random.sample(idx, opts.train_sample)
    rows = []
    for i in idx:
      row = ds.df.iloc[i]
      frames_i, _ = ds.window(row)
      g = gaze.clip_gaze('real', row, frames_i, ds.frame_dt)
      rows.append((scen.get(row['take_name'], ''), g[:, 3].mean() if len(g) else 0.0))
    r = pd.DataFrame(rows, columns=['scenario', 'valid'])
    r['type'] = np.where(r['scenario'].isin(PROCEDURAL), 'procedural', 'physical')
    for t, g in r.groupby('type'):
      print(f'[coverage] {split:5s} {t:10s} clips {len(g):6d}  mean valid share {100 * g.valid.mean():.2f}%  '
            f'clips with >=90% valid {100 * (g.valid >= 0.9).mean():.2f}%')


def head_yaw_rate(quat_xyzw, axis):
  """Rotation rate (rad per frame) about a device axis, from consecutive orientations."""
  from scipy.spatial.transform import Rotation as R
  r = R.from_quat(quat_xyzw)
  rel = (r[:-1].inv() * r[1:]).as_rotvec()  # in the device frame of the earlier pose
  return rel[:, axis]


def step_vor(opts, frames):
  """Eye-in-head yaw rate vs head rotation rate about each device axis, by lag (in 50 ms frames).

  The vestibulo-ocular reflex makes the eyes counter-rotate when the head turns, so for the device axis that is
  vertical the correlation should be strongly negative or positive (depending on sign conventions) at lag 0 and
  weaker at other lags. The same is computed with each clip's donor gaze as a negative control.
  """
  a = argparse.Namespace(use_pi3_pose=False, pre_sample_rate=50, sample_rate=1, encode_pose=11, ego_visible=False,
                         sample_dur=True, scenario='all', take_duration=8, test_take_duration=4, start_ratio=0.2,
                         test_time_ratio=0.5, gaze='real')
  ds = EgoExo4DCameraPoseLongSeqForPretraining(a, 'val')
  rng = random.Random(0)
  idx = rng.sample(range(len(ds)), opts.vor_clips)
  lags = range(-opts.max_lag, opts.max_lag + 1)
  for source in ('real', 'shuffled'):
    acc = {(ax, lag): [] for ax in range(3) for lag in lags}
    for i in idx:
      row = ds.df.iloc[i]
      traj = ds._load_cam_traj(row)  # stored window, (N, 7): t xyz, q xyzw
      n = len(traj)
      g = gaze.clip_gaze(source, row, np.arange(n), ds.frame_dt)
      ok = (g[:-1, 3] > 0) & (g[1:, 3] > 0)
      if ok.sum() < 40:
        continue
      eye = np.diff(g[:, 0])
      for ax in range(3):
        head = head_yaw_rate(traj[:, 3:7], ax)
        for lag in lags:
          # eye[k] against head[k + lag]
          k = np.arange(max(0, -lag), min(len(eye), len(head) - lag))
          m = ok[k]
          acc[(ax, lag)].append(np.stack([eye[k][m], head[k + lag][m]]))
    print(f'[vor] gaze source: {source}, {opts.vor_clips} validation clips sampled')
    for ax in range(3):
      line = []
      for lag in lags:
        e, h = np.concatenate(acc[(ax, lag)], axis=1)
        line.append(f'{lag:+d}:{np.corrcoef(e, h)[0, 1]:+.3f}')
      print(f'[vor]   device axis {ax}: ' + ' '.join(line))


def main():
  p = argparse.ArgumentParser()
  p.add_argument('steps', nargs='*', default=['cache', 'khz', 'donors', 'coverage', 'vor'])
  p.add_argument('--takes_dir', default=os.path.expanduser('~/dl4cv/data/egoexo4d/takes'))
  p.add_argument('--traj_t0', default=os.path.expanduser('~/dl4cv/gaze_check/traj_t0.csv'))
  p.add_argument('--train_sample', type=int, default=30000)
  p.add_argument('--vor_clips', type=int, default=1000)
  p.add_argument('--max_lag', type=int, default=5)
  opts = p.parse_args()
  frames = split_frames()
  for s in opts.steps:
    globals()[f'step_{s}'](opts, frames)
    sys.stdout.flush()


if __name__ == '__main__':
  main()
