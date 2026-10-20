"""Eye-gaze features for Ego-Exo4D pose clips (experiment 3, docs/gaze.md).

Gaze comes from each take's `eye_gaze/general_eye_gaze.csv` (Aria MPS, 10 Hz). `gaze_prep.py` turns it into one
small cache file per take, `<EGOEXO4D_GAZE_DIR>/<take>.npz`, with times in seconds from the first row of the
take's trajectory (the clock that the CSVs' `start_time` uses) and only the samples where yaw, pitch and depth are
all valid.

Per pose frame the features are: yaw (mean of both eyes), pitch, log depth (depth clamped to at least 0.1 m), and a
validity flag. They are linearly interpolated at the frame's time; a frame with no valid sample within 0.2 s gets
zeros and flag 0. Stored pose frame j of a clip is at `start_time + j * frame_dt` (data_prep/step3_build_longseq.py
cuts rows int(start_time*1e3) onwards of the 1 kHz trajectory with stride pre_sample_rate).
"""

import functools
import os

import numpy as np
import pandas as pd

GAZE_DIR = os.environ.get(
    'EGOEXO4D_GAZE_DIR', os.path.expanduser('~/dl4cv/data/egoexo4d/gaze_cache')
)
DONOR_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data_files',
    'egoexo4d',
    'gaze_donors_v1.csv.gz',
)
MAX_GAP_S = 0.2
MIN_DEPTH_M = 0.1
GAZE_DIM = 4


@functools.lru_cache(maxsize=None)
def load_take(take, gaze_dir=None):
  """(t, yaw, pitch, log_depth) arrays of a take's valid gaze samples, sorted by time; None if no gaze."""
  path = os.path.join(gaze_dir or GAZE_DIR, take + '.npz')
  if not os.path.exists(path):
    return None
  z = np.load(path)
  return z['t'], z['yaw'], z['pitch'], z['log_depth']


def features_at(times, take, gaze_dir=None, end_time=None):
  """(len(times), 4) float32 gaze features at the given take times (seconds from the trajectory start)."""
  times = np.asarray(times, dtype=np.float64)
  out = np.zeros((len(times), GAZE_DIM), dtype=np.float32)
  g = load_take(take, gaze_dir)
  if g is None or len(times) == 0:
    return out
  t, yaw, pitch, logd = g
  if len(t) == 0:
    return out
  # Distance to the nearest valid sample decides the flag.
  i = np.searchsorted(t, times)
  left = np.abs(times - t[np.clip(i - 1, 0, len(t) - 1)])
  right = np.abs(t[np.clip(i, 0, len(t) - 1)] - times)
  valid = np.minimum(left, right) <= MAX_GAP_S
  if end_time is not None:
    valid &= times <= end_time
  out[:, 0] = np.interp(times, t, yaw)
  out[:, 1] = np.interp(times, t, pitch)
  out[:, 2] = np.interp(times, t, logd)
  out[:, 3] = 1.0
  out[~valid] = 0.0
  return out


@functools.lru_cache(maxsize=None)
def load_donors(path=DONOR_FILE):
  """save_id -> (donor_save_id, donor_start_time, donor_end_time); donor_save_id is '' if the clip has none."""
  df = pd.read_csv(path, keep_default_na=False)
  return {
      r.save_id: (r.donor_save_id, float(r.donor_start_time or 0), float(r.donor_end_time or 0))
      for r in df.itertuples()
  }


def clip_gaze(mode, row, frame_idx, frame_dt):
  """Gaze features for the stored frames `frame_idx` of a CSV row.

  mode: 'real' (this clip), 'shuffled' (the committed donor clip of the same take, at the same relative frames,
  flag 0 past the donor's window), or 'zero' (all zeros, for checking that a trained model uses gaze).
  """
  frame_idx = np.asarray(frame_idx)
  if mode == 'zero':
    return np.zeros((len(frame_idx), GAZE_DIM), dtype=np.float32)
  rel = frame_idx * frame_dt
  if mode == 'real':
    return features_at(row['start_time'] + rel, row['take_name'])
  if mode == 'shuffled':
    donor, d_start, d_end = load_donors()[row['save_id']]
    if not donor:
      return np.zeros((len(frame_idx), GAZE_DIM), dtype=np.float32)
    return features_at(d_start + rel, row['take_name'], end_time=d_end)
  raise ValueError(f'unknown gaze mode {mode}')


def build_donor_map(windows, takes_with_gaze, seed=20261020):
  """One donor per clip window: same take, has gaze, a different window that does not overlap it in time.

  windows: DataFrame with save_id, take_name, start_time, end_time (one row per distinct save_id).
  The draw uses its own fixed seed, independent of any training seed. Clips without a valid donor get ''.
  """
  rng = np.random.default_rng(seed)
  rows = []
  for take, g in windows.sort_values(['take_name', 'start_time', 'save_id']).groupby('take_name', sort=True):
    ids = g['save_id'].to_numpy()
    s = g['start_time'].to_numpy()
    e = g['end_time'].to_numpy()
    for k in range(len(g)):
      cand = np.where((e <= s[k]) | (s >= e[k]))[0] if take in takes_with_gaze else np.array([], dtype=int)
      if len(cand):
        j = int(rng.choice(cand))
        rows.append((ids[k], ids[j], round(s[j], 5), round(e[j], 5)))
      else:
        rows.append((ids[k], '', '', ''))
  return pd.DataFrame(rows, columns=['save_id', 'donor_save_id', 'donor_start_time', 'donor_end_time'])
