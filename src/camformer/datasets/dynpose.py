# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""DynPose-100K camera-pose / text pairs for exocentric CamFormer pretraining.

This is the exocentric counterpart to `egoexo4d_pretrain.py`. Each sample is a
(camera-pose trajectory, video caption) pair, where the trajectory is either the
dataset's original pose (`--pose_source original`) or a ViPE-estimated pose
(`--pose_source vipe`). Pretraining runs through the base `tasks/pretrain.py`
LightningModule (the dataset name has no `longseq`, so `train.py` selects the
base task), with the flat `pose_text_collate` from `loader.py`.
"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp
import torch
from utils.dataset_utils import matrix_to_pose7d, transform_camera_pose, w2c_to_c2w

# Frame rate of the stored poses as supp. 2.2 states it: ViPE at 30 FPS, the dataset's own poses at 12 FPS. ViPE
# gives one pose per video frame, and about a third of the videos are 24 or 25 FPS, not 30 (dynpose_fps.py), so
# --exo_fps_table gives the per-clip rate; without it every ViPE clip is taken as 30 FPS (experiment 2, first runs).
NATIVE_FPS = {'vipe': 30.0, 'original': 12.0}
# OpenCV camera axes (x right, y down, z forward) to the ego camera frame as stored in the Ego-Exo4D
# trajectories: x down, y left, z forward. Measured with exo_checks.py (the ego camera x axis has a mean
# world-up component of -0.81, y +0.11, z -0.19), so a 90-degree turn about z, not the 180-degree turn that
# supp. 2.3 describes for its (x left, y up) frame. Right-multiplying camera-to-world matrices by this maps
# the columns: ego x = OpenCV y, ego y = -OpenCV x, ego z = OpenCV z.
_OPENCV_TO_ARIA = np.array(
    [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)


def resample_poses(poses7d, src_hz, dst_hz):
  """(N, 7) poses at src_hz -> poses at dst_hz over the same time span.

  Translation is interpolated linearly and rotation spherically.
  """
  n = len(poses7d)
  if n < 2 or src_hz == dst_hz:
    return poses7d
  t_src = np.arange(n) / src_hz
  # The last step can overshoot the clip end by a rounding error; clamp it.
  t_dst = np.minimum(np.arange(0.0, t_src[-1] + 1e-9, 1.0 / dst_hz), t_src[-1])
  trans = np.stack([np.interp(t_dst, t_src, poses7d[:, i]) for i in range(3)], axis=1)
  rots = Slerp(t_src, Rotation.from_quat(poses7d[:, 3:]))(t_dst).as_quat()
  return np.concatenate([trans, rots], axis=1)

# Packaged metadata CSVs shipped alongside this module (if present).
_PKG_DATA_FILES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data_files',
    'dynpose100k',
)
# Base dir holding `data_files/dynpose100k/metadata_*.csv` and the per-clip pose
# files referenced (relative to this dir) by the metadata. Overridable via
# DYNPOSE_DATA_DIR; defaults to the public `~/final_data` bundle.
_DEFAULT_DATA_DIR = os.environ.get(
    'DYNPOSE_DATA_DIR',
    os.path.expanduser('~/final_data'),
)


def _resolve_csv(fn):
  """Prefer the bundled data_files/dynpose100k/<fn>; fall back to the bundle."""
  local = os.path.join(_PKG_DATA_FILES_DIR, fn)
  if os.path.exists(local):
    return local
  return os.path.join(_DEFAULT_DATA_DIR, 'data_files', 'dynpose100k', fn)


class DynPoseCamTextPair(torch.utils.data.Dataset):
  """DynPose-100K (exocentric) camera-pose / caption pairs.

  Each sample yields: (cam_traj (N, D), caption). The pose source is chosen by
  `args.pose_source`: 'original' (the dataset's own poses) or 'vipe'
  (ViPE-estimated). Clips whose pose scale exceeds a threshold are dropped as
  unreliable. Pose files are read relative to DYNPOSE_DATA_DIR.
  """

  def __init__(self, args, mode):
    self.args = args
    self.encode_pose = args.encode_pose if args is not None else 2
    self.sample_rate = args.sample_rate if args is not None else 1
    self.mode = mode
    self.pose_source = 'original' if args is None else args.pose_source
    # Experiment 2 options (all off by default, which is the authors' exo setup):
    # longseq: samples in the Ego-Exo4D long-sequence format, the labelled window being the whole clip.
    self.longseq = args is not None and getattr(args, 'exo_longseq', False)
    self.resample_hz = 0.0 if args is None else getattr(args, 'exo_resample_hz', 0.0)
    self.to_aria_axes = args is not None and getattr(args, 'exo_to_aria_axes', False)
    # Per-clip native rate of the ViPE poses (uid -> FPS), used when resampling; None = 30 FPS for every clip.
    fps_table = '' if args is None else getattr(args, 'exo_fps_table', '')
    self.fps_table = None
    if fps_table:
      assert self.pose_source == 'vipe', '--exo_fps_table holds ViPE frame rates'
      assert self.resample_hz > 0, '--exo_fps_table is used only when resampling (--exo_resample_hz)'
      if not os.path.isabs(fps_table):  # relative to our data_files/ (not the upstream dynpose100k/ link)
        fps_table = os.path.join(os.path.dirname(_PKG_DATA_FILES_DIR), fps_table)
      t = pd.read_csv(fps_table)
      self.fps_table = dict(zip(t['uid'], t['fps']))
    # Test split: the 5-way test (default) or the validation clips that are not in the test (for choices
    # that must not be made on the test).
    self.eval_split = 'test' if args is None else getattr(args, 'exo_eval_split', 'test')
    assert self.pose_source in [
        'original',
        'vipe',
    ], f'Unsupported pose_source: {self.pose_source}'
    self.scale_threshold = 20
    # CLIP's context is 77 tokens; truncate text to keep it within budget.
    self.approx_max_chars = 300
    print(
        f'DynPose: encode_pose={self.encode_pose},'
        f' sample_rate={self.sample_rate}, pose_source={self.pose_source},'
        f' longseq={self.longseq}, resample_hz={self.resample_hz},'
        f' to_aria_axes={self.to_aria_axes}, fps_table={fps_table or "none (30 FPS)"}'
    )
    self.build_dataset()

  def build_dataset(self):
    # Test = the 5-way MCQ set (samevideo_test5000); train/val use the full splits.
    fn = 'samevideo_test5000' if self.mode == 'test' else f'{self.mode}_v1'
    if self.mode == 'test' and self.eval_split == 'val_nontest':
      fn = 'val_v1'
    df = pd.read_csv(_resolve_csv(f'metadata_{fn}.csv'))
    if self.mode == 'test' and self.eval_split == 'val_nontest':
      test_uids = set(pd.read_csv(_resolve_csv('metadata_samevideo_test5000.csv'))['uid'])
      df = df[~df['uid'].isin(test_uids)].reset_index(drop=True)
    if self.pose_source == 'vipe':
      df = df[df['vipe_pose_path'].notnull()].reset_index(drop=True)
    col_name = (
        'pose_scale' if self.pose_source == 'original' else 'vipe_pose_scale'
    )
    self.df = df[df[col_name] < self.scale_threshold].reset_index(drop=True)
    if self.fps_table is not None:  # every clip must have a rate; check now, not hours into training
      missing = ~self.df['uid'].isin(self.fps_table.keys())
      if missing.any():
        raise ValueError(f'{int(missing.sum())} clips have no rate in --exo_fps_table, e.g. {self.df.uid[missing].iloc[0]}')
    print(f'DynPose mode={self.mode}: loaded {len(self.df)}/{len(df)} samples')

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    if self.pose_source == 'original':
      pose_path = os.path.join(
          _DEFAULT_DATA_DIR, self.df.iloc[idx]['pose_path']
      )
      with open(pose_path, 'rb') as f:
        poses = pickle.load(f)['poses']
        poses = w2c_to_c2w(poses)
    else:
      pose_path = os.path.join(
          _DEFAULT_DATA_DIR, self.df.iloc[idx]['vipe_pose_path']
      )
      poses = np.load(pose_path)['data']  # (N, 4, 4)
    if self.to_aria_axes:
      bottom = np.tile([[[0, 0, 0, 1.0]]], (len(poses), 1, 1))
      poses = np.concatenate([poses[:, :3, :], bottom], axis=1) @ _OPENCV_TO_ARIA  # camera-to-world
    poses = matrix_to_pose7d(poses)
    row = self.df.iloc[idx]
    if self.resample_hz > 0:
      src_hz = NATIVE_FPS[self.pose_source]
      if self.fps_table is not None:
        src_hz = float(self.fps_table[row['uid']])
      poses = resample_poses(poses, src_hz, self.resample_hz)
    poses = poses[:: self.sample_rate]
    caption = row['description_text'][: self.approx_max_chars]
    if not self.longseq:
      poses = transform_camera_pose(poses, self.encode_pose)
      return poses, caption
    # Long-sequence format. The ego encoding (11) carries a gravity vector; exo clips have none, so its
    # three columns are zero and the model uses its learned "gravity unknown" token for them.
    poses = transform_camera_pose(poses, 2 if self.encode_pose == 11 else self.encode_pose)
    if self.encode_pose == 11:
      poses = np.hstack([poses, np.zeros((len(poses), 3))])
    # [t0, t1, caption, video id (plays the role of the take), domain 1 = exo]; scenario label -1 (none).
    return poses, [0, len(poses), caption, row['video_uid'], 1], -1, []
