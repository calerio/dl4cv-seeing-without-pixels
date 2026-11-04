"""Unit checks for the experiment-2 code, on synthetic data (CPU, seconds). Run: python test_ego_exo.py"""

import argparse
import os
import random
import tempfile

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import torch

import datasets.dynpose as dynpose
from datasets.dynpose import _OPENCV_TO_ARIA, resample_poses
from loader import DomainBatchSampler, domain_ids
from models.cm_encoder import CameraPoseSeqEncoder


def encoder(**kw):
  torch.manual_seed(0)
  return CameraPoseSeqEncoder(encode_pose=11, output_dim=512, pooling='no', **kw).eval()


def test_domain_token_starts_as_identity():
  x = torch.randn(4, 30, 12)
  pad = torch.zeros(4, 30, dtype=torch.bool)
  lengths = torch.full((4,), 30)
  base, tok = encoder(), encoder(domain_token=True)
  tok.load_state_dict(base.state_dict(), strict=False)  # domain_embed stays at zero
  out_base = base(x, pad, lengths)
  out_tok = tok(x, pad, lengths, domain_ids=torch.tensor([0, 1, 0, 1]))
  assert torch.equal(out_base, out_tok)
  # and it changes the output once it is non-zero
  with torch.no_grad():
    tok.domain_embed.weight[1] += 1.0
  assert not torch.allclose(out_base[1], tok(x, pad, lengths, domain_ids=torch.tensor([0, 1, 0, 1]))[1])


def test_gravity_mask():
  enc = encoder()
  x = torch.randn(3, 20, 12)
  pad = torch.zeros(3, 20, dtype=torch.bool)
  lengths = torch.full((3,), 20)
  plain = enc(x, pad, lengths)
  all_true = enc(x, pad, lengths, has_gravity=torch.tensor([True, True, True]))
  assert torch.equal(plain, all_true)  # ego clips: unchanged
  mixed = enc(x, pad, lengths, has_gravity=torch.tensor([True, False, True]))
  assert torch.equal(plain[0], mixed[0]) and not torch.allclose(plain[1], mixed[1])
  # an exo clip's gravity columns are ignored: any values give the same output
  x2 = x.clone()
  x2[1, :, 9:] = 5.0
  assert torch.equal(mixed[1], enc(x2, pad, lengths, has_gravity=torch.tensor([True, False, True]))[1])


def test_resample():
  t = np.arange(31) / 30.0  # 1 s at 30 FPS: constant speed and constant turn rate
  trans = np.stack([t, 2 * t, np.zeros_like(t)], 1)
  quats = Rotation.from_euler('z', 90 * t, degrees=True).as_quat()
  out = resample_poses(np.concatenate([trans, quats], 1), 30.0, 20.0)
  assert len(out) == 21
  t20 = np.arange(21) / 20.0
  assert np.allclose(out[:, 0], t20) and np.allclose(out[:, 1], 2 * t20)
  ang = Rotation.from_quat(out[:, 3:]).as_euler('zyx', degrees=True)[:, 0]
  assert np.allclose(ang, 90 * t20, atol=1e-6)
  # every real clip length (97 to 256 frames) must resample without stepping past the clip end
  for n in range(2, 300):
    p = np.concatenate([np.zeros((n, 3)), Rotation.identity(n).as_quat()], 1)
    resample_poses(p, 30.0, 20.0)


def test_fps_table():
  # Two 2-second ViPE clips, one filmed at 24 FPS and one at 30 FPS. Resampled to 20 Hz they must both give 2 s of
  # poses (41); without the table both are taken as 30 FPS, so the 24 FPS clip comes out 20% short.
  with tempfile.TemporaryDirectory() as d:
    rows = []
    for uid, fps in (('a', 24), ('b', 30)):
      n = 2 * fps + 1
      poses = np.tile(np.eye(4), (n, 1, 1))
      poses[:, 0, 3] = np.arange(n) / fps
      np.savez(os.path.join(d, f'{uid}.npz'), data=poses, inds=np.arange(n))
      rows.append(dict(video_uid=uid, uid=uid, segment_id=0, pose_path='', vipe_pose_path=f'{uid}.npz',
                       description_text='x', pose_scale=1.0, vipe_pose_scale=1.0))
    pd.DataFrame(rows).to_csv(os.path.join(d, 'meta.csv'), index=False)
    pd.DataFrame(dict(uid=['a', 'b'], fps=[24.0, 30.0])).to_csv(os.path.join(d, 'fps.csv'), index=False)
    saved = dynpose._resolve_csv, dynpose._DEFAULT_DATA_DIR
    dynpose._resolve_csv, dynpose._DEFAULT_DATA_DIR = (lambda fn: os.path.join(d, 'meta.csv')), d
    try:
      def lengths(table):
        args = argparse.Namespace(encode_pose=2, sample_rate=1, pose_source='vipe', exo_resample_hz=20.0,
                                  exo_to_aria_axes=False, exo_fps_table=table)
        ds = dynpose.DynPoseCamTextPair(args, 'train')
        return [len(ds[i][0]) for i in range(len(ds))]
      assert lengths(os.path.join(d, 'fps.csv')) == [41, 41]
      assert lengths('') == [33, 41]
    finally:
      dynpose._resolve_csv, dynpose._DEFAULT_DATA_DIR = saved


def test_axes():
  M = _OPENCV_TO_ARIA[:3, :3]
  assert np.isclose(np.linalg.det(M), 1.0)
  # camera-to-world columns: ego x = OpenCV y (down), ego y = -OpenCV x (left), ego z = OpenCV z
  R = np.eye(3) @ M
  assert np.allclose(R[:, 0], [0, 1, 0]) and np.allclose(R[:, 1], [-1, 0, 0]) and np.allclose(R[:, 2], [0, 0, 1])


def test_sampler():
  random.seed(0)
  s = DomainBatchSampler([10, 7], 3)
  batches = list(iter(s))
  assert len(batches) == len(s) == 3 + 2
  for b in batches:
    assert len(b) == 3 and (all(i < 10 for i in b) or all(i >= 10 for i in b))
  flat = [i for b in batches for i in b]
  assert len(set(flat)) == len(flat)
  random.seed(0)
  assert batches == list(iter(DomainBatchSampler([10, 7], 3)))  # same seed, same order


def test_domain_ids():
  assert domain_ids([[0, 3, 'a', 'take'], [0, 5, 'b', 'vid', 1]]).tolist() == [0, 1]


if __name__ == '__main__':
  for name, f in list(globals().items()):
    if name.startswith('test_'):
      f()
      print('ok', name)
