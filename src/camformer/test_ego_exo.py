"""Unit checks for the experiment-2 code, on synthetic data (CPU, seconds). Run: python test_ego_exo.py"""

import random

import numpy as np
from scipy.spatial.transform import Rotation
import torch

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
