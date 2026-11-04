"""Unit checks for the experiment-1 code and the dropped-layer guard, on synthetic data (CPU, seconds).
Run: python test_hard_negatives.py"""

import random

import torch
import torch.nn.functional as F

from loader import SameTakeBatchSampler
from models.cm_encoder import CameraPoseSeqEncoder
from tasks.pretrain import CLIPLoss, check_no_dropped_layers, primary_verb, within_take_loss


def test_same_take_sampler():
  random.seed(0)
  takes = [f't{i // 20}' for i in range(200)]  # 10 takes of 20 clips, 1 s apart
  times = [float(i % 20) for i in range(200)]
  s = SameTakeBatchSampler(takes, times, batch_size=32, k=8, min_gap=4.0)
  used = []  # record the groups this very epoch was built from
  build = s._groups
  s._groups = lambda: used.append(build()) or used[-1]
  batches = list(iter(s))
  assert len(used) == 1 and len(batches) == len(s) == 200 // 32
  seen = [i for b in batches for i in b]
  assert len(seen) == len(set(seen)) and all(len(b) == 32 for b in batches)
  assert seen == [i for g in used[0] for i in g][: len(seen)]  # batches are the groups, packed in order
  # each group: at most k clips of one take, at least min_gap apart
  for g in used[0]:
    assert len(g) <= 8 and len({takes[i] for i in g}) == 1
    assert all(abs(times[a] - times[b]) >= 4.0 for a in g for b in g if a != b)


def test_within_take_loss():
  torch.manual_seed(0)
  m, t = torch.randn(4, 8), torch.randn(4, 8)
  # clips 0-2 from one take (verbs a, b, a), clip 3 alone: 3 has no competitor, 0 and 2 compete only with 1
  loss, rows = within_take_loss(m, t, ['x', 'x', 'x', 'y'], ['a', 'b', 'a', 'c'])
  assert rows == 3
  mn, tn = F.normalize(m, dim=-1), F.normalize(t, dim=-1)
  logits = mn @ tn.t() / 0.07
  allowed = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=torch.bool)
  sub = logits[:3, :3].masked_fill(~allowed, float('-inf'))
  ref = (F.cross_entropy(sub, torch.arange(3)) + F.cross_entropy(sub.t(), torch.arange(3))) / 2
  assert torch.allclose(loss, ref)
  zero, n = within_take_loss(m, t, ['x', 'y', 'z', 'w'], ['a', 'b', 'c', 'd'])
  assert n == 0 and zero.item() == 0.0


def test_clip_loss_mask():
  torch.manual_seed(0)
  m, t = torch.randn(3, 8), torch.randn(3, 8)
  plain = CLIPLoss()(m, t)
  masked = CLIPLoss()(m, t, texts=['cut', 'stir', 'cut'])  # 0 and 2 share a key: not negatives of each other
  logits = F.normalize(m, dim=-1) @ F.normalize(t, dim=-1).t() / 0.07
  logits[0, 2] = logits[2, 0] = float('-inf')
  ref = (F.cross_entropy(logits, torch.arange(3)) + F.cross_entropy(logits.t(), torch.arange(3))) / 2
  assert torch.allclose(masked, ref) and not torch.allclose(masked, plain)
  assert torch.allclose(CLIPLoss()(m, t, texts=['a', 'b', 'c']), plain)


def test_primary_verb():
  assert primary_verb('C is picking the spoon.') == primary_verb('C picks up a spoon') == 'pick'
  assert primary_verb('C cuts the onion') != primary_verb('C stirs the pot')


def test_dropped_layer_guard():
  torch.manual_seed(0)
  trained = CameraPoseSeqEncoder(encode_pose=11, output_dim=512, pooling='no', gaze_input=True).state_dict()
  plain = CameraPoseSeqEncoder(encode_pose=11, output_dim=512, pooling='no')
  result = plain.load_state_dict(trained, strict=False)
  assert any(k.startswith('gaze_proj.') for k in result.unexpected_keys)
  try:
    check_no_dropped_layers(result.unexpected_keys, 'ckpt', 'hint')
  except RuntimeError as e:
    assert 'gaze_proj.' in str(e) and 'hint' in str(e)
  else:
    raise AssertionError('a trained gaze layer was dropped without an error')
  check_no_dropped_layers(['some_other_head.weight'], 'ckpt', 'hint')  # other extra keys only warn


if __name__ == '__main__':
  for name, f in list(globals().items()):
    if name.startswith('test_'):
      f()
      print('ok', name)
