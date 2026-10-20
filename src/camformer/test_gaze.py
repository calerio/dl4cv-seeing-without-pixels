"""Unit tests for experiment 3 (gaze), CPU only. Run from the code directory: python test_gaze.py

1. The zero-initialised gaze layer changes nothing.
2. Gaze frame k lands on pose token k, and the dataset gives pose frame k the gaze at that frame's time.
3. The committed donor map: every clip has a donor; same-take donors are a different, non-overlapping window;
   cross-take donors are another take of the same scenario.
4. The collate pads gaze exactly like the poses.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import torch

from datasets import gaze
from datasets.egoexo4d_pretrain import _resolve_csv
from loader import longseq_pose_text_collate
from models.cm_encoder import CameraPoseSeqEncoder


def test_zero_init_identity():
  torch.manual_seed(0)
  base = CameraPoseSeqEncoder(encode_pose=11, output_dim=512, pooling='no').eval()
  withg = CameraPoseSeqEncoder(encode_pose=11, output_dim=512, pooling='no', gaze_input=True).eval()
  missing = withg.load_state_dict(base.state_dict(), strict=False).missing_keys
  assert sorted(missing) == ['gaze_proj.bias', 'gaze_proj.weight'], missing
  x = torch.randn(3, 50, 12)
  pad = torch.zeros(3, 50, dtype=torch.bool)
  lengths = torch.full((3,), 50)
  g = torch.randn(3, 50, 4)
  with torch.no_grad():
    assert torch.equal(base(x, pad, lengths), withg(x, pad, lengths, gaze=g))
  print('ok zero-init identity')


def test_token_alignment():
  enc = CameraPoseSeqEncoder(encode_pose=11, output_dim=512, pooling='no', gaze_input=True)
  torch.nn.init.normal_(enc.gaze_proj.weight)
  tokens = torch.randn(2, 30, 128)
  g = torch.zeros(2, 30, 4)
  g[:, 17] = 1.0
  diff = (enc.add_gaze(tokens, g) - enc.add_gaze(tokens, torch.zeros_like(g))).abs().sum(-1)
  assert (diff[:, 17] > 0).all() and (diff[:, :17] == 0).all() and (diff[:, 18:] == 0).all()
  print('ok gaze frame k -> pose token k')


def test_dataset_time_alignment():
  # Synthetic gaze whose yaw equals its time: the feature of pose frame k must be that frame's time.
  with tempfile.TemporaryDirectory() as d:
    t = np.arange(0, 100, 0.1)
    np.savez(os.path.join(d, 'take_a.npz'), t=t, yaw=t.astype(np.float32), pitch=np.zeros_like(t, np.float32),
             log_depth=np.zeros_like(t, np.float32))
    old = gaze.GAZE_DIR
    gaze.GAZE_DIR = d
    gaze.load_take.cache_clear()
    try:
      row = pd.Series({'start_time': 12.34, 'take_name': 'take_a', 'save_id': 'x'})
      frame_idx = np.arange(7, 167, 2)  # random_start 7, sample_rate 2
      f = gaze.clip_gaze('real', row, frame_idx, 0.05)
      expect = 12.34 + frame_idx * 0.05
      assert np.allclose(f[:, 0], expect, atol=1e-4), (f[:3, 0], expect[:3])
      assert (f[:, 3] == 1).all()
      far = gaze.features_at([200.0], 'take_a')
      assert (far == 0).all()
    finally:
      gaze.GAZE_DIR = old
      gaze.load_take.cache_clear()
  print('ok pose frame k gets gaze at start_time + frame_idx[k] * frame_dt')


def test_donor_map():
  if not os.path.exists(gaze.DONOR_FILE):
    print('skip donor map (not built yet)')
    return
  donors = pd.read_csv(gaze.DONOR_FILE, keep_default_na=False)
  win = pd.concat([pd.read_csv(_resolve_csv(f))[['save_id', 'start_time', 'end_time']]
                   for f in ('train_v2.csv', 'val_v2.csv', 'test_alltasks_mcqv0.csv')]).drop_duplicates('save_id')
  win['take'] = win['save_id'].map(lambda s: '_'.join(s.split('_')[:-2]))
  scen = pd.read_csv(_resolve_csv('scenario_cls.csv')).set_index('take_name')['task']
  assert set(donors['save_id']) == set(win['save_id'])
  d = donors[donors['donor_rule'] != 'none'].merge(win, on='save_id').merge(
      win.add_prefix('d_'), left_on='donor_save_id', right_on='d_save_id')
  st, ss = d[d['donor_rule'] == 'same_take'], d[d['donor_rule'] == 'same_scenario']
  # Same-take donors: same take, different window, at least one window away (no overlap in time).
  assert (st['take'] == st['d_take']).all() and (st['save_id'] != st['donor_save_id']).all()
  assert not ((st['d_start_time'] < st['end_time']) & (st['d_end_time'] > st['start_time'])).any()
  # Cross-take donors: different take, same scenario, only where the take had no same-take donor.
  assert (ss['take'] != ss['d_take']).all()
  assert (ss['take'].map(scen) == ss['d_take'].map(scen)).all()
  assert (d['donor_take'] == d['d_take']).all()
  # Every clip has a donor.
  none = (donors['donor_rule'] == 'none').sum()
  assert none == 0, f'{none} clips without a donor'
  # Training, testing and diagnostics all read the same file through gaze.load_donors().
  m = gaze.load_donors()
  assert m is gaze.load_donors() and len(m) == len(donors)
  print(f'ok donor map: {len(st)} same-take and {len(ss)} same-scenario donors, every clip has one')


def test_donor_rules_synthetic():
  w = pd.DataFrame({'save_id': ['a1', 'a2', 'b1', 'c1'], 'take_name': ['A', 'A', 'B', 'C'],
                    'start_time': [0.0, 20.0, 0.0, 0.0], 'end_time': [10.0, 30.0, 10.0, 10.0]})
  d = gaze.build_donor_map(w, {'A', 'B', 'C'}, {'A': 'Cooking', 'B': 'Cooking', 'C': 'Soccer'}).set_index('save_id')
  assert d.loc['a1', 'donor_save_id'] == 'a2' and d.loc['a1', 'donor_rule'] == 'same_take'
  assert d.loc['b1', 'donor_rule'] == 'same_scenario' and d.loc['b1', 'donor_take'] == 'A'
  assert d.loc['c1', 'donor_rule'] == 'none'  # only take of its scenario
  print('ok donor rules on a synthetic map')


def test_collate():
  seqs = [np.random.randn(n, 12) for n in (5, 9)]
  extras = [{'gaze': np.random.randn(n, 4).astype(np.float32)} for n in (5, 9)]
  batch = [(s, [0, 1, 'c', 't'], 0, e) for s, e in zip(seqs, extras)]
  poses, pad, _, lengths, _, extra = longseq_pose_text_collate(batch)
  assert extra['gaze'].shape == (2, 9, 4) and poses.shape[:2] == (2, 9)
  assert torch.equal(extra['gaze'][0, :5], torch.as_tensor(extras[0]['gaze']))
  assert (extra['gaze'][0, 5:] == 0).all()
  print('ok collate')


if __name__ == '__main__':
  test_zero_init_identity()
  test_token_alignment()
  test_dataset_time_alignment()
  test_donor_map()
  test_donor_rules_synthetic()
  test_collate()
  print('all gaze tests passed')
