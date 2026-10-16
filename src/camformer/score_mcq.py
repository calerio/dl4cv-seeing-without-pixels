"""Scores Ego-Exo4D 5-way MCQ features per activity type and visibility, as in Table 2 of the paper.

Usage:
    python score_mcq.py <feature_dir> [--out per_question.csv]

<feature_dir> holds frames.pt and text.pt written by `train.py --test --scenario all`, in the row order of
data_files/egoexo4d/test_alltasks_mcqv0.csv (groups of 5 rows; the first row of each group is the answer).
Physical = Basketball, Dance, Music, Soccer, Rock Climbing; procedural = Cooking, Health, Bike Repair.
In-view / out-of-view is the `ego_visible` flag of the answer row. The per-question CSV is the input for
paired comparisons between runs.
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F

PROCEDURAL = {'Cooking', 'Health', 'Bike Repair'}
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_files', 'egoexo4d')


def per_question(feature_dir, group_size=5):
  frames = F.normalize(torch.load(f'{feature_dir}/frames.pt', map_location='cpu').float(), dim=-1)
  texts = F.normalize(torch.load(f'{feature_dir}/text.pt', map_location='cpu').float(), dim=-1)
  df = pd.read_csv(os.path.join(DATA, 'test_alltasks_mcqv0.csv'))
  if len(df) != len(frames):
    raise ValueError(f'{len(frames)} features but {len(df)} test rows')
  scen = pd.read_csv(os.path.join(DATA, 'scenario_cls.csv')).set_index('take_name')['task']
  n = len(df) // group_size
  sim = torch.einsum(
      'gid,gjd->gij',
      frames.view(n, group_size, -1),
      texts.view(n, group_size, -1),
  )
  answers = df.iloc[::group_size].reset_index(drop=True)
  take = answers['save_id'].str.rsplit('_', n=2).str[0]
  out = pd.DataFrame({
      'question': range(n),
      'take': take,
      'scenario': take.map(scen),
      'm2t': (sim[:, 0, :].argmax(-1) == 0).numpy(),
      't2m': (sim[:, :, 0].argmax(-1) == 0).numpy(),
  })
  out['type'] = out['scenario'].map(lambda s: 'procedural' if s in PROCEDURAL else 'physical')
  out['view'] = answers['ego_visible'].map(lambda v: 'iv' if str(v) == 'True' else 'oov')
  return out


def table(q):
  rows = []
  for (t, v), g in q.groupby(['type', 'view']):
    rows.append((t, v, len(g), 100 * g.m2t.mean(), 100 * g.t2m.mean()))
  for t, g in q.groupby('type'):
    rows.append((t, 'all', len(g), 100 * g.m2t.mean(), 100 * g.t2m.mean()))
  rows.append(('all', 'all', len(q), 100 * q.m2t.mean(), 100 * q.t2m.mean()))
  return pd.DataFrame(rows, columns=['type', 'view', 'n', 'm2t', 't2m'])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('feature_dir')
  parser.add_argument('--out', default='')
  args = parser.parse_args()
  q = per_question(args.feature_dir)
  if q.scenario.isna().any():
    raise ValueError(f'{q.scenario.isna().sum()} questions have no scenario label')
  print(table(q).to_string(index=False, float_format='%.2f'))
  if args.out:
    q.to_csv(args.out, index=False)
