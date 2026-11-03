"""Scores Nymeria zero-shot 5-way MCQ features, with one row per question for paired comparisons.

Usage:
    python score_nymeria.py <column> <feature_dir> [--out per_question.csv]

<column> is the narration type (a body posture, b hands/arms, c legs/feet, d focus of attention), as given to
`train.py --dataset nymeria_pretrain --test --text_column <column>`. <feature_dir> holds the frames.pt and
text.pt that run wrote, in the row order of data_files/nymeria/eval1000/split_by_<narration>.csv (groups of 5
rows; the first row of each group is the answer). The answer's recording (motion_file) takes the place of the
take, so compare_runs.py resamples recordings; the narration type goes in the `type` column.
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F

from datasets.nymeria import text_column_mapping, _resolve_eval_csv


def per_question(column, feature_dir, group_size=5):
  frames = F.normalize(torch.load(f'{feature_dir}/frames.pt', map_location='cpu').float(), dim=-1)
  texts = F.normalize(torch.load(f'{feature_dir}/text.pt', map_location='cpu').float(), dim=-1)
  name = text_column_mapping[column]
  df = pd.read_csv(_resolve_eval_csv(f"split_by_{name.replace('/', '_').replace(' ', '_')}.csv"))
  if len(df) != len(frames):
    raise ValueError(f'{len(frames)} features but {len(df)} test rows')
  n = len(df) // group_size
  answers = df.iloc[::group_size].reset_index(drop=True)
  if not answers['gt'].astype(bool).all():
    raise ValueError('the first row of some group is not the answer')
  if (df.groupby(df.index // group_size)['motion_file'].nunique() != 1).any():
    raise ValueError('some question mixes recordings')
  sim = torch.einsum('gid,gjd->gij', frames.view(n, group_size, -1), texts.view(n, group_size, -1))
  return pd.DataFrame({
      'question': [f'{column}{i}' for i in range(n)],
      'take': answers['motion_file'],
      'type': column,
      'view': 'all',
      'm2t': (sim[:, 0, :].argmax(-1) == 0).numpy(),
      't2m': (sim[:, :, 0].argmax(-1) == 0).numpy(),
  })


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('column', choices=sorted(text_column_mapping))
  parser.add_argument('feature_dir')
  parser.add_argument('--out', default='')
  args = parser.parse_args()
  q = per_question(args.column, args.feature_dir)
  print(f'Nymeria {args.column}: {len(q)} questions | m2t {100 * q.m2t.mean():.2f} | t2m {100 * q.t2m.mean():.2f}')
  if args.out:
    q.to_csv(args.out, index=False)
