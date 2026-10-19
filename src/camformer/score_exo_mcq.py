"""Scores DynPose-100K 5-way MCQ features, with one row per question for paired comparisons.

Usage:
    python score_exo_mcq.py <feature_dir> [--out per_question.csv]

<feature_dir> holds frames.pt and text.pt written by `train.py --test` on the DynPose test, in the row order
of data_files/dynpose100k/metadata_samevideo_test5000.csv (groups of 5 rows; the first row of each group is
the answer). The per-question file has the columns compare_runs.py reads; the answer's video id takes the
place of the take, so the bootstrap resamples videos.
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F

CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data_files', 'dynpose100k', 'metadata_samevideo_test5000.csv'
)


def per_question(feature_dir, group_size=5):
  frames = F.normalize(torch.load(f'{feature_dir}/frames.pt', map_location='cpu').float(), dim=-1)
  texts = F.normalize(torch.load(f'{feature_dir}/text.pt', map_location='cpu').float(), dim=-1)
  df = pd.read_csv(CSV)
  if len(df) != len(frames):
    raise ValueError(f'{len(frames)} features but {len(df)} test rows')
  n = len(df) // group_size
  answers = df.iloc[::group_size].reset_index(drop=True)
  if not answers['gt'].astype(bool).all():
    raise ValueError('the first row of some group is not the answer')
  sim = torch.einsum('gid,gjd->gij', frames.view(n, group_size, -1), texts.view(n, group_size, -1))
  return pd.DataFrame({
      'question': range(n),
      'take': answers['video_uid'],
      'm2t': (sim[:, 0, :].argmax(-1) == 0).numpy(),
      't2m': (sim[:, :, 0].argmax(-1) == 0).numpy(),
  })


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('feature_dir')
  parser.add_argument('--out', default='')
  args = parser.parse_args()
  q = per_question(args.feature_dir)
  print(f'DynPose 5-way test: {len(q)} questions | m2t {100 * q.m2t.mean():.2f} | t2m {100 * q.t2m.mean():.2f}')
  if args.out:
    q.to_csv(args.out, index=False)
