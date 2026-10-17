"""Paired comparison of two training setups on the Ego-Exo4D 5-way MCQ test.

Usage:
    python compare_runs.py --a A_s0.csv [A_s1.csv ...] --b B_s0.csv [B_s1.csv ...] [--metric m2t]

Each CSV is the per-question output of score_mcq.py --out. With several seeds per setup, each question's
score is its mean over that setup's seeds. For each split (all, physical, procedural, and type x view) the
script prints the two accuracies, the difference B - A, and a 95% confidence interval from a paired
bootstrap that resamples takes, not questions: all five options of a question, and many questions, come
from the same take, so questions are not independent. With one seed per setup it also prints an exact
McNemar test on the discordant questions. McNemar ignores that clustering and so is too optimistic; the
bootstrap interval is the number to report.
"""

import argparse
import math

import numpy as np
import pandas as pd

KEYS = ['question', 'take', 'type', 'view']


def load(paths, metric):
  runs = [pd.read_csv(p) for p in paths]
  base = runs[0][KEYS]
  for p, r in zip(paths, runs):
    if not r[KEYS].equals(base):
      raise ValueError(f'{p}: questions differ from {paths[0]}')
  scores = np.stack([r[metric].astype(float).to_numpy() for r in runs])
  return base, scores


def mcnemar_exact(a, b):
  """Two-sided exact McNemar p-value from 0/1 correctness vectors."""
  n01 = int(((a == 0) & (b == 1)).sum())
  n10 = int(((a == 1) & (b == 0)).sum())
  n = n01 + n10
  if n == 0:
    return 1.0, n01, n10
  k = min(n01, n10)
  p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n
  return min(1.0, p), n01, n10


def take_bootstrap(takes, diff, n_boot, rng):
  """95% interval of mean(diff) resampling takes with replacement."""
  codes, uniq = pd.factorize(takes)
  sums = np.bincount(codes, weights=diff)
  counts = np.bincount(codes).astype(float)
  idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
  means = sums[idx].sum(1) / counts[idx].sum(1)
  return np.percentile(means, [2.5, 97.5])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--a', nargs='+', required=True)
  parser.add_argument('--b', nargs='+', required=True)
  parser.add_argument('--metric', default='m2t', choices=['m2t', 't2m'])
  parser.add_argument('--n_boot', type=int, default=10000)
  parser.add_argument('--seed', type=int, default=0)
  opts = parser.parse_args()

  qa, sa = load(opts.a, opts.metric)
  qb, sb = load(opts.b, opts.metric)
  if not qa.equals(qb):
    raise ValueError('setups A and B were scored on different questions')
  a, b = sa.mean(0), sb.mean(0)
  single = len(opts.a) == 1 and len(opts.b) == 1
  rng = np.random.default_rng(opts.seed)

  splits = [('all', 'all', np.ones(len(qa), bool))]
  for t in sorted(qa['type'].unique()):
    splits.append((t, 'all', (qa['type'] == t).to_numpy()))
    for v in sorted(qa['view'].unique()):
      splits.append((t, v, ((qa['type'] == t) & (qa['view'] == v)).to_numpy()))

  print(f'metric {opts.metric} | A: {len(opts.a)} seed(s) | B: {len(opts.b)} seed(s) | '
        f'bootstrap over takes, {opts.n_boot} resamples')
  rows = []
  for t, v, m in splits:
    d = b[m] - a[m]
    lo, hi = take_bootstrap(qa['take'][m].to_numpy(), d, opts.n_boot, rng)
    row = {'type': t, 'view': v, 'n': int(m.sum()), 'takes': qa['take'][m].nunique(),
           'A': 100 * a[m].mean(), 'B': 100 * b[m].mean(), 'B-A': 100 * d.mean(),
           'ci_lo': 100 * lo, 'ci_hi': 100 * hi}
    if single:
      p, b_only, a_only = mcnemar_exact(sa[0][m], sb[0][m])
      row.update(A_only=a_only, B_only=b_only, mcnemar_p=p)
    rows.append(row)
  print(pd.DataFrame(rows).to_string(index=False, float_format='%.2f'))


if __name__ == '__main__':
  main()
