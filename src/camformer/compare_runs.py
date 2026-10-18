"""Paired comparison of two training setups on the Ego-Exo4D 5-way MCQ test.

Usage:
    python compare_runs.py --a A_s0.csv [A_s1.csv ...] --b B_s0.csv [B_s1.csv ...] [--metric m2t]

Each CSV is the per-question output of score_mcq.py --out, one per training seed. For each split (all,
physical, procedural, and type x view) the script prints each setup's accuracy (mean over its seeds) and
per-seed spread, the difference B - A, and a 95% interval from a paired two-level bootstrap: each resample
draws the seeds of each setup with replacement, then the test takes with replacement. Takes, not questions,
are resampled because all five options of a question, and many questions, come from the same take.

What the interval covers: test-take sampling, and training-seed variation only as far as the seeds given
can show it. With one seed per setup it covers test sampling alone and says nothing about whether the
difference would survive retraining; report it as such. With one seed per setup the script also prints an
exact McNemar test on the discordant questions, which ignores the clustering by take and is optimistic.
"""

import argparse
import math

import numpy as np
import pandas as pd

KEYS = ['question', 'take', 'type', 'view']


def load(paths, metric):
  runs = [pd.read_csv(p) for p in paths]
  for r in runs:  # the DynPose files (score_exo_mcq.py) have no type or view
    for k in ('type', 'view'):
      if k not in r:
        r[k] = 'all'
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


def two_level_bootstrap(takes, sa, sb, n_boot, rng):
  """95% interval of mean(B) - mean(A), resampling seeds of each setup, then takes, with replacement."""
  codes, uniq = pd.factorize(takes)
  counts = np.bincount(codes).astype(float)
  per_take = lambda s: np.stack([np.bincount(codes, weights=r, minlength=len(uniq)) for r in s])
  ta, tb = per_take(sa), per_take(sb)  # (seeds, takes): correct answers per take and seed
  t_idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
  a_idx = rng.integers(0, len(sa), size=(n_boot, len(sa)))
  b_idx = rng.integers(0, len(sb), size=(n_boot, len(sb)))
  a = ta[a_idx].mean(1)  # (n_boot, takes)
  b = tb[b_idx].mean(1)
  rows = np.arange(n_boot)[:, None]
  diff = (b[rows, t_idx] - a[rows, t_idx]).sum(1) / counts[t_idx].sum(1)
  return np.percentile(diff, [2.5, 97.5])


def seed_range(s):
  acc = 100 * s.mean(1)
  return f'{acc.min():.2f}-{acc.max():.2f}' if len(acc) > 1 else '-'


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
  for t in sorted(set(qa['type']) - {'all'}):
    splits.append((t, 'all', (qa['type'] == t).to_numpy()))
    for v in sorted(qa['view'].unique()):
      splits.append((t, v, ((qa['type'] == t) & (qa['view'] == v)).to_numpy()))

  print(f'metric {opts.metric} | A: {len(opts.a)} seed(s) | B: {len(opts.b)} seed(s) | '
        f'bootstrap over seeds and takes, {opts.n_boot} resamples')
  if len(opts.a) == 1 or len(opts.b) == 1:
    print('note: a setup with one seed adds no training-seed variation; the interval covers test sampling only')
  elif min(len(opts.a), len(opts.b)) < 5:
    print('note: with fewer than 5 seeds the seed level of the interval is rough and tends to be too narrow;'
          ' report the per-seed ranges next to it')
  rows = []
  for t, v, m in splits:
    lo, hi = two_level_bootstrap(qa['take'][m].to_numpy(), sa[:, m], sb[:, m], opts.n_boot, rng)
    row = {'type': t, 'view': v, 'n': int(m.sum()), 'takes': qa['take'][m].nunique(),
           'A': 100 * a[m].mean(), 'A_seeds': seed_range(sa[:, m]),
           'B': 100 * b[m].mean(), 'B_seeds': seed_range(sb[:, m]),
           'B-A': 100 * (b[m] - a[m]).mean(), 'ci_lo': 100 * lo, 'ci_hi': 100 * hi}
    if single:
      p, b_only, a_only = mcnemar_exact(sa[0][m], sb[0][m])
      row.update(A_only=a_only, B_only=b_only, mcnemar_p=p)
    rows.append(row)
  print(pd.DataFrame(rows).to_string(index=False, float_format='%.2f'))


if __name__ == '__main__':
  main()
