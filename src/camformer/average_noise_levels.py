"""Experiment 4 headline input: per-question score averaged over the four noise levels (D7).

Usage:
    python average_noise_levels.py --run_dir <log dir of one run> --out <run>_meanlevels.csv [--noise_seed 0]

Reads test_questions_position_noise_eval_sigma_<level>_position_noise_eval_seed_<k>.csv for the levels
0.005, 0.01, 0.02 and 0.05 m and writes one CSV with the same question columns, where m2t and t2m are the mean of
the four levels, each level weighted equally (Hendrycks and Dietterich 2019 average over severities the same way).
The output is a valid input to compare_runs.py, whose accuracy is then the mean over levels and whose paired
bootstrap gives the interval of the main comparison:
    python compare_runs.py --a A_s0_meanlevels.csv A_s1_... --b N_s0_meanlevels.csv N_s1_...
Use with 2 or more seeds per setup. With one seed compare_runs.py prints a McNemar test, which counts only 0/1
scores and so ignores averaged questions; do not use it on these files.
"""

import argparse
from pathlib import Path

import pandas as pd

LEVELS = ['0.005', '0.01', '0.02', '0.05']
KEYS = ['question', 'take', 'type', 'view']


def level_csv(run_dir, level, seed):
  flags = f'--position_noise_eval_sigma {level} --position_noise_eval_seed {seed}'
  name = ''.join(c if c.isalnum() else '_' for c in flags).strip('_')  # as eval_run.sh
  return Path(run_dir) / f'test_questions_{name}.csv'


def average(paths):
  runs = [pd.read_csv(p) for p in paths]
  for p, r in zip(paths, runs):
    if not r[KEYS].equals(runs[0][KEYS]):
      raise ValueError(f'{p}: questions differ from {paths[0]}')
  out = runs[0][KEYS].copy()
  for m in ('m2t', 't2m'):
    out[m] = sum(r[m].astype(float) for r in runs) / len(runs)
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--run_dir', required=True)
  parser.add_argument('--out', required=True)
  parser.add_argument('--noise_seed', type=int, default=0)
  opts = parser.parse_args()
  paths = [level_csv(opts.run_dir, l, opts.noise_seed) for l in LEVELS]
  missing = [str(p) for p in paths if not p.exists()]
  if missing:
    raise SystemExit('missing: ' + ', '.join(missing))
  average(paths).to_csv(opts.out, index=False)


if __name__ == '__main__':
  main()
