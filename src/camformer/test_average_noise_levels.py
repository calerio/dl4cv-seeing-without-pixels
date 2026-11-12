import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from average_noise_levels import average, level_csv  # noqa: E402


def test_file_name_matches_eval_run():
  assert level_csv('d', '0.005', 0).name == \
      'test_questions_position_noise_eval_sigma_0_005___position_noise_eval_seed_0.csv'


def test_equal_weight_mean(tmp_path):
  base = dict(question=[0, 1], take=['a', 'b'], type=['physical'] * 2, view=['in'] * 2)
  paths = []
  for i, m in enumerate([[1, 0], [1, 0], [0, 0], [0, 1]]):
    p = tmp_path / f'{i}.csv'
    pd.DataFrame(dict(base, m2t=m, t2m=m)).to_csv(p, index=False)
    paths.append(p)
  out = average(paths)
  assert out['m2t'].tolist() == [0.5, 0.25]
