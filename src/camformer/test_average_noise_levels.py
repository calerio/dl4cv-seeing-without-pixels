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


def test_t2m_and_mismatch(tmp_path):
  import pytest
  base = dict(question=[0, 1], take=['a', 'b'], type=['physical'] * 2, view=['in'] * 2)
  p1, p2, p3 = tmp_path / '1.csv', tmp_path / '2.csv', tmp_path / '3.csv'
  pd.DataFrame(dict(base, m2t=[1, 1], t2m=[1, 0])).to_csv(p1, index=False)
  pd.DataFrame(dict(base, m2t=[0, 0], t2m=[0, 0])).to_csv(p2, index=False)
  assert average([p1, p2])['t2m'].tolist() == [0.5, 0.0]
  pd.DataFrame(dict(base, question=[0, 2], m2t=[0, 0], t2m=[0, 0])).to_csv(p3, index=False)
  with pytest.raises(ValueError):
    average([p1, p3])
