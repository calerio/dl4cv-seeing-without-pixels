"""Unit tests for experiment 4 (position noise), CPU only. Run from the code directory: python test_position_noise.py

Covers the noise itself (inputs and rotations untouched, no random draws when off, size), the fixed test-time
noise, and the dataset's __getitem__ with the real pose conversion: clean validation, noisy training.
No PyTorch weights or datasets needed.
"""
import ast
import importlib.util
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from utils.position_noise import evaluation_rng, perturb_positions, validate_noise
from utils.dataset_utils import transform_camera_pose


def poses(n=40):
  x = np.zeros((n, 7), dtype=np.float32)
  x[:, 0] = np.arange(n) * .02
  x[:, 6] = 1
  return x


class NoiseTests(unittest.TestCase):
  def test_no_mutation_or_rotation_changes(self):
    x = poses(); before = x.copy()
    y = perturb_positions(x, .01, np.random.default_rng(0))
    np.testing.assert_array_equal(x, before)
    np.testing.assert_array_equal(y[:, 3:], x[:, 3:])
    self.assertEqual(y.dtype, x.dtype)
    self.assertFalse(np.array_equal(y[:, :3], x[:, :3]))

  def test_disabled_does_not_consume_randomness(self):
    for sigma, prob in [(0, 1), (.01, 0)]:
      rng = np.random.default_rng(0); before = repr(rng.bit_generator.state)
      np.testing.assert_array_equal(perturb_positions(poses(), sigma, rng, prob), poses())
      self.assertEqual(before, repr(rng.bit_generator.state))

  def test_reproducible_independent_of_call_order(self):
    def draw(key, seed=7):
      return perturb_positions(poses(), .01, evaluation_rng(seed, key, range(40)))
    first = draw('clip_a'); draw('clip_b')
    np.testing.assert_array_equal(first, draw('clip_a'))
    self.assertFalse(np.array_equal(first, draw('clip_a', 8)))

  def test_noise_scale(self):
    x = poses(100000)
    y = perturb_positions(x, .02, np.random.default_rng(4))
    self.assertAlmostEqual(float((y[:, :3] - x[:, :3]).std()), .02, delta=.0003)

  def test_invalid_parameters_and_shape(self):
    for sigma in [-1, float('nan'), float('inf')]:
      with self.assertRaises(ValueError): validate_noise(sigma)
    for prob in [-.1, 1.1, float('nan')]:
      with self.assertRaises(ValueError): validate_noise(.1, prob)
    with self.assertRaises(ValueError): perturb_positions(np.zeros((4,12)), .1, np.random.default_rng())
    x = poses(); x[0, 0] = np.nan  # passes through, as on the clean path
    y = perturb_positions(x, .1, np.random.default_rng(0))
    self.assertTrue(np.isnan(y[0, 0])); self.assertTrue(np.isfinite(y[1:]).all())

  def dataset_get(self, mode, sigma):
    # Execute the real method with small in-memory data, bypassing file I/O and torch imports.
    path = ROOT / 'datasets/egoexo4d_pretrain.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__getitem__')
    ns = dict(np=np, random=random, torch=SimpleNamespace(initial_seed=lambda: 123),
              transform_camera_pose=transform_camera_pose, perturb_positions=perturb_positions,
              evaluation_rng=evaluation_rng)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
    raw = poses()
    obj = SimpleNamespace(mode=mode, encode_pose=11, sample_rate=1, gaze='none',
        args=SimpleNamespace(position_noise_train_sigma=sigma, position_noise_eval_sigma=sigma,
                             position_noise_probability=1, position_noise_eval_seed=42),
        df=pd.DataFrame([dict(save_id='take_0_2', take_name='take', t0_idx=3, t1_idx=10, description_text='C walks.')]),
        take_mapping={'take':0}, _load_cam_traj=lambda row:raw,
        window=lambda row:(np.arange(40), 0))
    return ns['__getitem__'](obj, 0), raw

  def test_actual_loader_noise_before_conversion(self):
    (clean, labels, scenario, _), raw = self.dataset_get('test', 0)
    (noisy, labels2, scenario2, _), raw2 = self.dataset_get('test', .01)
    self.assertEqual(clean.shape, (40,12))
    np.testing.assert_array_equal(raw, raw2)
    np.testing.assert_array_equal(clean[:,3:], noisy[:,3:])
    self.assertFalse(np.array_equal(clean[:,:3], noisy[:,:3]))
    self.assertEqual(labels, labels2); self.assertEqual(scenario, scenario2)
    np.testing.assert_array_equal(noisy, self.dataset_get('test', .01)[0][0])

  def test_validation_stays_clean(self):
    np.testing.assert_array_equal(self.dataset_get('val', .01)[0][0], self.dataset_get('val', 0)[0][0])

  def test_training_changes_positions(self):
    clean = self.dataset_get('train', 0)[0][0]
    noisy = self.dataset_get('train', .01)[0][0]
    self.assertFalse(np.array_equal(clean[:,:3], noisy[:,:3]))
    np.testing.assert_array_equal(clean[:,3:], noisy[:,3:])


if __name__ == '__main__':
  unittest.main()
