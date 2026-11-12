"""Experiment 4: position-only perturbations before relative-pose conversion.

Sigma is in metres: Aria poses store positions in metres (checked on the stored files, docs/position_noise.md).
"""
import hashlib
import json
import math
import numpy as np


def validate_noise(sigma, probability=1.0):
  if not math.isfinite(sigma) or sigma < 0:
    raise ValueError('position noise sigma must be finite and nonnegative')
  if not math.isfinite(probability) or not 0 <= probability <= 1:
    raise ValueError('position noise probability must be in [0, 1]')


def evaluation_rng(seed, save_id, frame_indices):
  """Stable per-window RNG: independent of worker count, batching and Python hash."""
  payload = json.dumps([int(seed), str(save_id), [int(i) for i in frame_indices]], separators=(',', ':'))
  digest = hashlib.sha256(payload.encode('utf-8')).digest()
  return np.random.default_rng(int.from_bytes(digest[:16], 'little'))


def perturb_positions(poses, sigma, rng, probability=1.0):
  """Copy (T,7) world poses and add independent N(0,sigma²) position errors.

  No padding is present at this point. Quaternions, shape, dtype and caller input
  are preserved. Zero sigma/probability consumes no RNG, preserving defaults.
  Non-finite poses pass through as they would on the clean path (noise added to
  NaN stays NaN), so a noisy run never stops on a clip the clean run trains on.
  """
  validate_noise(sigma, probability)
  poses = np.asarray(poses)
  if poses.ndim != 2 or poses.shape[1] != 7 or not np.issubdtype(poses.dtype, np.floating):
    raise ValueError('expected floating (T,7) raw position/quaternion poses')
  result = poses.copy()
  if sigma == 0 or probability == 0:
    return result
  if probability < 1 and rng.random() >= probability:
    return result
  result[:, :3] += rng.normal(0, sigma, size=(len(poses), 3)).astype(poses.dtype)
  return result
