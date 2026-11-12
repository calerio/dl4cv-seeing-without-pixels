"""Evaluate one completed experiment-4 run at fixed noise levels.

Run sequentially; do not launch duplicate evaluations of the same run concurrently.
Uses the repository's unmodified five-way scorer. Does not select models/settings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('run_dir', type=Path)
  p.add_argument('--sigmas', type=float, nargs='+', required=True)
  p.add_argument('--noise-seeds', type=int, nargs='+', default=[0, 1, 2])
  p.add_argument('--take-duration', type=float, default=8)
  a = p.parse_args()
  from utils.position_noise import validate_noise
  for sigma in a.sigmas:
    validate_noise(sigma)
  if not a.noise_seeds or a.take_duration <= 0:
    p.error('need noise seeds and positive duration')
  root = Path(__file__).resolve().parent
  run = a.run_dir.expanduser().resolve()
  args = json.loads((run / 'args.json').read_text())
  if args.get('dataset') != 'egoexo4d_pretrain_longseq' or args.get('gaze', 'none') != 'none' or args.get('use_pi3_pose', False):
    p.error('expected a trajectory-only Aria Ego-Exo4D run')
  if args.get('take_duration') != a.take_duration:
    p.error('evaluation duration must match the planned training duration')
  subprocess.run([sys.executable, str(root / 'check_finished.py'), str(run)], check=True)
  out = run / 'position_noise_evaluation'
  out.mkdir(exist_ok=True)
  # Unique checkpoint basename gives this run its own feature directory.
  ref = out / (hashlib.sha256(str(run).encode()).hexdigest()[:16] + '.ckpt')
  if not ref.exists():
    ref.symlink_to(run / 'checkpoints/last.ckpt')
  from score_mcq import per_question, table
  for sigma in dict.fromkeys(a.sigmas):
    for seed in (a.noise_seeds if sigma else [a.noise_seeds[0]]):
      tag = f'sigma{sigma!r}_seed{seed}'
      cmd = [sys.executable, str(root / 'train.py'), '--test', '--dataset', 'egoexo4d_pretrain_longseq',
             '--scenario', 'all', '--pose_encoding', 'rel9d_grav', '--sample_dur',
             '--take_duration', str(a.take_duration), '--test_take_duration', str(a.take_duration),
             '--num_gpus', '1', '--num_workers', '6', '--batch_size', '1000',
             '--ckpt', str(ref), '--job_name', f'noise_eval_{ref.stem}_{tag}',
             '--position_noise_eval_sigma', str(sigma), '--position_noise_eval_seed', str(seed), '--seed', '0']
      # Stream progress and retain exact command/log for audit.
      with (out / f'{tag}.log').open('w') as log:
        proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        feature_dir = None
        for line in proc.stdout:
          print(line, end='', flush=True)
          log.write(line)
          match = re.match(r'^Saved features.* to (final_data/.*)\s*$', line)
          if match:
            feature_dir = root / match.group(1).strip()
        if proc.wait() != 0 or feature_dir is None:
          raise RuntimeError(f'evaluation failed; see {out / (tag + ".log")}')
      q = per_question(str(feature_dir))
      if q.scenario.isna().any():
        raise ValueError('test questions missing scenario mapping')
      q.to_csv(out / f'{tag}_questions.csv', index=False)
      table(q).to_csv(out / f'{tag}_summary.csv', index=False)
      manifest = dict(command=cmd, sigma=sigma, noise_seed=seed, training_args=args,
                      revision=(root / 'REVISION').read_text().strip() if (root / 'REVISION').exists() else 'unknown',
                      checkpoint=str(ref.resolve()), checkpoint_size=ref.stat().st_size,
                      feature_dir=str(feature_dir), note='sigma uses raw position units; synthetic iid position noise')
      (out / f'{tag}_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
      print(table(q).to_string(index=False))


if __name__ == '__main__':
  main()
