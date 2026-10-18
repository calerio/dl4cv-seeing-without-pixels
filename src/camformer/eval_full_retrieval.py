"""Full retrieval (each clip against all clips) from dumped features: recall at 1, 5 and 10.

Used for choices that must not be made on the test, e.g. the DynPose pose rate (docs/ego_exo.md, item 5),
on the validation clips that the test does not use (`train.py --test --exo_eval_split val_nontest`).

Usage:
    python eval_full_retrieval.py <feature_dir>
"""

import argparse

import torch
import torch.nn.functional as F


def recalls(feature_dir):
  frames = F.normalize(torch.load(f'{feature_dir}/frames.pt', map_location='cpu').float(), dim=-1)
  texts = F.normalize(torch.load(f'{feature_dir}/text.pt', map_location='cpu').float(), dim=-1)
  sim = frames @ texts.t()
  out = {}
  for name, s in (('m2t', sim), ('t2m', sim.t())):
    # rank of the matching item = how many items score strictly higher
    rank = (s > s.diag()[:, None]).sum(1)
    for k in (1, 5, 10):
      out[f'{name}_r{k}'] = 100 * (rank < k).float().mean().item()
  return len(frames), out


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('feature_dir')
  n, out = recalls(parser.parse_args().feature_dir)
  print(f'full retrieval over {n} clips: ' + ' | '.join(f'{k} {v:.2f}' for k, v in out.items()))
