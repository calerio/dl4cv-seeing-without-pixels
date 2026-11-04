"""Exits with an error unless a run's last checkpoint is from the last epoch the run was set to train.

Usage:  python check_finished.py <run dir>
Used by the test scripts so that a run still in its job chain, or one that crashed, is never scored as final.
train.py refuses a resume that changes --epochs, so args.json's count is the one the run started with.
"""

import json
import os
import sys

import torch

run = sys.argv[1]
epochs = json.load(open(os.path.join(run, 'args.json')))['epochs']
ckpt = os.path.join(run, 'checkpoints', 'last.ckpt')
epoch = torch.load(ckpt, map_location='cpu', weights_only=False, mmap=True)['epoch']
if epoch + 1 != epochs:
  sys.exit(f'{ckpt}: checkpoint of epoch {epoch + 1} of {epochs}; training has not finished')
