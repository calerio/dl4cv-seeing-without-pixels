"""Unit checks for the resume guard in train.py (CPU, seconds). Run: python test_train_guards.py"""

import argparse
import json
import os
import tempfile

from train import check_resume


def run(log_dir, revision, epochs=20, resume=True):
  check_resume(argparse.Namespace(resume=resume, epochs=epochs), log_dir, revision)
  with open(os.path.join(log_dir, 'args.json'), 'w') as f:  # as train() does right after the check
    json.dump({'epochs': epochs}, f)


def refused(*a, **kw):
  try:
    run(*a, **kw)
  except SystemExit:
    return True
  return False


def test_resume_guard():
  with tempfile.TemporaryDirectory() as d:
    start = os.path.join(d, 'REVISION_AT_START')
    run(d, 'aaa')  # fresh start records the revision
    assert open(start).read().strip() == 'aaa'
    run(d, 'bbb')  # no checkpoint yet: a fresh start again, on new code, records the new revision
    assert open(start).read().strip() == 'bbb'
    os.makedirs(os.path.join(d, 'checkpoints'))
    open(os.path.join(d, 'checkpoints', 'last.ckpt'), 'w').close()
    run(d, 'bbb')  # resume on the same revision
    assert refused(d, 'ccc')  # resume on another revision
    assert refused(d, 'bbb', epochs=40)  # resume with another epoch count
    assert json.load(open(os.path.join(d, 'args.json')))['epochs'] == 20  # a refused resume rewrote nothing


def test_older_run_without_record():
  with tempfile.TemporaryDirectory() as d:
    os.makedirs(os.path.join(d, 'checkpoints'))
    open(os.path.join(d, 'checkpoints', 'last.ckpt'), 'w').close()
    run(d, 'aaa')  # an older run: its first checked resume records the revision
    assert open(os.path.join(d, 'REVISION_AT_START')).read().strip() == 'aaa'
    assert refused(d, 'bbb')


if __name__ == '__main__':
  for name, f in list(globals().items()):
    if name.startswith('test_'):
      f()
      print('ok', name)
