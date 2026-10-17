"""Recomputes the data counts stated in docs/ego_exo.md from the released CSVs.

Usage:
    python scripts/ego_exo_facts.py <data_files dir>

<data_files dir> is the upstream repository's data_files/ folder (egoexo4d/ and dynpose100k/). Light: reads
CSVs only. The scale filter is the one in datasets/dynpose.py (scale < 20).
"""

import os
import sys

import pandas as pd

SCALE_THRESHOLD = 20


def main(root):
  ego = os.path.join(root, 'egoexo4d')
  dyn = os.path.join(root, 'dynpose100k')
  ego_train = pd.read_csv(os.path.join(ego, 'train_v2.csv'))
  ego_test = pd.read_csv(os.path.join(ego, 'test_alltasks_mcqv0.csv'))
  train = pd.read_csv(os.path.join(dyn, 'metadata_train_v1.csv'))
  val = pd.read_csv(os.path.join(dyn, 'metadata_val_v1.csv'))
  test = pd.read_csv(os.path.join(dyn, 'metadata_samevideo_test5000.csv'))

  print(f'Ego-Exo4D training pairs: {len(ego_train):,}')
  print(f'Ego-Exo4D test questions: {len(ego_test) // 5:,}')

  vipe = train[train['vipe_pose_path'].notnull() & (train['vipe_pose_scale'] < SCALE_THRESHOLD)]
  orig = train[train['pose_scale'] < SCALE_THRESHOLD]
  print(f'DynPose training clips after the scale filter: ViPE {len(vipe):,}, original {len(orig):,}')

  # Test rows come in groups of 5; `gt` marks the answer row of each group.
  test = test.assign(question=test.index // 5)
  print(f'DynPose test questions: {test["question"].nunique():,} ({len(test):,} rows)')
  uids = set(test['uid'])
  print(f'  test clips found in val: {len(uids & set(val["uid"])):,} of {len(uids):,};'
        f' in train: {len(uids & set(train["uid"])):,}')
  videos = set(test['video_uid'])
  print(f'  test videos: {len(videos):,}; also with segments in train: {len(videos & set(train["video_uid"])):,}')
  answers = test[test['gt'].astype(bool)].set_index('question')['video_uid']
  same = test[~test['gt'].astype(bool)].assign(answer_video=lambda d: d['question'].map(answers))
  n_same = same[same['video_uid'] == same['answer_video']]['question'].nunique()
  print(f'  questions with a distractor from the answer\'s video: {n_same:,}')
  print(f'DynPose validation clips not in the test set: {len(set(val["uid"]) - uids):,}')


if __name__ == '__main__':
  main(sys.argv[1])
