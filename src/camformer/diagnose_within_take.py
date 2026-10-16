"""Diagnostics for the same-take hard-negatives experiment (no training).

For one CamFormer checkpoint on the Ego-Exo4D training split, measures:
  1. On training batches (random, and same-take with K clips per take): the authors'
     loss, the within-take loss, and the share of each anchor's softmax mass that
     falls on same-take negatives with a different verb (motion -> text).
  2. A within-take 5-way multiple choice built from training takes with the paper's
     rule (four distractors from the same take, with a different verb from the
     answer), scored like the test.

Usage (one GPU, from the code directory):
    python diagnose_within_take.py <checkpoint> [--batches 20] [--questions 5000]
"""

import argparse
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

from loader import SameTakeBatchSampler, longseq_pose_text_collate
from tasks.pretrain import primary_verb, within_take_loss
from tasks.pretrain_longseq import CameraPoseTextPretrainingLongSeq
import train as train_script


def build(ckpt):
  sys.argv = [
      'train.py', '--dataset', 'egoexo4d_pretrain_longseq', '--pose_encoding', 'rel9d_grav',
      '--take_duration', '8', '--sample_dur', '--num_gpus', '1', '--init_ckpt', ckpt,
  ]
  args = train_script.get_args()
  task = CameraPoseTextPretrainingLongSeq(args).cuda().eval()
  return args, task


@torch.no_grad()
def encode(task, batch):
  poses, pad_mask, text_list, lengths, labels, full = batch
  dev = 'cuda'
  _, feats = task.encode_motion(poses.to(dev), pad_mask.to(dev), text_list, lengths.to(dev), labels.to(dev))
  texts = task.encode_text(text_list, full)
  return feats.float(), texts.float(), text_list


def batch_stats(task, ds, indices):
  m, t, tl = encode(task, longseq_pose_text_collate([ds[i] for i in indices]))
  takes = [x[3] for x in tl]
  verbs = [primary_verb(x[2]) for x in tl]
  mn, tn = F.normalize(m, dim=-1), F.normalize(t, dim=-1)
  logits = mn @ tn.t() / 0.07
  labels = torch.arange(len(m), device=m.device)
  full = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)).item() / 2
  wt, rows = within_take_loss(m, t, takes, verbs)
  tk = torch.tensor([hash(x) for x in takes], device=m.device)
  vb = torch.tensor([hash(x) for x in verbs], device=m.device)
  hard = (tk[:, None] == tk[None, :]) & (vb[:, None] != vb[None, :])
  p = logits.softmax(dim=1)
  p_neg = p.clone()
  p_neg.fill_diagonal_(0)
  share = (p_neg * hard).sum(1) / p_neg.sum(1).clamp_min(1e-12)
  n_hard = hard.sum(1).float()
  return full, wt.item(), rows, share.mean().item(), n_hard.mean().item()


def within_take_mcq(task, ds, n_questions, rng):
  df = ds.df
  verbs = df['description_text'].map(primary_verb).to_numpy()
  by_take = df.groupby('take_name').indices
  questions = []
  takes = list(by_take)
  while len(questions) < n_questions:
    idx = by_take[rng.choice(takes)]
    a = rng.choice(list(idx))
    cands = [j for j in idx if verbs[j] != verbs[a]]
    if len(cands) >= 4:
      questions.append([a] + rng.sample(cands, 4))
  m2t = t2m = 0
  for q0 in range(0, len(questions), 200):
    qs = questions[q0:q0 + 200]
    flat = [i for q in qs for i in q]
    m, t, _ = encode(task, longseq_pose_text_collate([ds[i] for i in flat]))
    m = F.normalize(m, dim=-1).view(len(qs), 5, -1)
    t = F.normalize(t, dim=-1).view(len(qs), 5, -1)
    sim = torch.einsum('qid,qjd->qij', m, t)
    m2t += (sim[:, 0, :].argmax(-1) == 0).sum().item()
    t2m += (sim[:, :, 0].argmax(-1) == 0).sum().item()
  return 100 * m2t / len(questions), 100 * t2m / len(questions)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('ckpt')
  parser.add_argument('--batches', type=int, default=20)
  parser.add_argument('--questions', type=int, default=5000)
  parser.add_argument('--k', type=int, default=8)
  opts = parser.parse_args()
  random.seed(0)
  torch.manual_seed(0)
  args, task = build(opts.ckpt)
  ds = task.train_dataloader().dataset
  n = len(ds)

  rows = []
  sampler = SameTakeBatchSampler(ds.df['take_name'], ds.df['timestamp'], 1024, opts.k, 4.0)
  same = list(iter(sampler))[: opts.batches]
  rand = [random.sample(range(n), 1024) for _ in range(opts.batches)]
  for name, batches in [('random', rand), (f'same-take K={opts.k}', same)]:
    s = np.array([batch_stats(task, ds, b) for b in batches])
    rows.append((name, *s.mean(0)))
    print(
        f'{name:>16}: full loss {s[:, 0].mean():.3f} | within-take loss {s[:, 1].mean():.3f}'
        f' ({s[:, 2].mean():.0f} anchors/batch) | same-take negatives/anchor {s[:, 4].mean():.2f}'
        f' | softmax share on them {100 * s[:, 3].mean():.2f}%'
    )

  # Deterministic windows, as in the test.
  ds.mode = 'test'
  m2t, t2m = within_take_mcq(task, ds, opts.questions, random.Random(0))
  print(f'within-take 5-way MCQ on training takes ({opts.questions} questions): m2t {m2t:.2f}  t2m {t2m:.2f}')


if __name__ == '__main__':
  main()
