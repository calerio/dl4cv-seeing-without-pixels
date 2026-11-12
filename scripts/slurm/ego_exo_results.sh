#!/bin/bash
#SBATCH --job-name="ego_exo_results"
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

# Every number in the version 2 section of results/ego_exo.md (decision D6), from the per-question files written
# by `ego_exo_seed.sh <seeds> v2` (runs ee2_*). CPU only. Version 1 files (ee_*) are read only for the rows that
# compare the two versions.
# Needs Z first (the released ego model on the DynPose test with the version 2 inputs; one short GPU job):
#   OUT=~/data/logs/released/egoexo4d_dur8/test_questions_exo_v2.csv sbatch --job-name=Z2_ego_on_exo \
#     ~/dl4cv/exo_test.sh --pose_encoding rel9d_grav --pose_source vipe --exo_resample_hz 20 --exo_to_aria_axes \
#     --exo_fps_table dynpose100k_fps/vipe_fps_v1.csv.gz --init_ckpt checkpoints/egoexo4d_dur8.pt
# Usage, from ~/dl4cv/camformer:  sbatch ~/dl4cv/ego_exo_results.sh

module load sw/miniconda3
eval "$(conda shell.bash hook)"
conda activate camformer
set -euo pipefail
echo "code revision $(cat REVISION)"
L=~/data/logs
declare -A DIR=([Sego]=egoexo4d_pretrain_longseq [Sexo]=dynpose_pretrain [EX]=dynpose_longseq
                [U0]=egoexo4d_dynpose_longseq [U1]=egoexo4d_dynpose_longseq)
f() { for s in 0 1 2; do echo -n "$L/${DIR[$2]}/$1_$2_s$s/test_questions_$3.csv "; done; }  # version, setup, test

echo "== every version 2 run finished its 20 epochs, all on one code revision"
revs=""
for x in Sego Sexo EX U0 U1; do
  for s in 0 1 2; do
    d=$L/${DIR[$x]}/ee2_${x}_s$s
    python check_finished.py "$d"
    r=$(cat "$d/REVISION_AT_START")
    echo "ee2_${x}_s$s finished, revision $r"
    revs="$revs $r"
  done
done
[ "$(echo $revs | tr ' ' '\n' | sort -u | wc -l)" -eq 1 ] || { echo "FAIL: runs differ in code revision"; exit 1; }

python - "$L" <<'PYEOF'
import sys

import numpy as np
import pandas as pd
from scipy import stats

L = sys.argv[1]
DIR = {'Sego': 'egoexo4d_pretrain_longseq', 'Sexo': 'dynpose_pretrain', 'EX': 'dynpose_longseq',
       'U0': 'egoexo4d_dynpose_longseq', 'U1': 'egoexo4d_dynpose_longseq'}
TESTS = {'Sego': ['ego'], 'Sexo': ['exo'], 'EX': ['ego', 'exo'], 'U0': ['ego', 'exo'], 'U1': ['ego', 'exo']}


def read(x, s, test, v='ee2'):
  return pd.read_csv(f'{L}/{DIR[x]}/{v}_{x}_s{s}/test_questions_{test}.csv')


def acc(x, test, mask=None):
  """Accuracy (%) of each seed of one setup."""
  out = []
  for s in range(3):
    m = read(x, s, test)['m2t'].astype(float).to_numpy()
    out.append(100 * (m if mask is None else m[mask]).mean())
  return np.array(out)


print('\n== per-run accuracies, version 2')
for x in DIR:
  for s in range(3):
    parts = []
    if 'ego' in TESTS[x]:
      q = read(x, s, 'ego')
      by = {t: 100 * g.m2t.mean() for t, g in q.groupby('type')}
      parts.append(f"ego m2t all {100 * q.m2t.mean():.2f} proc {by['procedural']:.2f} phys {by['physical']:.2f}"
                   f" t2m {100 * q.t2m.mean():.2f} ({len(q)} questions)")
    if 'exo' in TESTS[x]:
      q = read(x, s, 'exo')
      parts.append(f"exo m2t {100 * q.m2t.mean():.2f} t2m {100 * q.t2m.mean():.2f}"
                   f" ({len(q)} questions, {q['take'].nunique()} answer videos)")
    print(f'{x} s{s} | ' + ' | '.join(parts))
z = pd.read_csv(f'{L}/released/egoexo4d_dur8/test_questions_exo_v2.csv')
print(f'Z, released ego model | exo m2t {100 * z.m2t.mean():.2f} t2m {100 * z.t2m.mean():.2f}')

print('\n== DynPose test clips from videos that also have training clips')
test = pd.read_csv('data_files/dynpose100k/metadata_samevideo_test5000.csv')
train = pd.read_csv('data_files/dynpose100k/metadata_train_v1.csv')
assert (test['video_uid'].iloc[::5].to_numpy() == z['take'].to_numpy()).all(), 'question order differs'
used = train[train['vipe_pose_path'].notnull() & (train['vipe_pose_scale'] < 20)]  # what the exo runs train on
group = test.index // 5
for name, t in (('training split', train), ('training clips used (ViPE, scale filter)', used)):
  seen = test['video_uid'].isin(set(t['video_uid']))
  print(f'{name}: {seen.sum()} of {len(test)} test clips, {test.loc[seen, "video_uid"].nunique()} of '
        f'{test["video_uid"].nunique()} test videos, {seen.groupby(group).any().sum()} of {group.nunique()} questions')
print(f"test clips that are themselves training clips: {test['uid'].isin(set(train['uid'])).sum()}")
clean = ~test['video_uid'].isin(set(used['video_uid'])).groupby(group).any().to_numpy()
print(f'questions with none of their five clips from a video used in training: {clean.sum()}')
sub = {x: acc(x, 'exo', clean) for x in ('Sexo', 'EX', 'U0', 'U1')}
for x, a in sub.items():
  print(f'{x}: all questions {acc(x, "exo").mean():.2f} | those {clean.sum()} {a.mean():.2f} '
        f'(seeds {" / ".join(f"{v:.2f}" for v in a)})')
for b, a in (('U1', 'Sexo'), ('U1', 'U0'), ('U1', 'EX'), ('U0', 'EX')):
  print(f'{b} - {a} on those questions: {sub[b].mean() - sub[a].mean():+.2f}')

print('\n== seed-only intervals: Welch t over the three seeds of each setup, test questions held fixed')
for label, a, b, t in (('U1 - S-ego', 'Sego', 'U1', 'ego'), ('U1 - S-exo', 'Sexo', 'U1', 'exo'),
                       ('U1 - U0', 'U0', 'U1', 'exo'), ('U1 - U0', 'U0', 'U1', 'ego'),
                       ('U1 - E->X', 'EX', 'U1', 'exo'), ('U0 - E->X', 'EX', 'U0', 'exo')):
  sa, sb = acc(a, t), acc(b, t)
  va, vb = sa.var(ddof=1) / 3, sb.var(ddof=1) / 3
  df = (va + vb) ** 2 / (va ** 2 / 2 + vb ** 2 / 2)
  h = stats.t.ppf(0.975, df) * np.sqrt(va + vb)
  d = sb.mean() - sa.mean()
  print(f'{t} {label}: {d:+.2f} ({d - h:+.2f} to {d + h:+.2f}), df {df:.1f}')
PYEOF

compare() { echo; echo "== $1"; shift; python compare_runs.py "$@"; }
# The comparisons fixed in docs/ego_exo.md before the first runs, motion-to-text.
compare "ego m2t U1 - S-ego" --a $(f ee2 Sego ego) --b $(f ee2 U1 ego)
compare "exo m2t U1 - S-exo" --a $(f ee2 Sexo exo) --b $(f ee2 U1 exo)
compare "exo m2t U1 - U0" --a $(f ee2 U0 exo) --b $(f ee2 U1 exo)
compare "ego m2t U1 - U0" --a $(f ee2 U0 ego) --b $(f ee2 U1 ego)
compare "exo m2t U1 - E->X" --a $(f ee2 EX exo) --b $(f ee2 U1 exo)
# Descriptive: not fixed in advance.
compare "exo m2t U0 - E->X (descriptive)" --a $(f ee2 EX exo) --b $(f ee2 U0 exo)
compare "exo m2t U0 - S-exo (descriptive)" --a $(f ee2 Sexo exo) --b $(f ee2 U0 exo)
compare "exo m2t E->X - S-exo (descriptive)" --a $(f ee2 Sexo exo) --b $(f ee2 EX exo)
compare "ego m2t E->X - S-ego (descriptive)" --a $(f ee2 Sego ego) --b $(f ee2 EX ego)
compare "ego m2t U0 - S-ego (descriptive)" --a $(f ee2 Sego ego) --b $(f ee2 U0 ego)
compare "ego t2m U1 - S-ego (descriptive)" --metric t2m --a $(f ee2 Sego ego) --b $(f ee2 U1 ego)
compare "exo t2m U1 - S-exo (descriptive)" --metric t2m --a $(f ee2 Sexo exo) --b $(f ee2 U1 exo)
compare "exo t2m U1 - U0 (descriptive)" --metric t2m --a $(f ee2 U0 exo) --b $(f ee2 U1 exo)
compare "ego t2m U1 - U0 (descriptive)" --metric t2m --a $(f ee2 U0 ego) --b $(f ee2 U1 ego)
compare "exo t2m U1 - E->X (descriptive)" --metric t2m --a $(f ee2 EX exo) --b $(f ee2 U1 exo)
# Version 2 - version 1 of the same setup (descriptive): what the frame-rate fix changed. S-ego and S-exo do not
# resample, so their rows show how much a rerun moves without any change to their inputs.
for x in Sexo EX U0 U1; do cmp "exo m2t $x: version 2 - version 1" --a $(f ee $x exo) --b $(f ee2 $x exo); done
for x in Sego EX U0 U1; do cmp "ego m2t $x: version 2 - version 1" --a $(f ee $x ego) --b $(f ee2 $x ego); done
echo "ego_exo results done"
