# Follow-up runs: grouping only, and Nymeria zero-shot (design, 3 Nov 2026)

Two additions to experiment 1 ([`results/hard_negatives.md`](../results/hard_negatives.md)), fixed before any of
their results are seen. Decision D5 in [`decisions.md`](decisions.md).

## 1. Grouping only (setup G)

**Question.** B' changes two things at once: batches grouped by take and the within-take loss term. The results
file lists this as a limit. G isolates them.

**Runs.** G = B' without the loss term: `--same_take_k 8 --same_take_gap 4 --neg_mask verb`, everything else as in
experiment 1 (common flags in the results file), seeds 0, 1, 2, 20 epochs, last checkpoint tested with
`scripts/slurm/eval_run.sh`. No pilot gate: all three seeds run.

**Result table** (Ego-Exo4D 5-way test, motion-to-text, `compare_runs.py`, 3 seeds per setup):

| Comparison | Split | Difference | 95% interval |
|---|---|---|---|
| B' - G | procedural / physical / overall | | |
| G - C | procedural / physical / overall | | |

plus G's per-seed rows in the per-seed table.

**Rule, fixed now.**
- "The within-take loss is needed for the gain": interval of B' - G on procedural above zero, and B' above G in
  each seed.
- "Grouping alone helps": interval of G - C on procedural above zero. If it includes zero, we write that no gain
  from grouping alone was detected, not that grouping has no effect.

**Comparability.** A, C and B' were trained on code a8fc1ec / bb5dc84; G runs on current `main`. Every change since
then sits behind a flag that is off here, and the regression check (`check_reproduction.sh`, 44.81 / 33.61) is run
on the deployed revision before G is queued. The revision of each run is recorded in the results file.

## 2. Nymeria zero-shot

**Question.** Do the experiment-1 changes help or hurt transfer to Nymeria, the paper's zero-shot egocentric test
(paper, Sec. 4 "How generalizable is the learned CamFormer?", Table 2 right: 4,000 questions, 1,000 per narration
type)?

**Difference from the paper.** The paper's Nymeria figures come from the 16-second checkpoint
(`egoexo4d_dur16.pt`, upstream README); all our runs fine-tune the 8-second one. Our figures are therefore compared
with the released 8-second model scored the same way, never with the paper's numbers. Both released checkpoints
are scored once so that the size of this gap is on record.

**Path.** The authors' own test (`train.py --dataset nymeria_pretrain --test --text_column a..d`), run by
`scripts/slurm/eval_nymeria.sh`; `score_nymeria.py` writes one row per question, with the recording as the
resampling unit (130 recordings; every question's five options come from one recording). Runs with gaze or the
domain token are refused, because this path feeds neither, so experiment 2 and 3 models are not scored here.

**Models.** A, C, B' (seeds 0-2), G (seeds 0-2, once trained), released 8 s and 16 s.

**Result table** (motion-to-text, per narration type and pooled over 4,000 questions):

| Model | Body posture | Hands/arms | Legs/feet | Focus | All |
|---|---|---|---|---|---|
| Released 16 s (paper's model) | | | | | |
| Released 8 s | | | | | |
| A, C, B', G (mean of 3 seeds, per-seed range) | | | | | |

and `compare_runs.py` intervals for B' - A and G - A (pooled and per type).

**Rule, fixed now.** These are secondary, descriptive results. We say B' transfers better (or worse) than A only
if the pooled B' - A interval excludes zero; per-type differences are reported without a claim, since four types
on 1,000 questions each are four chances for a chance finding.
