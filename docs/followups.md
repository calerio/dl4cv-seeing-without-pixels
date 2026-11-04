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
- "The within-take loss adds to grouping": interval of B' - G on procedural above zero, and B' above G in each seed.
  This says the loss adds something on top of grouping, not that grouping contributes nothing; the share of the
  B' - C gain each part carries is read from both comparisons together.
- "Grouping alone helps": interval of G - C on procedural above zero, and G above C in each seed (the same bar as
  B' - G). Otherwise we write that no gain from grouping alone was detected, not that grouping has no effect.

(Amended 4 Nov, after review and before any G result was computed: the first rule was worded "the loss is needed
for the gain", which B' - G alone cannot show, and the second had no per-seed condition.)

**Comparability.** A, C and B' were trained on code a8fc1ec / bb5dc84; G was trained on 3d394e7. Since bb5dc84 the
training code changed in ways meant to be inert with the experiment-1 flags (the clip loader was refactored, the
encoder now always receives domain ids and gravity flags, the within-take block moved under a check for exo
batches), but reading the code is not a measurement, and `check_reproduction.sh` only tests the released model, not
training. So `scripts/slurm/check_training_equivalence.sh` trains one epoch of B' (seed 0; it uses every code path
G uses) once on bb5dc84 and twice on the current code, on one node, and compares the weights. Rule, fixed before it
runs: old and new code are equivalent if they differ by at most twice the difference between the two current-code
runs (GPU nondeterminism). If the check fails, B' - G and G - C are reported as descriptive only, confounded with the
code revision. The revision of each run is recorded in the results file.

## 2. Nymeria zero-shot

**Question.** Do the experiment-1 changes help or hurt transfer to Nymeria, the paper's zero-shot egocentric test
(paper, Sec. 4 "How generalizable is the learned CamFormer?", Table 2 right: 4,000 questions, 1,000 per narration
type)?

**Difference from the paper.** The paper's Nymeria figures come from the 16-second checkpoint
(`egoexo4d_dur16.pt`, upstream README); all our runs fine-tune the 8-second one. Our figures are therefore compared
with the released 8-second model scored the same way, never with the paper's numbers. Both released checkpoints
are scored once so that the size of this gap is on record.

**Path.** The authors' own test (`train.py --dataset nymeria_pretrain --test --text_column a..d`), run by
`scripts/slurm/eval_nymeria.sh`; `score_nymeria.py` writes one row per question. Every question's five options come
from one recording (130 body posture, 192 hands/arms, 167 legs/feet, 7 focus of attention; 198 pooled), but
recordings of one person in one session are not independent, and one session gives 1,132 of the 4,000 questions
(805 of the 1,000 focus questions). The resampling unit is therefore the session: 53 / 61 / 58 / 4 per type, 61
pooled. No test row has an empty trajectory,
so the dataset's fallback to the next row never fires (checked 3 Nov on the cluster). Our runs are loaded as
Lightning checkpoints (`--ckpt`), the released models with `--init_ckpt`; the feasibility job 736059 showed both
paths load and that the released 16 s model gives the paper's 36.30 on body posture. Runs with gaze or the
domain token are refused, because this path feeds neither, so experiment 2 and 3 models are not scored here.

**Models.** A, C, B' (seeds 0-2), G (seeds 0-2, once trained), released 8 s and 16 s.

**Result table** (motion-to-text, per narration type and pooled over 4,000 questions):

| Model | Body posture | Hands/arms | Legs/feet | Focus | All |
|---|---|---|---|---|---|
| Released 16 s (paper's model) | | | | | |
| Released 8 s | | | | | |
| A, C, B', G (mean of 3 seeds, per-seed range) | | | | | |

and `compare_runs.py` intervals for B' - A and G - A (pooled, and per type except focus of attention).

**Rule, fixed now.** These are secondary, descriptive results. We say B' transfers better (or worse) than A only
if the pooled B' - A interval excludes zero; per-type differences are reported without a claim, since four types
on 1,000 questions each are four chances for a chance finding. Focus of attention gets no interval: 4 sessions are
too few clusters for a bootstrap (`compare_runs.py` prints none below 20 clusters).

(Amended 4 Nov, after review: the resampling unit was the recording. The per-model accuracies of job 736108 were in
its log at that point; no comparison between models had been computed.)
