# Experiment 1: same-take hard negatives (design, 16 Oct 2026)

> **Read with D3.** This is the first design. After the first two runs (scored on the test: 44.79 and 44.84, no
> change), the design was revised before any seeded run: a within-take loss term was added (setup B'), a pilot gate
> was set, and the comparison became a two-level bootstrap over three seeds and takes (`compare_runs.py`). The
> revised design and its rules are decision D3 in [`decisions.md`](decisions.md); results in
> [`results/hard_negatives.md`](../results/hard_negatives.md).

## Idea
CamFormer is trained with a contrastive loss: in a batch of 1,024 (camera motion, caption) pairs, each motion
must pick its own caption out of the 1,024. Batches are drawn at random from 159k training clips in 2,820
takes, so a clip meets on average about one other clip from its own take per batch. The other 1,023
captions are mostly from other activities ("C dribbles the ball" against a cooking clip) and are easy to rule out.

The Ego-Exo4D test is harder than that. It is a 5-way multiple choice, and in all 7,079 test questions the five
options come from the same take (checked 16 Oct). The model has to tell apart moments of one recording, which
random batches rarely ask it to do. We build batches so that each clip meets several clips from its own take.

## Data facts that shape the design (training split)
- Clips per take: median 34, mean 56, maximum 1,329.
- Captions are dense: the median gap between consecutive captions in a take is 0.8 s, while a training clip
  spans up to 8 s. Neighbouring clips therefore share most of their camera motion.
- 57% of clips have a caption that appears word for word elsewhere in the same take (70% across the whole
  training set: "C walks forward."). The current loss treats such repeats as negatives, so the model is
  pushed to separate two identical sentences.

## Design
1. **Batch sampler** (`--same_take_k K`, default 0 = the authors' random batches). Each epoch, the clips of each
   take are shuffled and split into groups of up to K whose caption timestamps are at least `--same_take_gap`
   seconds apart (default 4 s, half the clip length). Because each clip's window sits at a random position around its caption time, two grouped clips can still share 20-80% of their camera motion; the test has the same property, since its options come from captions as little as 0.8 s apart. Groups are
   shuffled and packed into batches of 1,024. Every clip is still used once per epoch and an epoch has the same
   155 steps, so runs stay comparable. Default K = 8. Measured on the training set: 7.3 same-take clips per clip per batch (1.2 with random batches), of which 5.9 remain negatives after the verb mask (1.0 with random batches).
2. **Negative mask** (`--neg_mask verb`). The paper builds its test so that the four wrong options have a
   different primary action verb from the answer (supplementary material, section 2.1.1). We apply the same rule
   to training: two pairs in a batch whose captions share the primary verb are not used as negatives of each
   other (their logits are set to minus infinity, in both directions). The primary verb is the first word after
   the subject ("C", "The man O", ...); under this definition 98.7% of the released test questions satisfy the
   authors' rule. In a random batch this removes about 35 of 1,023 negatives per clip. Without it, same-take
   batches would teach the model to separate captions such as "C picks the knife" and "C picks the spoon",
   which the test never asks it to do. (`--neg_mask caption` masks only identical captions.)

## Runs
All runs fine-tune from the released Ego-Exo4D checkpoint (`egoexo4d_dur8.pt`) with the authors' settings
(learning rate 1e-4, batch 1,024, 8-second clips, sampled durations) for the same number of epochs, with seeds
0, 1 and 2, and are tested from their last checkpoint on the standard 5-way Ego-Exo4D test.

| Run | Batches | Negative mask | Purpose |
|---|---|---|---|
| A. Baseline | random | none | Same extra training as the others, so gains are not just "more training" |
| B. Hard negatives | same-take, K = 8 | verb | The experiment |
| C. Mask only | random | verb | Control: B against C isolates the same-take batches |

## Evaluation
- Table per run: motion-to-text and text-to-motion accuracy for physical and procedural activities, in-view and
  out-of-view (as in Table 2 of the paper), with the number of questions per cell, and the released checkpoint as
  the first row (`score_mcq.py`, which reproduces the paper's 56.1 / 46.4 / 34.3 / 32.7 / 44.8).
- Mean and standard deviation over the three seeds. For each seed, B against C and C against A are compared on
  the same questions with a paired bootstrap that resamples takes (about 10 questions per take).
- The 7,079 test questions are built from the validation takes. So nothing is chosen on validation: no
  `best` checkpoint, no hyperparameter picked on validation loss.

## Risks
- Same-take clips that are 4 s apart can still describe the same action in different words ("C cuts the
  onion" / "C slices the onion"). The verb mask does not catch synonyms.
- Packing groups changes which clips share a batch only, not the data, loss weights or step count.
- The mask applies to the training loss only; validation loss is unchanged.
