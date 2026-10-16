# Experiment 1: same-take hard negatives (design, 16 Oct 2026)

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
   seconds apart (default 4 s, half the clip length, so the clips in a group overlap by at most half). Groups are
   shuffled and packed into batches of 1,024. Every clip is still used once per epoch and an epoch has the same
   155 steps, so runs stay comparable. Default K = 8: each clip meets up to 7 same-take negatives.
2. **Duplicate-caption mask** (`--mask_dup_text`). Pairs in a batch with the identical caption are removed from
   the loss as negatives (their logits are set to minus infinity), in both directions. Without it, same-take
   batches would multiply the number of identical-caption "negatives" and teach the model something false.

## Runs
All runs fine-tune from the released Ego-Exo4D checkpoint (`egoexo4d_dur8.pt`) with the authors' settings
(learning rate 1e-4, batch 1,024, 8-second clips, sampled durations) for the same number of epochs (20, about
3 hours on one GPU), and are tested on the standard 5-way Ego-Exo4D test.

| Run | Sampler | Mask | Purpose |
|---|---|---|---|
| A. Baseline | random | off | Same extra training as the others, so gains are not just "more training" |
| B. Hard negatives | K = 8 | on | The experiment |
| C. Mask only | random | on | Separates the effect of the mask from the effect of the sampler |

A and B run first, in parallel; C follows. Extra seeds (November) decide whether differences are larger than
run-to-run noise. If B helps, K = 4 and K = 16 are the first follow-ups.

## Risks
- Same-take clips that are 4 s apart can still describe the same action in different words ("C cuts the
  onion" / "C slices the onion"). The mask only catches identical captions. If B gets worse, this is the first
  suspect.
- Packing groups changes which clips share a batch only, not the data, loss weights or step count.
- The mask applies to the training loss only. Validation loss is unchanged, and every run is tested from its
  last checkpoint (fixed epoch count), so checkpoint choice does not depend on how each run's loss is defined.
