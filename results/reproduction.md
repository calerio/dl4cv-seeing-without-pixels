# Reproduction of the released retrieval results

Run on 13 Oct 2026 with `scripts/slurm/eval_features.sh` (job 699023; CPU only, about two minutes), using the
authors' precomputed features. The task is 5-way multiple choice, so chance is 20%. The paper's metric is motion-to-text.

| Split | Questions | Motion-to-text (%) | Text-to-motion (%) |
|---|---|---|---|
| Ego-Exo4D | 7,079 | 44.81 | 47.18 |
| DynPose-100K, original poses | 1,000 | 39.20 | 44.70 |
| DynPose-100K, ViPE poses | 1,000 | 46.30 | 48.90 |
| Nymeria, body posture (zero-shot) | 1,000 | 36.30 | 25.90 |
| Nymeria, hands and arms (zero-shot) | 1,000 | 26.00 | 25.40 |
| Nymeria, legs and feet (zero-shot) | 1,000 | 30.80 | 23.80 |
| Nymeria, focus of attention (zero-shot) | 1,000 | 33.20 | 21.20 |

The Ego-Exo4D figure matches the paper (44.8 in Table 2, 44.81 in Tables 9-10), and the four Nymeria figures and
the DynPose ViPE figure (46.3, Table 4) match its tables as well. One figure does not: DynPose with the dataset's own
poses gives 39.20 from the released features, against 36.2 in Table 4. The paper reports motion-to-text only; the
text-to-motion column is ours.

The paper's Nymeria figures come from the authors' 16-second checkpoint (`egoexo4d_dur16.pt`; their README pairs it
with Nymeria, and it reproduces 36.30 on body posture, job 736059), while the paper itself describes only an
8-second context. The released checkpoints were saved as the best validation loss of their runs (their
`source_ckpt` fields: `egoexo4d_dur8.pt` epoch 498, `egoexo4d_dur16.pt` epoch 272, `dynpose100k_original.pt`
epoch 188, `dynpose100k_vipe.pt` epoch 315), and the Ego-Exo4D test is drawn from the validation split, so the
released models' test figures carry a small selection on test-overlapping data. Our own runs are always tested at
their last epoch.

## From the released checkpoint

Run on 15 Oct 2026 with `scripts/slurm/eval_checkpoint.sh` (job 702022; one A100 GPU, under three minutes in all): the model is
loaded from `checkpoints/egoexo4d_dur8.pt`, run on the Ego-Exo4D test set, and the features it writes are scored
with the same script.

| Split | Questions | Motion-to-text (%) | Text-to-motion (%) |
|---|---|---|---|
| Ego-Exo4D, from the checkpoint | 7,079 | 44.81 | 47.18 |

Both figures are identical to the released features above, so the code, the environment and the data on the
cluster reproduce the paper's main result end to end.
