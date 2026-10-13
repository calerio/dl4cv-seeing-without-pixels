# Reproduction of the released retrieval results

Run on 13 Oct 2026 with `scripts/slurm/eval_features.sh` (CPU only, about two minutes), using the authors'
precomputed features. The task is 5-way multiple choice, so chance is 20%. The paper's metric is motion-to-text.

| Split | Questions | Motion-to-text (%) | Text-to-motion (%) |
|---|---|---|---|
| Ego-Exo4D | 7,079 | 44.81 | 47.18 |
| DynPose-100K, original poses | 1,000 | 39.20 | 44.70 |
| DynPose-100K, ViPE poses | 1,000 | 46.30 | 48.90 |
| Nymeria, body posture (zero-shot) | 1,000 | 36.30 | 25.90 |
| Nymeria, hands and arms (zero-shot) | 1,000 | 26.00 | 25.40 |
| Nymeria, legs and feet (zero-shot) | 1,000 | 30.80 | 23.80 |
| Nymeria, focus of attention (zero-shot) | 1,000 | 33.20 | 21.20 |

The Ego-Exo4D figure matches the 44.81% the paper reports for CamFormer, and the four Nymeria figures and the DynPose ViPE figure match the paper's tables as well.

This checks the evaluation script and the released features only. Running the model itself from the released
checkpoints is the next step.
