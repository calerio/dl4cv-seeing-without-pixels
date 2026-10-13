# Run schedule (paper 20, Seeing without Pixels)

Written 13 Oct 2026. These are latest dates. Tick a row when it is done and write the date next to it.

## Fixed dates
- **Tue 20 Oct 2026:** last safe day to download anything from Ego-Exo4D (the download keys expire on 21 Oct).
- **Tue 24 Nov 2026:** final project presentation.
- **Tue 1 Dec 2026:** poster session.

## Schedule

| Done | Finish by | What runs | Why then |
|---|---|---|---|
| [ ] | Thu 15 Oct | Reproduce the paper's retrieval numbers from the precomputed features (no GPU), then from the released checkpoints (one short GPU job) | Proves the setup works before anything is built on it |
| [ ] | Sun 18 Oct | One from-scratch training run, only to time an epoch; cache the frozen CLIP text embeddings | Every later date depends on this number |
| [ ] | Tue 20 Oct | Inspect the gaze files: sampling rate, how many takes have gaze, whether they line up with the released pose clips | Last chance to download more before the keys expire |
| [ ] | Sun 1 Nov | Same-take hard negatives, plus a retrained baseline for a fair comparison | Smallest code change, so it gives a first result early |
| [ ] | Sun 15 Nov | Single ego+exo model, with and without the domain token | Main contribution and the most engineering (frame rates, axes, two training tasks) |
| [ ] | Wed 18 Nov | Eye gaze as a second input: frozen versus full fine-tuning | Depends on the gaze check; first to be cut if time runs short |
| [ ] | Tue 24 Nov | No new experiments from 18 Nov: only reruns for seeds, then tables, report and slides | Leaves six days before the presentation |

## Rules of thumb
- Every experiment starts from the released checkpoints: reuse what is trained, train only what is new.
- One cluster account runs 2 jobs at once (2 GPUs, 12 CPUs, 24 h per job), so the hard-negatives and unified-model runs can overlap from early November.
- Gaze experiments run only on the account of the licence holder: the Ego-Exo4D licence is individual.
- If the timing run shows an epoch takes minutes, run several seeds per experiment. If the script's 500-epoch default would take days, cut epochs first and drop the gaze experiment second.

## Where things are on the HPC
```
~/dl4cv/seeing_without_pixels/   upstream code, with checkpoints/ and retrieval_features/ unpacked inside
~/dl4cv/data/camformer_data/     released pose trajectories (Ego-Exo4D, Pi3, DynPose-100K, Nymeria)
~/dl4cv/data/egoexo4d/           licensed Ego-Exo4D parts: metadata, annotations, take_eye_gaze
conda env: camformer             (training and evaluation);  dl  (download tools only)
```
