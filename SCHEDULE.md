# Run schedule (paper 20, Seeing without Pixels)

Written 13 Oct 2026, brought forward 15 Oct 2026. These are latest dates. Tick a row when it is done and write the date next to it.

## Fixed dates
- **Tue 20 Oct 2026:** last safe day to download anything from Ego-Exo4D (the download keys expire on 21 Oct).
- **Tue 24 Nov 2026:** final project presentation.
- **Tue 1 Dec 2026:** poster session.

## Schedule

Brought forward on 15 Oct 2026: experiments run in October, while the cluster is quiet, and November is kept for
extra seeds, spare time and writing. The original latest dates are in brackets.

| Done | Finish by | What runs | Why then |
|---|---|---|---|
| [x] 15 Oct | Thu 15 Oct | Reproduce the paper's retrieval numbers from the precomputed features (no GPU), then from the released checkpoints (one short GPU job) | Proves the setup works before anything is built on it |
| [x] 15 Oct | Fri 16 Oct (18 Oct) | One from-scratch training run, only to time an epoch (about 8 minutes); cache the frozen CLIP text embeddings (postponed: not measured yet whether it saves time) | Every later date depends on this number |
| [x] 16 Oct | Sat 17 Oct (20 Oct) | Inspect the gaze files: sampling rate, how many takes have gaze, whether they line up with the released pose clips | Leaves three days to download more before the keys expire on 21 Oct |
| [x] 18 Oct | Tue 20 Oct (1 Nov) | Same-take hard negatives, plus a retrained baseline for a fair comparison | Smallest code change, so it gives a first result early; the baseline is reused by every later experiment |
| [ ] started 19 Oct | Wed 28 Oct (15 Nov) | Single ego+exo model, with and without the domain token | Main contribution and the most engineering (frame rates, axes, two training tasks); code is written while the hard-negative runs train |
| [ ] | Wed 4 Nov (18 Nov) | Eye gaze as a second input: frozen versus full fine-tuning | Depends on the gaze check; first to be cut if time runs short |
| [ ] | Wed 18 Nov | Extra seeds and any follow-up runs the results call for | Spare time if an experiment needs a second attempt |
| [ ] | Sun 8 Nov | First drafts of the report sections, split across the team | Writing starts while experiments still run |
| [ ] | Tue 24 Nov | No new runs from 18 Nov: tables, report and slides; slides ready Fri 20 Nov for a practice run | Leaves six days before the presentation |
| [ ] | Fri 27 Nov | Poster ready | Four days before the poster session |

## Rules of thumb
- Every experiment starts from the released checkpoints: reuse what is trained, train only what is new.
- One cluster account runs 2 jobs at once (2 GPUs, 12 CPUs, 24 h per job), so the hard-negatives and unified-model runs can overlap from late October.
- Gaze experiments run only on the account of the licence holder: the Ego-Exo4D licence is individual.
- If the timing run shows an epoch takes minutes, run several seeds per experiment. If the script's 500-epoch default would take days, cut epochs first and drop the gaze experiment second.

## Where things are on the HPC
```
~/dl4cv/seeing_without_pixels/   upstream code, with checkpoints/ and retrieval_features/ unpacked inside
~/dl4cv/data/camformer_data/     released pose trajectories (Ego-Exo4D, Pi3, DynPose-100K, Nymeria)
~/dl4cv/data/egoexo4d/           licensed Ego-Exo4D parts: metadata, annotations, take_eye_gaze
conda env: camformer             (training and evaluation);  dl  (download tools only)
```
