# Project log

What was done, by date. Newest first. Plans are in [`SCHEDULE.md`](SCHEDULE.md).

## Tue 20 Oct 2026
- Experiment 2 finished: seeds 0-2 of S-ego, S-exo, E->X, U0 and U1 ([`results/ego_exo.md`](results/ego_exo.md)). Against the rule fixed in advance: the joint model matches the ego specialist within 1.5 points but cannot be shown to match the exo specialist (2.6 points lower on average, interval -5.6 to +0.4); the domain token adds nothing measurable. Training the ego model on exo data alone loses 12.7 points on the ego test, which joint training avoids.

**Next:** design note for experiment 3 (eye gaze).

## Mon 19 Oct 2026
- Merged the experiment 2 code ([`docs/ego_exo.md`](docs/ego_exo.md), decision D4). Checks: the reproduction still gives 44.81 with the new flags off and with the domain token on; the released exo model gives 46.30 through the authors' path and 46.20 through ours.
- Measured that the stored ego camera frame has x pointing down, so exo frames are turned by 90 degrees about the forward axis.
- Started seed 0 of experiment 2 (runs U0, U1, S-ego, S-exo, E->X).
- Started the project wiki (GitHub Wiki): status, FAQ, glossary.

**Next:** read the seed-0 results and apply the pilot rule in `docs/ego_exo.md`.

## Sun 18 Oct 2026
- Experiment 1 finished: seeds 0-2 of A, C and B'. B' beats A by 3.5 points on procedural questions, in each seed; all three claims of the rule fixed in D3 are met ([`results/hard_negatives.md`](results/hard_negatives.md)).
- Recorded the pilot gate outcome and the rule for the final claim in D3.

## Sat 17 Oct 2026
- The HPC datacenter has been unreachable since the evening of 16 Oct (cable cut); the pilot results are pending.
- Changes now go through pull requests, reviewed against [`REVIEW.md`](REVIEW.md). Added a regression check that re-scores the released checkpoint (`scripts/slurm/check_reproduction.sh`).
- Added `compare_runs.py`: a paired comparison of two setups with a bootstrap over seeds and takes.
- Wrote the design of the single ego+exo model ([`docs/ego_exo.md`](docs/ego_exo.md)), with its data counts recomputed by `scripts/ego_exo_facts.py`.

**Next:** when the cluster is back, score the pilot, run the regression check, then write the ego+exo code.

## Fri 16 Oct 2026
- Checked the Ego-Exo4D eye-gaze recordings against the released pose clips: gaze is at 10 Hz, starts within 0.1 s of the camera trajectory, and covers every validation and test clip and 99.8% of training clips. No further download is needed. Details in [`results/gaze_check.md`](results/gaze_check.md).
- Checkpoints now leave out the frozen CLIP model: 12 MB instead of 620 MB, tested across a resumed run ([`docs/decisions.md`](docs/decisions.md), D2).
- Found that every question in the Ego-Exo4D test offers five options from the same recording, which motivates training with same-take negatives. Design in [`docs/hard_negatives.md`](docs/hard_negatives.md).
- Added a batch sampler that groups clips of one take (`--same_take_k`), a mask that stops clips with the same main verb from counting as negatives (`--neg_mask`), a within-take loss term (`--within_take_weight`) and `--seed`; all off by default.
- Added a scorer that reproduces Table 2 of the paper per activity type and visibility (`score_mcq.py`), and measured on the released model how much same-take negatives weigh in the loss (decision D3).
- Started two 20-epoch fine-tuning runs from the released checkpoint: baseline (A) and same-take negatives with the mask (B). Tests are queued to run when each finishes.

- Started a seeded pilot pair (baseline and same-take negatives with the within-take term), with a pass/fail rule fixed in advance (D3).

**Next:** score the pilot and apply the D3 rule.

## Thu 15 Oct 2026
- Ran the released Ego-Exo4D checkpoint on the test set (one GPU, under three minutes): 44.81% motion-to-text and 47.18% text-to-motion, identical to the released features and the paper. Details in [`results/reproduction.md`](results/reproduction.md).
- First row of the schedule done.
- Timed training from scratch on one GPU: about 8 minutes per epoch (under 6 for training, 2.5 for validation), so the paper's 500 epochs would take about 70 hours.
- Added our copy of the authors' code under [`src/camformer/`](src/camformer/), with one change: `--resume`, so a long run can continue across several 24-hour cluster jobs ([`scripts/submit_chain.sh`](scripts/submit_chain.sh)). Tested by splitting a run across two jobs. Reasoning and test in [`docs/decisions.md`](docs/decisions.md).
- Brought the experiment schedule forward to October.

**Next:** the gaze check, then same-take hard negatives.

## Tue 13 Oct 2026
- Paper chosen and registered: *Seeing without Pixels* (Xue et al., CVPR 2026).
- Cluster workspace set up under `~/dl4cv/`, with the upstream code cloned.
- Downloaded the authors' public release: four checkpoints, precomputed retrieval features, and camera-pose trajectories for Ego-Exo4D, DynPose-100K and Nymeria (about 18 GB).
- Downloaded the licensed Ego-Exo4D parts we need: metadata, annotations and eye gaze for 4,990 takes (about 13 GB). Contents and sizes are in [`data/EGOEXO4D_ANNOTATIONS.md`](data/EGOEXO4D_ANNOTATIONS.md).
- Unpacked all pose archives and checked them (no errors, no empty files).
- Built the conda environment from the upstream `requirements.txt` (PyTorch 2.7, Lightning 2.5).
- Reproduced the released retrieval numbers from the precomputed features: 44.81% on Ego-Exo4D, matching the paper. Full table in [`results/reproduction.md`](results/reproduction.md).
- Wrote the feasibility notes ([`docs/feasibility.md`](docs/feasibility.md)) and the run schedule.
- Created this repository.

**Next:** run the model from the released checkpoints and check it gives the same numbers.
