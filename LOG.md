# Project log

What was done, by date. Newest first. Plans are in [`SCHEDULE.md`](SCHEDULE.md).

## Fri 16 Oct 2026
- Checked the Ego-Exo4D eye-gaze recordings against the released pose clips: gaze is at 10 Hz, starts within 0.1 s of the camera trajectory, and covers every validation and test clip and 99.8% of training clips. No further download is needed. Details in [`results/gaze_check.md`](results/gaze_check.md).
- Checkpoints now leave out the frozen CLIP model: 12 MB instead of 620 MB, tested across a resumed run ([`docs/decisions.md`](docs/decisions.md), D2).

**Next:** same-take hard negatives and a retrained baseline.

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
