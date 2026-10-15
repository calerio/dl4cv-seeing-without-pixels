# Project log

What was done, by date. Newest first. Plans are in [`SCHEDULE.md`](SCHEDULE.md).

## Thu 15 Oct 2026
- Ran the released Ego-Exo4D checkpoint on the test set (one GPU, under three minutes): 44.81% motion-to-text and 47.18% text-to-motion, identical to the released features and the paper. Details in [`results/reproduction.md`](results/reproduction.md).
- First row of the schedule done.

**Next:** a short training run from scratch to time one epoch.

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
