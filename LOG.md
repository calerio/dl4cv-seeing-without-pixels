# Project log

What was done, by date. Newest first. Plans are in [`SCHEDULE.md`](SCHEDULE.md).

## Tue 13 Oct 2026
- Paper chosen and registered: *Seeing without Pixels* (Xue et al., CVPR 2026).
- Cluster workspace set up under `~/dl4cv/`, with the upstream code cloned.
- Downloaded the authors' public release: four checkpoints, precomputed retrieval features, and camera-pose trajectories for Ego-Exo4D, DynPose-100K and Nymeria (about 18 GB).
- Downloaded the licensed Ego-Exo4D parts we need: metadata, annotations and eye gaze for 4,990 takes (about 13 GB). Contents and sizes are in [`data/EGOEXO4D_ANNOTATIONS.md`](data/EGOEXO4D_ANNOTATIONS.md).
- Started building the conda environment from the upstream `requirements.txt`.
- Wrote the feasibility notes ([`docs/feasibility.md`](docs/feasibility.md)) and the run schedule.
- Created this repository.

**Next:** reproduce the paper's retrieval numbers from the precomputed features, then from the checkpoints.
