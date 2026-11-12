# Working on this repository

How changes get into `main`, for everyone on the team. What a reviewer looks for is in [`REVIEW.md`](REVIEW.md).

## Every change
1. Make a branch from `main` and open a pull request. One change per pull request; nothing is pushed to `main`
   directly.
2. If GitHub reports conflicts with `main`, fix them on your branch before asking for a review.
3. Every pull request is reviewed against `REVIEW.md` before it is merged. Fix what the review finds; after three
   rounds, step back and rethink the approach instead of patching.
4. Once a pull request is merged, push nothing more to its branch: open a new pull request instead. Commits added
   to a merged branch never reach `main`.

## Changes to training, data or test code (`src/camformer/`, `scripts/slurm/`)
- New behaviour goes behind a flag whose default is the authors' setup, so the released model still scores as in
  the paper.
- A change that affects how our numbers compare with the paper gets a numbered entry in
  [`docs/decisions.md`](docs/decisions.md) (D1, D2, ...) in the same pull request: the situation, the decision,
  why, and the effect on comparisons with the paper.
- Record each change in [`src/camformer/CHANGES.md`](src/camformer/CHANGES.md).
- CPU tests sit next to the code as `src/camformer/test_*.py`.
- Before any new run, the changed code is deployed with `scripts/deploy.sh` and
  `scripts/slurm/check_reproduction.sh` must print `reproduction OK` (44.81 overall, 33.61 procedural). Its job id
  goes in the pull request.
- A change to data loading gets a one-epoch smoke test through `train.py` first.

## New experiments
No GPU job until a design note in `docs/` fixes the result table, the seeds, the settings (with the paper passage
or measurement behind each) and the rule for reading the result, and it has been reviewed. Settings are never
chosen by looking at test or validation scores: the test questions are built from the validation recordings.

## Results
Final numbers go into a file in `results/` (commands, runs, job ids, code revisions) through a pull request before
they are quoted anywhere else.
