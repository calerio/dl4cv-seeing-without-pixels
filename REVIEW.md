# Review guide

What a reviewer (person or tool) should look for in a pull request to this repository.
This is research code: a bug rarely crashes, it quietly produces a wrong number. Review for that.

## What counts as a finding
1. **A number we cannot reproduce.** A result in `results/`, `docs/` or the README without the command,
   checkpoint, seed and code revision that produced it.
2. **Leakage.** Test takes or test questions used in training, model selection or tuning. All test takes
   are validation takes; selecting on validation is selecting on test.
3. **A metric that differs from the paper's.** 5-way MCQ, first row of each group is the answer,
   procedural = Cooking, Health, Bike Repair, in-view from `ego_visible`. Changes here must still
   reproduce Table 2 (`scripts/slurm/check_reproduction.sh`).
4. **Uncontrolled comparisons and loose rules.** Runs compared with different seeds, epochs, starting
   checkpoints or code revisions; a single seed reported as an effect; a pass/fail rule set after seeing results.
   A gate or claim rule with a criterion that has no number, uses test data, or has margins finer than the test can
   resolve; a threshold not checked against a known effect (would it have passed a result we already trust?).
   A comparison outside the pre-set ones that is not labelled descriptive.
5. **Silent training changes.** Samplers, losses, masks or defaults that change behaviour of existing
   flags. New behaviour goes behind a flag whose default is the authors' setup.
6. **Docs that overstate.** A claim stronger than the evidence ("improves" from one run, "significant"
   without a test). Also: a hypothesis stated as fact; causal wording for a descriptive result; a pooled number
   written as if it held in each seed; something reasoned from code described as measured, or a measurement cited
   for a different quantity than it measured. For docs, failure means "it lies".
7. **Cluster waste.** A job script that skips the smoke test, saves full checkpoints, cannot resume, or
   silently resumes a stale run from an older code revision.
8. **Misaligned inputs.** Two signals combined in time or space (poses and gaze, ego and exo axes, frame rates)
   without stating, with file and line, where frame i of each comes from, and without a check that would fail if
   they were misaligned. Conventions taken from a paper's text instead of the data or the code that wrote it.
9. **Train and test that differ.** A new input, flag or setting that training uses but the test or diagnostic
   scripts do not pass, so a model is scored with the new part silently off; controls (shuffled or donor inputs)
   that are not identical in training, gates and test.

## What does not count
- Style, naming and formatting, unless it hides a bug. At most three nits per review.
- Upstream code under `external/` and the untouched upstream clone. Flag only what our change breaks.

## How to report
- Each finding gives a file and line as evidence and a concrete failure (inputs, wrong output). No guesses
  from a function's name.
- Tag each one **Important**, **Nit** or **Pre-existing** (a bug this PR did not introduce).
- Start the summary with the count of findings. If there is nothing to fix, say so plainly.
- Review the latest commit. If a check could not be done (a file on another branch, a number only on the
  cluster), say so: an unchecked item is not a pass.
- At most three review rounds per pull request. After the third, step back and reconsider the approach.
