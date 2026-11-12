# Experiment 4: robustness to synthetic position measurement noise

Status: implementation and CPU tests only. No accuracy results, GPU tests, or real-data verification yet.
Independent of experiments 1–3; starts from the released eight-second ego checkpoint.

## Question and scope
Does fine-tuning with noisy camera positions improve five-way action matching under position measurement errors,
without an unacceptable decrease on clean inputs? This measures synthetic robustness, not real tracking performance.
No new dataset is required: use the existing Ego-Exo4D train and test CSVs, Aria NPZ poses and checkpoint on HPC.

The loader selects its existing window, copies raw (T,7) world poses, and perturbs only xyz before
transform_camera_pose. Independent zero-mean Gaussian errors are drawn per frame and coordinate with standard
deviation sigma. Quaternion columns are untouched; the normal relative-pose conversion still runs afterwards.
The noisy middle-frame position also changes the relative translation origin: resulting relative errors are
correlated, as expected when relative poses are formed from noisy absolute measurements. Rotations/gravity and
caption/window indices remain unchanged. We do not reinterpret rotations as arbitrary numeric vectors.

Sigma is in RAW STORED POSITION UNITS. Confirm units from Aria/source metadata or a known-distance check before
calling sigma 0.01 'one centimetre'. The attached data_structure.md does not establish those units explicitly.
Aria is the only supported pose source. Gaze, Pi3 and joint ego/exo training are deliberately rejected with noise.
IID frame jitter is a simple stress test, not a realistic model of all drift or tracking loss. Large errors can
obliterate the action signal; use small, justified values. Missing-frame and rotation perturbations are not included.

## Controlled comparison
- A: 20 epochs of clean fine-tuning from checkpoints/egoexo4d_dur8.pt.
- N: same start, duration (8 s), batch (1024), learning rate (1e-4), epochs, code and seeds, with noise applied to
  each sampled training window with probability 0.5. The other half remain clean in expectation.
- Existing contrastive loss; no experiment-1 sampler, masks or within-take term.
- Seeds 0,1,2 for training. Compare last completed checkpoints, never a validation-selected best checkpoint.
- Validation is clean monitoring only. Validation recordings overlap benchmark test recordings: do not use
  validation/test scores to select sigma, stop runs, or decide whether to continue after seeing a promising pilot.
- Choose and record sigma and the evaluation plan BEFORE looking at final test scores. If settings need tuning,
  implement a disjoint development split from training takes first; this patch does not add that split.
- Evaluate each model on clean inputs and the same selected sigma with noise seeds 0,1,2. Evaluation perturbations
  are keyed by SHA256(noise_seed, save_id, sampled frame indices), independent of batch and worker order and model.
  Repeated rows of the same window get the same perturbation. Training uses a separate worker-local RNG, so adding
  noise does not consume the RNG used to choose windows. Restarted training is reproducible to the same extent as
  upstream sampling, not guaranteed bit-identical to an uninterrupted run.
- Existing unmodified score_mcq.py provides motion-to-text primary and text-to-motion descriptive scores.
- Report per-training-seed accuracy and differences for every condition. Three noise repeats are repeated
  measurements of the same trained model, NOT nine independent training seeds. A simple prespecified primary
  comparison can use noise seed 0 across training seeds via compare_runs.py; noise seeds 1,2 are sensitivity checks.
  A mean over noise seeds requires an analysis that accounts for repeated questions, takes and training seeds.
- Suggested decision to agree BEFORE evaluation: primary noisy m2t improvement with paired uncertainty above zero,
  and clean deterioration no worse than an agreed practical tolerance. Do not call 'no clean harm' merely because
  a difference is not statistically significant. No quantitative success rule is claimed to have been registered.

## Added command-line options (train.py)
--position_noise_train_sigma FLOAT   default 0 (off)
--position_noise_probability FLOAT   default 0.5, per training window
--position_noise_eval_sigma FLOAT    default 0 (off), only with --test
--position_noise_eval_seed INT       default 0

Evaluation does not infer noise from training options: clean and noisy evaluations must remain distinct.
Noisy features get a sigma/seed subdirectory, preserving clean features. All defaults preserve the old input path.
Resume rejects changed training noise sigma/probability as well as the existing revision/epoch guards.

## What Valerio does
1. Apply the supplied changes to the same repository revision (review extension4.patch). If the repository changed,
   port/review the three edited files rather than overwriting newer code. Add and commit ALL new files before deploy;
   scripts/deploy.sh does not fingerprint untracked files. Deploy using the existing scripts/deploy.sh workflow.
2. Run CPU tests from repository root:
   python -m unittest discover -s tests -p test_position_noise.py -v
3. On HPC check actual NPZ shapes, floating dtype, finite xyz, units, CSV availability, and checkpoint location.
   The public ZIP omits data_files; deployment assumes the cluster's existing data_files links are intact.
4. From ~/dl4cv/camformer, create out/ and err/ before sbatch. For each seed run a ONE-EPOCH smoke job:
   sbatch ../position_noise.sh smoke exp4_v1 0.01 0
   Here 0.01 is an EXAMPLE raw-unit scale, not a justified final choice. Substitute the agreed sigma everywhere.
   Smoke trains through the actual loader and saves a checkpoint; it does not inspect final test accuracy.
5. Inspect smoke output: finite losses, full epoch, sensible xyz perturbation magnitude and unchanged rotations.
   Once units/settings are agreed:
   POSITION_UNITS_CONFIRMED=1 sbatch ../position_noise.sh train exp4_v1 0.01 0
   Repeat smoke/train for seeds 1,2. Use the same committed revision and prefix/settings. Jobs run sequentially
   within each allocation. If timed out, resubmit identical arguments; completed baseline may perform no new
   updates before the augmented run resumes. Existing resume guards apply.
6. After runs complete and the evaluation plan is fixed:
   POSITION_UNITS_CONFIRMED=1 sbatch ../position_noise.sh eval exp4_v1 0.01 0
   Repeat for seeds 1,2. This tests both A and N and writes question CSVs, summary CSVs, commands, logs and manifests
   under ~/data/logs/egoexo4d_pretrain_longseq/exp4_v1_{A,N}_s*/position_noise_evaluation/.
   Evaluations must run sequentially for a given run; concurrent duplicates share feature paths. The commands
   refuse incomplete training checkpoints. Use new prefixes for genuinely new experiments.
7. Run the existing released-checkpoint reproduction check on HPC with all noise options off. Expect 44.81 m2t.
   This check has NOT been run here. Return output CSVs/logs for analysis, including negative results.

## Local verification and limitations
Eight NumPy/SciPy/pandas CPU tests cover input immutability, quaternion preservation, disabled RNG behavior,
noise magnitude, deterministic evaluation, actual loader method + real relative-pose transformation, clean
validation, and training activation. Tests invoke the actual __getitem__ function on in-memory fixtures; they
bypass disk I/O, DataLoader and Lightning. PyTorch/CLIP are unavailable in this workspace, so GPU training,
checkpoint round trips, multi-worker execution, scoring and full reproduction remain HPC checks.
Python syntax and shell syntax are checked locally. No full-training or accuracy claims are made.
