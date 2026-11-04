# What we changed in the authors' code

This folder is the authors' release ([google-deepmind/seeing_without_pixels](https://github.com/google-deepmind/seeing_without_pixels),
commit `ef8a3ba`, 2 Jun 2026) with our additions. Their `README.md` and `DATA.md` here are unchanged.

**Rule we kept throughout:** every change is behind a flag whose default is the authors' setup. Before any run on
changed code, `scripts/slurm/check_reproduction.sh` scores their released checkpoint with our code at defaults and
must give their 44.81 (33.61 on procedural questions); it always has.

To see every line: clone their repository at `ef8a3ba` and run `diff -ru <their clone> src/camformer`.

## Their files we changed: 7 of 19 (the other 12 are identical)
Almost all of it is additions (about 690 lines added, 37 changed or removed).

| File | What we added | Flags (default = authors') | For |
|---|---|---|---|
| `train.py` | the flags below; a seed for every run; resuming across 24-hour cluster jobs, refused if the code revision or the epoch count changed; validation every n epochs | `--seed`, `--resume`, `--val_every` | all (D1) |
| `loader.py` | a batch sampler that puts up to 8 clips of one recording in a batch; the joint ego + exo loader with single-domain batches | `--same_take_k`, `--same_take_gap`; dataset `egoexo4d_dynpose_longseq`, `--test_domain` | exp. 1, 2 |
| `tasks/pretrain.py` | the verb mask in the contrastive loss; the within-recording loss term; checkpoints without the frozen CLIP model; a check that refuses to drop a trained domain token or gaze layer | `--neg_mask`, `--within_take_weight` | exp. 1 (D2, D3) |
| `tasks/pretrain_longseq.py` | the within-recording term in training; domain ids, gravity flags and gaze passed to the encoder; exo test features | `--within_take_weight`, `--domain_token`, `--gaze` | exp. 1-3 |
| `models/cm_encoder.py` | a domain token and a gaze layer, both starting at zero so the released model is unchanged at step 0; a learned "gravity unknown" token for exo clips; a frozen-encoder option | `--domain_token`, `--gaze_input`, `--freeze_encoder` | exp. 2, 3 |
| `datasets/egoexo4d_pretrain.py` | the input window as frame indices (same random draws as theirs), so gaze can be read for exactly those frames | `--gaze` | exp. 3 |
| `datasets/dynpose.py` | exo clips in the long-sequence format, resampled to 20 poses per second at each clip's own frame rate, turned into the ego camera axes | `--exo_resample_hz`, `--exo_fps_table`, `--exo_to_aria_axes`, `--exo_eval_split` | exp. 2 (D4, D6) |

## Files we added: 15 code files and 3 data files
| Purpose | Files |
|---|---|
| Scoring, one row per test question | `score_mcq.py` (Ego-Exo4D, by activity type and visibility), `score_exo_mcq.py` (DynPose), `score_nymeria.py`, `eval_full_retrieval.py` |
| Comparing setups | `compare_runs.py` (paired two-level bootstrap over seeds and recordings), `check_finished.py` (only finished runs are scored) |
| Data preparation | `gaze_prep.py`, `datasets/gaze.py` (exp. 3), `dynpose_fps.py` (exp. 2 frame rates) |
| Diagnostics | `diagnose_within_take.py` (exp. 1 and 3 gates), `exo_checks.py` (exp. 2 axes, rates, speeds) |
| Tests (CPU, seconds) | `test_hard_negatives.py`, `test_ego_exo.py`, `test_gaze.py`, `test_train_guards.py` |
| Data files | `data_files/egoexo4d/gaze_donors_v1.csv.gz`, `data_files/egoexo4d/gaze_alignment_decision.txt`, `data_files/dynpose100k_fps/vipe_fps_v1.csv.gz` |

Why each change was made, and what it changes in comparisons with the paper: `docs/decisions.md` (D1-D6) and the
design notes in `docs/`.
