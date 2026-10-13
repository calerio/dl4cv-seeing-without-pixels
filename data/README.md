Datasets live on the Bocconi HPC, not in git. Record here where each one is and how to fetch it.

Fetched on 13 Oct 2026. Paths are relative to the home directory of the account that holds the data.

## CamFormer release (public, from the authors' Google Drive)
Source: the Drive folder linked in the upstream `DATA.md`. Fetched with `gdown <folder url>` on the login node.

| What | Where on the HPC | Size |
|---|---|---|
| Released checkpoints (4) | `~/dl4cv/seeing_without_pixels/checkpoints/` | 30 MB |
| Precomputed retrieval features | `~/dl4cv/seeing_without_pixels/retrieval_features/` | 145 MB |
| Pose trajectories: Ego-Exo4D, Ego-Exo4D Pi3, DynPose-100K, Nymeria | `~/dl4cv/data/camformer_data/` | about 18 GB |

Environment variables the upstream loaders expect:
```bash
export CAMFORMER_DATA=~/dl4cv/data/camformer_data
export EGOEXO4D_PRETRAIN_TRAJ_DIR="$CAMFORMER_DATA/data/egoexo4d_pretrain"
export DYNPOSE_DATA_DIR="$CAMFORMER_DATA"
export NYMERIA_DATA_DIR="$CAMFORMER_DATA"
export EGOEXO4D_PI3_TRAJ_DIR="$CAMFORMER_DATA/pi3_preds"   # only for --use_pi3_pose
```

## Ego-Exo4D (licensed)
Source: `egoexo -o ~/dl4cv/data/egoexo4d --parts metadata annotations take_eye_gaze --s3_profile egoexo4d -y` (from `pip install ego4d`). The download keys expire on 21 Oct 2026. The licence is individual, so these files stay in the licence holder's account.

| What | Where on the HPC | Size |
|---|---|---|
| Metadata (`takes.json`, `captures.json`, ...) | `~/dl4cv/data/egoexo4d/` | small |
| Annotations (keystep, proficiency, atomic descriptions, expert commentary, relations, ego pose, splits) | `~/dl4cv/data/egoexo4d/annotations/` | 11 GB |
| Eye gaze, 4,990 takes, 3D and 2D, general and personalized CSVs | `~/dl4cv/data/egoexo4d/takes/<take>/eye_gaze/` | 3.1 GB |

## Environments
- `camformer`: Python 3.9 with the upstream `requirements.txt` (training and evaluation).
- `dl`: download tools only (`ego4d`, `gdown`, `awscli`).
