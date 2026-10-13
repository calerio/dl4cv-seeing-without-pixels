# What is in the Ego-Exo4D annotations download

Checked on the HPC on 13 Oct 2026 (`~/dl4cv/data/egoexo4d/annotations/`, 11 GB).

Annotations are the labels attached to the videos: text and numbers describing what happens in each take. There is no video in them. They are what our experiments are scored against. The downloader fetches the labels for every benchmark of the dataset as one block, so most of the 11 GB is for tasks we do not do.

| Part | Size | What it is | Do we need it? |
|---|---|---|---|
| `ego_pose/` | 8.1 GB | Per-frame body and hand joint positions of the camera wearer (body 2.2 GB, hand 0.8 GB), plus per-frame camera poses (5.2 GB) | Not planned; the camera poses may come in handy |
| `relations_{train,val,test}.json` | 2.0 GB | Object masks matched between the ego and exo views | No |
| `atomic_descriptions_{train,val}.json` | 160 MB | Timestamped sentences saying what the person is doing | Yes: the kind of text the model matches motion against |
| `expert_commentary_{train,val}.json` | 60 MB | Experts' spoken critiques of the performance, transcribed | No |
| `keystep_{train,val}.json` | about 10 MB | Start and end time of each step of a procedure | Yes: procedural-activity experiment |
| `proficiency_demonstration_*`, `proficiency_demonstrator_*` | about 12 MB | Skill ratings of the person | Maybe: the paper tests proficiency |
| `procedure_understanding_*`, `splits.json` | small | Step-ordering labels and the official train/val/test split | Splits yes |

The files we actually use come to roughly 200 MB.

## If storage gets tight
Delete `relations_*` (2 GB) and `ego_pose/*/body` and `ego_pose/*/hand` (about 3 GB) first. Do not delete anything before the gaze check is done: the download keys expire on 21 Oct 2026 and nothing can be fetched again after that.

## Next to the annotations
- `~/dl4cv/data/egoexo4d/takes/<take>/eye_gaze/`: gaze for 4,990 takes (3.1 GB). Each take has `general_eye_gaze.csv`, `personalized_eye_gaze.csv`, their `_2d` versions and `summary.json`.
- `~/dl4cv/data/egoexo4d/*.json`: metadata (`takes.json`, `captures.json`, `participants.json`, `physical_setting.json`, `visual_objects.json`, `metadata.json`).
