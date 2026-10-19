# Status

Updated each working day at about 18:00. Newest diary entry first.

## At a glance

| Part | Status |
|---|---|
| Reproduce the paper | ✅ Done: 44.81%, exact match |
| Experiment 1: harder wrong answers | ✅ Done: clear gain on step-by-step tasks |
| Experiment 2: one model for both viewpoints | 🟡 First training runs (seed 0) running |
| Experiment 3: eye gaze | ⬜ Data checked and usable; not started |
| Report, slides, poster | ⬜ Not started |

Key dates: presentation Tue 24 Nov, poster Tue 1 Dec. Plan in [`SCHEDULE.md`](../../SCHEDULE.md).

## Results so far

**Experiment 1** (3 training runs per setup; % of 5-way questions answered correctly, motion-to-text):

| Setup | Step-by-step tasks | Sports/dance | Overall |
|---|---|---|---|
| Authors' released model | 33.6 | 51.4 | 44.8 |
| A: trained a bit more, authors' way | 33.4 | 51.9 | 45.0 |
| C: verb rule only | 33.4 | 52.0 | 45.1 |
| **B′: same-video wrong answers + within-take loss** | **36.9** | **53.4** | **47.3** |

B′ beat A in every run. Gain on step-by-step tasks: +3.5 points (95% interval +2.4 to +4.8). The verb rule alone (C)
does nothing; the gain comes from training on same-video clips.

## Diary

### Mon 19 Oct
- Experiment 2 code finished, reviewed twice, and merged. Checks: our code still gives 44.81%, and the authors'
  third-person model gives its published 46.3% through our code.
- Found that the camera axes in the first-person files differ from the paper's description; third-person clips are
  now turned 90° to match.
- First experiment 2 runs started (seed 0, about 7 hours).
- Wiki started.

### Sun 18 Oct
- Experiment 1 finished: all 3 training runs confirm the gain (table above).

### Sat 17 Oct
- The cluster was offline all day (cut cable). Wrote the comparison script and the experiment 2 design.

### Fri 16 Oct
- Eye-gaze data checked: usable for every test clip.
- First experiment 1 pilot started.
