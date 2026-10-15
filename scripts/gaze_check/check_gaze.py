# For every released Ego-Exo4D clip, measure how much of its time window has eye-gaze samples.
import pandas as pd, numpy as np, os, sys
D = os.path.expanduser("~/dl4cv")
traj = pd.read_csv(f"{D}/gaze_check/traj_t0.csv").set_index("take")
print("trajectory read errors:", traj["err"].notna().sum())
gz = {}
def gaze(take):
    if take not in gz:
        p = f"{D}/data/egoexo4d/takes/{take}/eye_gaze/general_eye_gaze.csv"
        gz[take] = pd.read_csv(p, usecols=["tracking_timestamp_us"])["tracking_timestamp_us"].to_numpy() if os.path.exists(p) else None
    return gz[take]
F = f"{D}/seeing_without_pixels/data_files/egoexo4d"
rows = []
for split, fn in [("train", "train_v2.csv"), ("val", "val_v2.csv"), ("test", "test_alltasks_mcqv0.csv")]:
    df = pd.read_csv(f"{F}/{fn}")
    if split == "test" and "start_time" not in df: print(df.columns.tolist()); continue
    for sid, st, en in zip(df.save_id, df.start_time, df.end_time):
        take = sid.rsplit("_", 2)[0]
        g = gaze(take)
        if g is None or take not in traj.index: rows.append((split, take, en - st, np.nan, np.nan, np.nan)); continue
        a = traj.at[take, "traj_t0_us"] + st * 1e6; b = traj.at[take, "traj_t0_us"] + en * 1e6
        t = g[(g >= a) & (g <= b)]
        if len(t) < 2: rows.append((split, take, en - st, len(t), 0.0, en - st)); continue
        gaps = np.diff(np.concatenate([[a], t, [b]])) / 1e6
        cov = 1 - gaps[gaps > 0.25].sum() / (en - st)
        rows.append((split, take, en - st, len(t), cov, gaps.max()))
r = pd.DataFrame(rows, columns=["split", "take", "dur", "n", "cov", "maxgap"])
r.to_csv(f"{D}/gaze_check/clip_coverage.csv", index=False)
allg = np.concatenate([np.diff(v) for v in gz.values() if v is not None and len(v) > 1]) / 1e3
print("gaze sample spacing ms: median %.1f, p1 %.1f, p99 %.1f" % tuple(np.percentile(allg, [50, 1, 99])))
off = [(gz[t][0] - traj.at[t, "traj_t0_us"]) / 1e6 for t in gz if gz[t] is not None and len(gz[t]) and t in traj.index]
print("gaze start minus trajectory start, s: median %.2f, p5 %.2f, p95 %.2f, max %.2f" % (np.median(off), *np.percentile(off, [5, 95]), np.max(off)))
for s, x in r.groupby("split"):
    print(s, "clips", len(x), "takes", x["take"].nunique(), "| no gaze file", x["cov"].isna().sum(),
          "| coverage>=0.95:", round((x["cov"] >= 0.95).mean(), 4), ">=0.8:", round((x["cov"] >= 0.8).mean(), 4),
          "| median samples/s", round((x.n / x.dur).median(), 2))
