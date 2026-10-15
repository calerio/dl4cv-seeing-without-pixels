# Fetch only the first and last lines of each take closed_loop_trajectory.csv (byte-range reads).
import boto3, json, csv, sys
from concurrent.futures import ThreadPoolExecutor
s = boto3.Session(profile_name="egoexo4d").client("s3")
m = json.loads(s.get_object(Bucket="ego4d-consortium-sharing", Key="egoexo-public/v2/take_trajectory/manifest.json")["Body"].read())
jobs = []
for e in m:
    for p in e["paths"]:
        if p["relative_path"].endswith("closed_loop_trajectory.csv"):
            jobs.append((p["relative_path"].split("/")[1], p["source_path"], p["size"]))
def get(j):
    take, src, size = j
    b, k = src[5:].split("/", 1)
    try:
        head = s.get_object(Bucket=b, Key=k, Range="bytes=0-4095")["Body"].read().decode().splitlines()
        tail = s.get_object(Bucket=b, Key=k, Range="bytes=-4096")["Body"].read().decode().splitlines()
        cols = head[0].split(","); i = cols.index("tracking_timestamp_us")
        return take, int(head[1].split(",")[i]), int(tail[-1].split(",")[i]), size, ""
    except Exception as ex:
        return take, "", "", size, repr(ex)[:100]
with ThreadPoolExecutor(16) as ex, open(sys.argv[1], "w") as f:
    w = csv.writer(f); w.writerow(["take", "traj_t0_us", "traj_t1_us", "size", "err"])
    for r in ex.map(get, jobs): w.writerow(r)
print(len(jobs))
