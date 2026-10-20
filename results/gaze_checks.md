# Experiment 3 checks: job logs (key lines, verbatim)

Copied from the cluster job outputs (`~/dl4cv/camformer/out/`) so the numbers in `docs/gaze.md` can be traced.
The donor lines of jobs 709477 (one rule, 65,301 windows without a donor) and 709874 (two rules, no split
restriction) are superseded by job 709883 (two rules within split groups), whose output is the committed map.

## Job 709477
```
code revision d900001
[cache] 3597 takes cached to /home/3195720/dl4cv/data/egoexo4d/gaze_cache; 4 without gaze file or traj_t0
[khz] 3601 takes. max(end_time) - trajectory length (s): max +0.0010, takes over +0.005: 0; takes within 0.005 of the end: 3360
[khz] OK: no clip ends after its trajectory
[donors] 207176 clip windows, 141875 with a donor, 65301 without; same take: True; overlapping in time: 0; written to /mnt/beegfsstudents/home/3195720/dl4cv/camformer/data_files/egoexo4d/gaze_donors_v1.csv.gz
[coverage] train physical   clips  19869  mean valid share 99.67%  clips with >=90% valid 99.67%
[coverage] train procedural clips  10131  mean valid share 100.00%  clips with >=90% valid 100.00%
[coverage] val   physical   clips  26493  mean valid share 100.00%  clips with >=90% valid 100.00%
[coverage] val   procedural clips  42580  mean valid share 100.00%  clips with >=90% valid 100.00%
[coverage] test  physical   clips  22290  mean valid share 100.00%  clips with >=90% valid 100.00%
[coverage] test  procedural clips  13105  mean valid share 100.00%  clips with >=90% valid 100.00%
[vor] gaze source: real, 1000 validation clips sampled
[vor]   device axis 0: -5:+0.122 -4:+0.117 -3:+0.098 -2:+0.053 -1:-0.028 +0:-0.132 +1:-0.218 +2:-0.247 +3:-0.224 +4:-0.180 +5:-0.138
[vor]   device axis 1: -5:-0.015 -4:-0.015 -3:-0.014 -2:-0.011 -1:-0.005 +0:-0.001 +1:+0.002 +2:+0.006 +3:+0.011 +4:+0.016 +5:+0.019
[vor]   device axis 2: -5:+0.029 -4:+0.030 -3:+0.028 -2:+0.017 -1:-0.005 +0:-0.031 +1:-0.050 +2:-0.058 +3:-0.058 +4:-0.054 +5:-0.047
[vor] gaze source: shuffled, 1000 validation clips sampled
[vor]   device axis 0: -5:+0.000 -4:+0.001 -3:+0.001 -2:+0.001 -1:+0.002 +0:+0.002 +1:+0.002 +2:+0.002 +3:+0.002 +4:+0.002 +5:+0.002
[vor]   device axis 1: -5:-0.000 -4:-0.001 -3:-0.002 -2:-0.002 -1:-0.001 +0:-0.001 +1:-0.001 +2:-0.001 +3:-0.000 +4:+0.000 +5:+0.001
[vor]   device axis 2: -5:-0.001 -4:+0.000 -3:+0.000 -2:+0.001 -1:+0.002 +0:+0.003 +1:+0.003 +2:+0.003 +3:+0.003 +4:+0.002 +5:+0.002
```

## Job 709871
```
code revision e0abf27
[vor_native] 3000 validation clips, 1558780 native gaze samples, 527730 VOR-dominated, 627 takes
[vor_native] yaw   vs head axis 0 all: most negative at +120 ms (r -0.277, r at 0 ms -0.161, eye/head gain -0.20); -300:+0.152 -250:+0.149 -200:+0.132 -150:+0.092 -100:+0.024 -50:-0.066 +0:-0.161 +50:-0.236 +100:-0.274 +150:-0.271 +200:-0.239 +250:-0.193 +300:-0.150
[vor_native] yaw   vs head axis 0 VOR: most negative at +150 ms (r -0.078, r at 0 ms +0.035, eye/head gain -0.03); -300:+0.219 -250:+0.231 -200:+0.230 -150:+0.209 -100:+0.165 -50:+0.103 +0:+0.035 +50:-0.026 +100:-0.065 +150:-0.078 +200:-0.071 +250:-0.053 +300:-0.035
[vor_native] pitch vs head axis 1 all: most negative at +130 ms (r -0.258, r at 0 ms -0.163, eye/head gain -0.33); -300:+0.056 -250:+0.046 -200:+0.030 -150:+0.003 -100:-0.040 -50:-0.098 +0:-0.163 +50:-0.221 +100:-0.254 +150:-0.256 +200:-0.232 +250:-0.194 +300:-0.152
[vor_native] pitch vs head axis 1 VOR: most negative at +200 ms (r -0.104, r at 0 ms +0.018, eye/head gain -0.08); -300:+0.075 -250:+0.077 -200:+0.076 -150:+0.073 -100:+0.064 -50:+0.047 +0:+0.018 +50:-0.025 +100:-0.067 +150:-0.095 +200:-0.104 +250:-0.095 +300:-0.077
[vor_native] per-take peak (yaw, VOR, takes with >=40 samples and r <= -0.2): 160 takes; p10/p25/median/p75/p90 = +90/+110/+140/+180/+250 ms; share > 0: 100%, share within +-30 ms of 0: 0%
[vor_native]   Basketball     takes  22  median peak +125 ms
[vor_native]   Bike Repair    takes  24  median peak +185 ms
[vor_native]   Cooking        takes  15  median peak +150 ms
[vor_native]   Dance          takes  29  median peak +100 ms
[vor_native]   Health         takes  28  median peak +130 ms
[vor_native]   Music          takes   4  median peak +150 ms
[vor_native]   Rock Climbing  takes  25  median peak +140 ms
[vor_native]   Soccer         takes  13  median peak +230 ms
[vor_native] 20 Hz model features, yaw, central differences, all: most negative at +100 ms; -300:+0.130 -250:+0.131 -200:+0.123 -150:+0.098 -100:+0.043 -50:-0.045 +0:-0.149 +50:-0.232 +100:-0.263 +150:-0.245 +200:-0.201 +250:-0.155 +300:-0.118
[vor_native] 20 Hz model features, yaw, central differences, VOR: most negative at +150 ms; -300:+0.202 -250:+0.217 -200:+0.222 -150:+0.209 -100:+0.168 -50:+0.102 +0:+0.026 +50:-0.038 +100:-0.073 +150:-0.074 +200:-0.055 +250:-0.033 +300:-0.017
```

## Job 709874
```
code revision 77f7cc7
[donors] 207176 clip windows written to /mnt/beegfsstudents/home/3195720/dl4cv/camformer/data_files/egoexo4d/gaze_donors_v1.csv.gz; checks passed
[donors] train physical   clips 104920  same take  42146 ( 40.2%)  other take, same scenario  62774 ( 59.8%)  none 0
[donors] train procedural clips  54266  same take  51050 ( 94.1%)  other take, same scenario   3216 (  5.9%)  none 0
[donors] val   physical   clips  26493  same take   9684 ( 36.6%)  other take, same scenario  16809 ( 63.4%)  none 0
[donors] val   procedural clips  42580  same take  41670 ( 97.9%)  other take, same scenario    910 (  2.1%)  none 0
[donors] test  physical   clips  22290  same take  12787 ( 57.4%)  other take, same scenario   9503 ( 42.6%)  none 0
[donors] test  procedural clips  13105  same take  12356 ( 94.3%)  other take, same scenario    749 (  5.7%)  none 0
```

## Job 709878
```
code revision a2194b1
[coverage] train physical   clips  19869  real: mean valid share 99.67%, >=90% valid 99.67%  shuffled: mean 97.63%, >=90% valid 95.05%
[coverage]   train physical   donor rule same_scenario clips  11832  shuffled mean valid share 98.24%  >=90% valid 96.42%
[coverage]   train physical   donor rule same_take     clips   8037  shuffled mean valid share 96.72%  >=90% valid 93.02%
[coverage] train procedural clips  10131  real: mean valid share 100.00%, >=90% valid 100.00%  shuffled: mean 98.71%, >=90% valid 97.10%
[coverage]   train procedural donor rule same_scenario clips    597  shuffled mean valid share 99.39%  >=90% valid 98.66%
[coverage]   train procedural donor rule same_take     clips   9534  shuffled mean valid share 98.67%  >=90% valid 97.00%
[coverage] val   physical   clips  26493  real: mean valid share 100.00%, >=90% valid 100.00%  shuffled: mean 97.64%, >=90% valid 95.02%
[coverage]   val   physical   donor rule same_scenario clips  16809  shuffled mean valid share 98.45%  >=90% valid 96.88%
[coverage]   val   physical   donor rule same_take     clips   9684  shuffled mean valid share 96.23%  >=90% valid 91.79%
[coverage] val   procedural clips  42580  real: mean valid share 100.00%, >=90% valid 100.00%  shuffled: mean 99.37%, >=90% valid 98.37%
[coverage]   val   procedural donor rule same_scenario clips    910  shuffled mean valid share 99.24%  >=90% valid 98.46%
[coverage]   val   procedural donor rule same_take     clips  41670  shuffled mean valid share 99.38%  >=90% valid 98.37%
[coverage] test  physical   clips  22290  real: mean valid share 100.00%, >=90% valid 100.00%  shuffled: mean 99.03%, >=90% valid 97.93%
[coverage]   test  physical   donor rule same_scenario clips   9503  shuffled mean valid share 99.19%  >=90% valid 98.67%
[coverage]   test  physical   donor rule same_take     clips  12787  shuffled mean valid share 98.91%  >=90% valid 97.37%
[coverage] test  procedural clips  13105  real: mean valid share 100.00%, >=90% valid 100.00%  shuffled: mean 99.79%, >=90% valid 99.48%
[coverage]   test  procedural donor rule same_scenario clips    749  shuffled mean valid share 99.57%  >=90% valid 99.47%
[coverage]   test  procedural donor rule same_take     clips  12356  shuffled mean valid share 99.81%  >=90% valid 99.48%
```

## Job 709879
```
ok zero-init identity
ok gaze frame k -> pose token k
ok pose frame k gets gaze at start_time + frame_idx[k] * frame_dt
ok donor map: 141875 same-take and 65301 same-scenario donors, every clip has one
ok donor rules on a synthetic map
ok shuffled gaze: donor take, donor start + relative time, flag 0 past the donor window
ok collate
all gaze tests passed
```

## Job 709883
```
code revision ee408bd
[donors] sha256 of the CSV content: 2172ba9551d8d09b40b9fa9f82d0a177fc5bb0692d6f9c22e5c938383b848ffa
[donors] 207176 clip windows written to /mnt/beegfsstudents/home/3195720/dl4cv/camformer/data_files/egoexo4d/gaze_donors_v1.csv.gz; checks passed
[donors] train physical   clips 104920  same take  42146 ( 40.2%)  other take, same scenario  62774 ( 59.8%)  none 0
[donors] train procedural clips  54266  same take  51050 ( 94.1%)  other take, same scenario   3216 (  5.9%)  none 0
[donors] val   physical   clips  26493  same take   9684 ( 36.6%)  other take, same scenario  16809 ( 63.4%)  none 0
[donors] val   procedural clips  42580  same take  41670 ( 97.9%)  other take, same scenario    910 (  2.1%)  none 0
[donors] test  physical   clips  22290  same take  12787 ( 57.4%)  other take, same scenario   9503 ( 42.6%)  none 0
[donors] test  procedural clips  13105  same take  12356 ( 94.3%)  other take, same scenario    749 (  5.7%)  none 0
[coverage] train physical   clips  19869  real: mean valid share 99.67%, >=90% valid 99.67%  shuffled: mean 97.65%, >=90% valid 95.06%
[coverage]   train physical   donor rule same_scenario clips  11832  shuffled mean valid share 98.28%  >=90% valid 96.44%
[coverage]   train physical   donor rule same_take     clips   8037  shuffled mean valid share 96.72%  >=90% valid 93.02%
[coverage] train procedural clips  10131  real: mean valid share 100.00%, >=90% valid 100.00%  shuffled: mean 98.69%, >=90% valid 97.07%
[coverage]   train procedural donor rule same_scenario clips    597  shuffled mean valid share 99.08%  >=90% valid 98.16%
[coverage]   train procedural donor rule same_take     clips   9534  shuffled mean valid share 98.67%  >=90% valid 97.00%
[coverage] val   physical   clips  26493  real: mean valid share 100.00%, >=90% valid 100.00%  shuffled: mean 97.57%, >=90% valid 94.83%
[coverage]   val   physical   donor rule same_scenario clips  16809  shuffled mean valid share 98.34%  >=90% valid 96.59%
[coverage]   val   physical   donor rule same_take     clips   9684  shuffled mean valid share 96.23%  >=90% valid 91.79%
[coverage] val   procedural clips  42580  real: mean valid share 100.00%, >=90% valid 100.00%  shuffled: mean 99.38%, >=90% valid 98.36%
[coverage]   val   procedural donor rule same_scenario clips    910  shuffled mean valid share 99.41%  >=90% valid 98.02%
[coverage]   val   procedural donor rule same_take     clips  41670  shuffled mean valid share 99.38%  >=90% valid 98.37%
[coverage] test  physical   clips  22290  real: mean valid share 100.00%, >=90% valid 100.00%  shuffled: mean 98.98%, >=90% valid 97.70%
[coverage]   test  physical   donor rule same_scenario clips   9503  shuffled mean valid share 99.07%  >=90% valid 98.15%
[coverage]   test  physical   donor rule same_take     clips  12787  shuffled mean valid share 98.91%  >=90% valid 97.37%
[coverage] test  procedural clips  13105  real: mean valid share 100.00%, >=90% valid 100.00%  shuffled: mean 99.81%, >=90% valid 99.49%
[coverage]   test  procedural donor rule same_scenario clips    749  shuffled mean valid share 99.93%  >=90% valid 99.60%
[coverage]   test  procedural donor rule same_take     clips  12356  shuffled mean valid share 99.81%  >=90% valid 99.48%
```

## Job 709888
```
ok zero-init identity
ok gaze frame k -> pose token k
ok pose frame k gets gaze at start_time + frame_idx[k] * frame_dt
ok donor map: 141875 same-take and 65301 same-scenario donors, every clip has one
ok donor rules on a synthetic map
ok shuffled gaze: donor take, donor start + relative time, flag 0 past the donor window
ok collate
all gaze tests passed
```
