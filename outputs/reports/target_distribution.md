# Target Distribution

## train
- min/max/mean/std: 0.000000 / 1.000000 / 0.140640 / 0.138475
- p80/p85/p90/p95: 0.224487 / 0.267822 / 0.330078 / 0.431885

## val
- min/max/mean/std: 0.000000 / 0.402100 / 0.094133 / 0.063940
- p80/p85/p90/p95: 0.142944 / 0.160645 / 0.183838 / 0.220404

## test
- min/max/mean/std: 0.000000 / 0.469238 / 0.103191 / 0.068976
- p80/p85/p90/p95: 0.157349 / 0.176025 / 0.200562 / 0.235297

## Threshold Counts
### train
- fixed_0.50: threshold=0.500000, count=1842
- fixed_0.60: threshold=0.600000, count=779
- fixed_0.70: threshold=0.700000, count=274
- fixed_0.80: threshold=0.800000, count=89
- train_p80: threshold=0.224487, count=12496
- val_p80: threshold=0.142944, count=22876
- train_p85: threshold=0.267822, count=9368
- val_p85: threshold=0.160645, count=19892
- train_p90: threshold=0.330078, count=6244
- val_p90: threshold=0.183838, count=16710
- train_p95: threshold=0.431885, count=3128
- val_p95: threshold=0.220404, count=12851
### val
- fixed_0.50: threshold=0.500000, count=0
- fixed_0.60: threshold=0.600000, count=0
- fixed_0.70: threshold=0.700000, count=0
- fixed_0.80: threshold=0.800000, count=0
- train_p80: threshold=0.224487, count=611
- val_p80: threshold=0.142944, count=2667
- train_p85: threshold=0.267822, count=242
- val_p85: threshold=0.160645, count=2004
- train_p90: threshold=0.330078, count=48
- val_p90: threshold=0.183838, count=1335
- train_p95: threshold=0.431885, count=0
- val_p95: threshold=0.220404, count=667
### test
- fixed_0.50: threshold=0.500000, count=0
- fixed_0.60: threshold=0.600000, count=0
- fixed_0.70: threshold=0.700000, count=0
- fixed_0.80: threshold=0.800000, count=0
- train_p80: threshold=0.224487, count=861
- val_p80: threshold=0.142944, count=3322
- train_p85: threshold=0.267822, count=344
- val_p85: threshold=0.160645, count=2553
- train_p90: threshold=0.330078, count=74
- val_p90: threshold=0.183838, count=1765
- train_p95: threshold=0.431885, count=4
- val_p95: threshold=0.220404, count=952

## Suggested Threshold
{
  "mode": "quantile",
  "reference_split": "val",
  "value": 0.9,
  "threshold": 0.183837890625
}
- Warning: threshold_0.70_has_no_positive_cases_in_test
