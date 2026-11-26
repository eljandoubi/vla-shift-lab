## Success rate in %, [95% Wilson interval] (successes/episodes)

| arm | Spatial | Object | Goal | Long | All |
|---|---|---|---|---|---|
| keep | 88.0 [76.2, 94.4] (44/50) | 92.0 [81.2, 96.8] (46/50) | 88.0 [76.2, 94.4] (44/50) | not run | 89.3 [83.4, 93.3] (134/150) |
| blank | 0.0 [0.0, 7.1] (0/50) | 2.0 [0.4, 10.5] (1/50) | not run | not run | 1.0 [0.2, 5.4] (1/100) |
| other | 6.0 [2.1, 16.2] (3/50) | not run | not run | not run | 6.0 [2.1, 16.2] (3/50) |
| bright_0.5 | 96.0 [86.5, 98.9] (48/50) | not run | not run | not run | 96.0 [86.5, 98.9] (48/50) |
| noise_0.1 | 80.0 [67.0, 88.8] (40/50) | not run | not run | not run | 80.0 [67.0, 88.8] (40/50) |
| occl_0.4 | 48.0 [34.8, 61.5] (24/50) | not run | not run | not run | 48.0 [34.8, 61.5] (24/50) |

## Hypotheses, with the thresholds written in the README

| hypothesis | points, keep minus arm | verdict |
|---|---|---|
| H1: the instruction matters little where the scene fixes the task | blank: Spatial +88.0, Object +90.0 | falsified |
| H2: a wrong instruction changes nothing where the policy follows its visual habit | other: Spatial +82.0 | falsified |
| H3: noise and occlusion hurt more than brightness | noise_0.1 +8.0 [-4.0, +20.0], occl_0.4 +40.0 [+16.0, +64.0], bright_0.5 -8.0 [-14.0, -2.0] | unclear |

## Paired comparison with keep

| arm | scope | pairs | keep minus arm (points) | 95% CI, tasks resampled | keep only / arm only | p, exact McNemar, uncorrected |
|---|---|---|---|---|---|---|
| blank | Spatial | 50 | +88.0 | [+80.0, +96.0] | 44 / 0 | <0.001 |
| blank | Object | 50 | +90.0 | [+78.0, +100.0] | 45 / 0 | <0.001 |
| blank | All | 100 | +89.0 | [+81.0, +95.0] | 89 / 0 | <0.001 |
| other | Spatial | 50 | +82.0 | [+68.0, +94.0] | 42 / 1 | <0.001 |
| other | All | 50 | +82.0 | [+68.0, +94.0] | 42 / 1 | <0.001 |
| bright_0.5 | Spatial | 50 | -8.0 | [-14.0, -2.0] | 1 / 5 | 0.219 |
| bright_0.5 | All | 50 | -8.0 | [-14.0, -2.0] | 1 / 5 | 0.219 |
| noise_0.1 | Spatial | 50 | +8.0 | [-4.0, +20.0] | 8 / 4 | 0.388 |
| noise_0.1 | All | 50 | +8.0 | [-4.0, +20.0] | 8 / 4 | 0.388 |
| occl_0.4 | Spatial | 50 | +40.0 | [+16.0, +64.0] | 23 / 3 | <0.001 |
| occl_0.4 | All | 50 | +40.0 | [+16.0, +64.0] | 23 / 3 | <0.001 |

## Timing

9 runs, 450 episodes, 35.4 s per episode on average, 4.4 h of run time summed over the runs.
