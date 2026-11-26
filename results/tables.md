## Success rate in %, [95% Wilson interval] (successes/episodes)

| arm | Spatial | Object | Goal | Long | All |
|---|---|---|---|---|---|
| keep | 88.0 [76.2, 94.4] (44/50) | 92.0 [81.2, 96.8] (46/50) | 88.0 [76.2, 94.4] (44/50) | not run | 89.3 [83.4, 93.3] (134/150) |
| blank | 0.0 [0.0, 7.1] (0/50) | 2.0 [0.4, 10.5] (1/50) | not run | not run | 1.0 [0.2, 5.4] (1/100) |

## Hypotheses, with the thresholds written in the README

| hypothesis | points, keep minus arm | verdict |
|---|---|---|
| H1: the instruction matters little where the scene fixes the task | blank: Spatial +88.0, Object +90.0 | falsified |
| H2: a wrong instruction changes nothing where the policy follows its visual habit |  | not run |
| H3: noise and occlusion hurt more than brightness |  | not run |

## Paired comparison with keep

| arm | scope | pairs | keep minus arm (points) | 95% CI, tasks resampled | keep only / arm only | p, exact McNemar, uncorrected |
|---|---|---|---|---|---|---|
| blank | Spatial | 50 | +88.0 | [+80.0, +96.0] | 44 / 0 | <0.001 |
| blank | Object | 50 | +90.0 | [+78.0, +100.0] | 45 / 0 | <0.001 |
| blank | All | 100 | +89.0 | [+81.0, +95.0] | 89 / 0 | <0.001 |

## Timing

5 runs, 250 episodes, 35.6 s per episode on average, 2.5 h of run time summed over the runs.
