# vla-shift-lab

An evaluation of a public LIBERO-trained SmolVLA checkpoint, `lerobot/smolvla_libero`: does it read its instruction, and how does its success change when the camera images or the robot's start pose are perturbed? Nothing is trained. Every comparison is paired (same seed and init state in every arm) and comes with Wilson intervals.

## Results

A first pass on one Colab T4 produced the baseline on three suites and the empty instruction on two. The runtime was then reclaimed, and the instruction of another task and three image shifts, on Spatial only, were run again on a second T4 runtime (see Status and limits). The tables below count only what is in `results/run_files/`, and `scripts/make_tables.py` writes them.

<!-- results:start -->
**Success rate in %**, with the 95% Wilson interval and (successes/episodes)

| arm | Spatial | Object | Goal | Long | All |
|---|---|---|---|---|---|
| keep | 88.0 [76.2, 94.4] (44/50) | 92.0 [81.2, 96.8] (46/50) | 88.0 [76.2, 94.4] (44/50) | not run | 89.3 [83.4, 93.3] (134/150) |
| blank | 0.0 [0.0, 7.1] (0/50) | 2.0 [0.4, 10.5] (1/50) | not run | not run | 1.0 [0.2, 5.4] (1/100) |
| other | 6.0 [2.1, 16.2] (3/50) | not run | not run | not run | 6.0 [2.1, 16.2] (3/50) |
| bright_0.5 | 96.0 [86.5, 98.9] (48/50) | not run | not run | not run | 96.0 [86.5, 98.9] (48/50) |
| noise_0.1 | 80.0 [67.0, 88.8] (40/50) | not run | not run | not run | 80.0 [67.0, 88.8] (40/50) |
| occl_0.4 | 48.0 [34.8, 61.5] (24/50) | not run | not run | not run | 48.0 [34.8, 61.5] (24/50) |

The All column adds up the suites an arm ran, so rows that ran different suites do not compare directly; the paired table in `results/tables.md` does.

**Hypotheses**, with the thresholds of the Design section; points are keep minus arm

| hypothesis | points, keep minus arm | verdict |
|---|---|---|
| H1: the instruction matters little where the scene fixes the task | blank: Spatial +88.0, Object +90.0 | falsified |
| H2: a wrong instruction changes nothing where the policy follows its visual habit | other: Spatial +82.0 | falsified |
| H3: noise and occlusion hurt more than brightness | noise_0.1 +8.0 [-4.0, +20.0], occl_0.4 +40.0 [+16.0, +64.0], bright_0.5 -8.0 [-14.0, -2.0] | unclear |
<!-- results:end -->

Rebuild the block, the tables and the figures from the committed files, without a GPU:

```bash
uv run python scripts/make_tables.py --runs results/run_files --out results --readme README.md
```

`results/run_files/<arm>/<suite>/` keeps the four small files of each run (`eval_info.json`, `meta.json`, `vlashift_applied.json`, `task_descriptions.json`); the logs are not kept. After a new run on Colab, unpack the archive made by `colab/collect.sh` into `runs/` and point `--runs` there.

It also writes `results/tables.md` (the table above, the paired comparison with the baseline, timing), `results/summary.json`, `results/episodes.csv` (one line per episode, enough to redo every paired test) and two figures in `results/figures/`. The same command, with another `--out` and without `--readme`, reads `results/first_attempt/` (the runs of the first attempt, see Design) and `results/task5_mujoco_3.3.7/` and `results/task5_mujoco_3.8.1/` (the two checks of Spatial task 5).

### What the numbers show

50 episodes per cell, one checkpoint, simulation only.

- The baseline `keep` succeeds in 44 of 50 episodes on Spatial (88%), 46 on Object (92%) and 44 on Goal (88%). The paper's table gives 90, 96 and 92 for the 0.45B model, so the baseline is 2, 4 and 4 points below it, which fits inside the interval of 50 episodes (76.2 to 94.4 on Spatial).
- An empty instruction stops the policy. `blank` succeeds in 0 of 50 episodes on Spatial and 1 of 50 on Object, 88 and 90 points below `keep` (task-resampled intervals +80 to +96 and +78 to +100). On the same episodes `keep` succeeds and `blank` fails 44 and 45 times, and the reverse never happens. H1 needed a drop under 15 points on both suites and one at least 15 points larger on Goal, and it counts as false from 30 points on either of the first two; it is falsified by a wide margin.
- An empty string is an input the checkpoint never saw in training, so `blank` alone says only that the policy needs a language input. The `other` arm is the test that separates the two readings. With the instruction of the next task of the same suite, `other` succeeds in 3 of 50 Spatial episodes (6%), 82 points below `keep` (task-resampled interval +68 to +94): `keep` succeeds and `other` fails 42 times, and the reverse happens once. H2 counts as false from a drop of 20 points on Spatial or Object, and it is falsified on Spatial, the only suite run for this arm. The policy almost never completes the original task when the words name another one. Whether it carries out the new instruction or is only confused stays open, because success is scored on the original task.
- H3 stays unclear. Occluding 40% of the image side costs 40 points (+16 to +64) and noise of 0.1 costs 8 (-4 to +20), while halving the brightness gives 48 of 50 against 44: the arm is 8 points above `keep` (task-resampled interval +2 to +14, McNemar p = 0.22, uncorrected). The rule needs the intervals of noise and of occlusion to lie above the one of brightness; occlusion does, noise does not (its lower end, -4, is under the upper end of brightness, -2). One level of each shift, 50 episodes, Spatial only.

## Run it on a Colab GPU

A T4 is enough. Mount Drive first so results survive a disconnect. Every block is a `%%bash` cell.

1. Setup, once (installs LeRobot and LIBERO in a Python 3.12 virtualenv, enables GPU rendering, builds the patched evaluation script, checks offscreen rendering):

```bash
git clone https://github.com/eljandoubi/vla-shift-lab.git /content/vla-shift-lab
bash /content/vla-shift-lab/colab/setup.sh
```

The Colab image ships NVIDIA's EGL library without an EGL vendor file, so MuJoCo renders on the CPU through Mesa's llvmpipe: on a T4 runtime with 2 vCPUs a LIBERO step took 430 to 500 ms there and 33 to 36 ms with the GPU. `colab/egl.sh` writes the vendor file, and the setup prints the OpenGL renderer so a CPU fallback is visible (`llvmpipe` instead of `Tesla T4`).

2. Smoke test, nine one-episode runs (every hook once), about 10 to 15 minutes. It prints success, seconds per episode, the start pose of the arm, the GPU memory peak and `SMOKE OK`:

```bash
cd /content/vla-shift-lab && bash colab/smoke.sh
```

3. Plan the hours from the measured speed of the `keep` run (seconds per `libero_spatial` episode; the other suites are scaled by their step limit):

```bash
/content/lr/bin/python -m vlashift.cli plan --sec-per-episode 8
```

4. Run the stages in order. Finished runs are skipped, so after a disconnect start the same cell again:

```bash
export OUT=/content/drive/MyDrive/vla_runs
mkdir -p $OUT
cd /content/vla-shift-lab
STAGES=baseline nohup bash colab/run_all.sh > $OUT/baseline.log 2>&1 &
# then STAGES=instruction, STAGES=shifts, STAGES=shifts_extra
```

Progress of a run: `tail -f $OUT/<arm>/<suite>.partial/run.log`. A run that fails is kept as `<suite>.failed` with its log.

On a runtime with 2 vCPUs, `colab/worker.sh` can run two arms at once. Each worker takes the next run that no other worker has claimed (a claim is a directory made with `mkdir`, which is atomic), and a run of another arm waits for the baseline of its suite. On the T4 each episode took roughly 1.5 to 1.8 times as long as alone (see First pass), so the second worker added roughly 15 to 40% more episodes per hour. Start it twice with the same settings:

```bash
export OUT=/content/vla_runs EPISODES=5 SUITES="libero_spatial libero_object libero_goal"
export ARMS="keep blank other noise_0.1 occl_0.4 bright_0.5 shuffled paraphrase joint_0.1"
mkdir -p $OUT && cd /content/vla-shift-lab
nohup bash colab/worker.sh > $OUT/worker_a.log 2>&1 &
nohup bash colab/worker.sh > $OUT/worker_b.log 2>&1 &
```

To repeat a run that failed, delete its claim `$OUT/.claims/<arm>.<suite>` and start a worker again.

5. Pack the small files (no videos, no checkpoints) and download the archive:

```bash
export OUT=/content/drive/MyDrive/vla_runs
bash /content/vla-shift-lab/colab/collect.sh
```

Settings are environment variables (defaults in `colab/common.sh`): `OUT`, `SUITES`, `EPISODES` (per task, 10), `N_ACTION_STEPS` (10), `STAGES`, `ARMS`, `RESET` (`hard`, or `soft` for speed), `IMAGE_SIZE` (256), `EXTRA_ARGS` (added to every `lerobot-eval` call, for example `--policy.dtype=torch.float32`), `VIDEOS`, `DUMP_FRAMES`, `DRY_RUN=1` (print the commands only). Use the same settings for every arm.

## Design

**Question and prior work.** LIBERO-PRO (2510.03827), LIBERO-Plus (2510.13626), LIBERO-Para (2603.28301) and "Seeing Is Not Reaching" (2611.32206) report, for other models, that VLAs lean on the scene more than on the instruction and lose a lot under modest perturbations. The LIBERO-Plus text has no SmolVLA row. I did not read the other three in full. This study measures one public checkpoint with the same kind of ablations, at a scale one T4 can handle.

**Protocol.** Checkpoint `lerobot/smolvla_libero` at revision `a24f1f380461b665158d52acd140152c79e8b36d` (450,046,176 parameters, from the Hub metadata of that revision). Software: LeRobot 0.7.0, PyTorch 2.15.0, MuJoCo 3.3.7, robosuite 1.4.0 and hf-libero 0.1.5 on Python 3.12.3, on one Tesla T4. Four suites (Spatial, Object, Goal, Long = `libero_10`), 10 tasks each, 10 episodes per task. One environment per task (`--eval.batch_size=1`): episode j gets seed 1000 + j and init state j in every arm, which I read in the LeRobot 0.7.0 source and which the smoke test checks through the start pose. Hard resets. Images are rendered at 256 by 256, the training size of the checkpoint; LeRobot 0.7.0 renders 360 by 360 unless told otherwise. Episodes end at the LeRobot step limits (280, 280, 300 and 520).

**Two changes made after a first attempt.** The first attempt used what the install gives: the 50 actions of each chunk played before the next look at the scene (the checkpoint ships `n_action_steps=50`) and MuJoCo 3.8.1. The `keep` baseline came out at 60% on Spatial (30 of 50 episodes) and 78% on Goal (39 of 50), against 90% and 92% in the paper's table for the 0.45B model. I changed two things before running any other arm.

- Actions per chunk. The paper's simulation protocol looks again after every action, and its ablation of this setting (Table 13 of 2506.01844, models trained without robotics pretraining, so not this checkpoint) gives an average LIBERO success of 80.3 with 1 action played per chunk, 82.8 with 10, 70.8 with 30 and 51.8 with 50. I play 10 (`--policy.n_action_steps=10`): ten times fewer policy calls than looking after every action, and five times more than the shipped value.
- MuJoCo 3.3.7. On Spatial task 5 the run with MuJoCo 3.8.1 succeeded in 0 of 5 episodes and the run with 3.3.7 in 5 of 5, from the same start pose of the arm (the printed end-effector position at reset is the same). I did not look for the cause. `colab/setup.sh` pins 3.3.7.

**First pass.** The plan above assumed 8 seconds per episode. Measured alone on a T4 runtime (2 vCPUs, rendering on the GPU), an episode takes 13 seconds when it succeeds and 32 when it runs to the 280-step limit (Spatial task 5 under the two MuJoCo versions, `results/task5_mujoco_3.3.7/` and `results/task5_mujoco_3.8.1/`). With two runs at a time each episode took 26 to 27 seconds for `keep` and 47 to 52 for `blank`, which is roughly 1.5 to 1.8 times as long as alone. The pair delivered about 180 episodes per hour. Four suites with 10 episodes per task and 14 arms are 5,600 episodes, so more than 30 hours at that rate. The first pass was cut to Spatial, Object and Goal, the three suites the hypotheses use, with 5 episodes per task: 9 arms of 150 episodes, 27 runs, about 8 hours. The arms were queued so that a partial result stays usable (the baseline, `blank`, `other`, the three first-level image shifts, then the exploratory arms). Seeds and init states are those of the first five episodes of the plan. Long, the second level of each shift and the other five episodes of each task are not part of this pass. On the second runtime, two runs at a time, the four Spatial arms took 63 minutes in all: 44 for `other` (51 seconds per episode, most episodes run to the step limit), 26 for `noise_0.1`, 33 for `occl_0.4` and 19 for `bright_0.5`.

**Arms.** `keep` is the baseline.

| arm | what changes |
|---|---|
| `blank` | the instruction is empty (SmolVLA still appends a newline) |
| `other` | the instruction of the next task of the same suite |
| `shuffled` | the words of the instruction in a fixed order that depends only on the text |
| `paraphrase` | a hand-written paraphrase of each of the 40 instructions (`src/vlashift/data/paraphrases.json`) |
| `bright_0.5`, `bright_1.5` | pixel values times 0.5 or 1.5, clipped to [0, 1] |
| `noise_0.1`, `noise_0.05` | Gaussian noise with that standard deviation on pixels in [0, 1] |
| `occl_0.4`, `occl_0.2` | a black square at the image centre, side 40% or 20% of the image side |
| `joint_0.1`, `joint_0.05`, `joint_0.2` | Gaussian noise of that standard deviation in radians on the 7 arm joints of the init state |

Image shifts apply to both cameras, after the environment pre-processor and before the policy pre-processor. Joint noise is added to columns 1 to 7 of the init states (column 0 is time; layout read from the shipped `.pruned_init` files). The draw of a row depends on the task, the seed and the row only, so the levels are rescalings of one draw. Stages: `baseline` (1 arm), `instruction` (4), `shifts` (the first level of each shift, 4) and `shifts_extra` (the other 5).

**Hypotheses, written before any result.** D(S, arm) is success(keep) minus success(arm) on suite S in points.

- H1: the instruction matters little where the scene fixes the task. It holds if D(Spatial, blank) and D(Object, blank) are below 15 and D(Goal, blank) exceeds the larger of them by at least 15. It is falsified if either of the first two is 30 or more, or if D(Goal, blank) does not exceed both. In between the answer is "unclear".
- H2: a wrong instruction changes nothing where the policy follows its visual habit. It holds if D(Spatial, other) and D(Object, other) are below 10, and is falsified if either is 20 or more.
- H3: noise and occlusion hurt more than brightness. With D over the episodes that the three arms ran at `noise_0.1`, `occl_0.4` and `bright_0.5`, it holds if the 95% intervals of D(noise) and D(occl) lie entirely above the interval of D(bright), and is falsified if D(bright) is at least as large as either.

No direction is committed for `shuffled`, `paraphrase` and `joint`: the audits above disagree on language (LIBERO-Plus finds models largely insensitive to language variations, LIBERO-Para finds drops of 22 to 52 points under paraphrases), so these arms are reported as exploratory.

`scripts/make_tables.py` applies these thresholds (`h1_verdict`, `h2_verdict` and `h3_verdict` in `src/vlashift/report.py`) and prints the verdict next to the numbers it used. H1 and H2 are falsified as soon as one suite crosses the line, so a suite that did not run cannot hold that verdict back; holding needs every suite the rule names. H3 is only judged when the three arms ran the same episodes.

**Statistics.** Wilson 95% interval per cell. Each arm is compared with `keep` on the same episodes in two ways. The exact McNemar test on the discordant pairs is labelled uncorrected: every row of the paired table is a separate test and nothing is corrected. The mean difference gets a 95% interval from resampling whole tasks, because the episodes of a task share a scene and an instruction. McNemar's p-value treats episodes as independent and is therefore optimistic; the task-resampling interval is the one to trust.

**Failure modes.** The `lerobot-eval` output holds success, summed reward and maximum reward per episode, and nothing on how a failed episode went. The study reports success only. `VIDEOS=1` saves a few videos (rendered from the raw simulator, not from the shifted images) for a hand-made look.

**Code.** `vlashift.patch` builds a copy of the installed `lerobot_eval.py` with five edits, each located by a string that must occur exactly once, and stops otherwise. The hooks in `vlashift.runtime` read `VLASHIFT_*` variables and write `vlashift_applied.json` at exit; `vlashift check-applied` rejects a run whose hooks did not do what its arm needs, because a shift that silently does nothing would look like robustness.

## Status and limits

- The first pass ran on one Colab T4 and was cut short. The runtime was reclaimed while `occl_0.4` and `bright_0.5` on Spatial were running, and after `other` and `noise_0.1` on Spatial had finished but before I had downloaded them. The baseline on three suites and `blank` on two come from that runtime. The four Spatial arms were then run again on a second T4 runtime with the same seeds, settings and software, and the files of `other`, `noise_0.1`, `occl_0.4` and `bright_0.5` come from it; the lines the first runtime had printed for them are not used. `keep` was not rerun there. The `meta.json` of every run records the GPU model, the Python and package versions and the settings, and they are identical across the two runtimes, as are the task texts and the start pose of the arm at reset. Not run: Object and Goal for these four arms, `blank` on Goal, the eight other arms, Long, the second level of each shift and the other five episodes of each task. To run more, copy `results/run_files/keep` to `$OUT/keep` first (a worker holds every other arm until the baseline files of its suite exist, so without them it waits forever), then start the two workers with the arms and suites you want and `EPISODES=5`; `make_tables.py` then rebuilds the tables. LIBERO needs Linux and its assets come from the Hub, and the weights could not be downloaded on the Mac this was written on, so there is no run there.
- GPU memory is not the limit: one worker peaked at 1,722 MiB in the smoke test, and two workers used 3,376 MiB of the T4's 15,360 MiB. A utilization sample read 96% with two workers, which fits the small gain from the second one.
- Done on a Mac: the fast test suite, a build of the patched script against the LeRobot 0.7.0 source (it compiles, and the diff to the original is the five edits), and the five instruction arms through a SmolVLA with random weights on CPU (`results/plumbing_check.json`: shapes, token counts, the image hook on torch tensors). Random weights say nothing about success.
- One checkpoint, one training seed. A suite and arm is 50 episodes in the first pass: at 90% success the Wilson interval runs from 78.6 to 95.7, and over the three suites (150 episodes) from 84.2 to 93.8.
- The `other` arm scores success on the original task. It cannot tell a policy that follows the new instruction from one that is only confused.
- Noise is drawn from a seeded generator but follows the policy's trajectory, so it is not paired across arms; the simulator is.
- The occluder, the choice of both cameras and the paraphrases are my choices.
- LIBERO-Plus (a second environment with 10,030 variants) is not part of this code.
- `results/mutants.txt` is the output of `scripts/mutate_check.py`: 45 hand-made faults (in the statistics, the instruction transforms, the parsing of results, the image and joint shifts, the start-pose check, the patch and the verdict thresholds) are applied one at a time to a copy of the sources, and each must make a test fail. It says nothing about faults it does not contain.

## Licence

MIT, see `LICENSE`.
