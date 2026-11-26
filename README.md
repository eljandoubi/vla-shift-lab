# vla-shift-lab

An evaluation of a public LIBERO-trained SmolVLA checkpoint, `lerobot/smolvla_libero`: does it read its instruction, and how does its success change when the camera images or the robot's start pose are perturbed? Nothing is trained. Every comparison is paired (same seed and init state in every arm) and comes with Wilson intervals.

## Results

No run yet. The block below is written by `scripts/make_tables.py` from the `<arm>/<suite>/eval_info.json` files of a run.

<!-- results:start -->
No run yet.
<!-- results:end -->

To fill it, unpack the archive made by `colab/collect.sh` into `runs/` and run

```bash
uv run python scripts/make_tables.py --runs runs --out results --readme README.md
```

It also writes `results/tables.md` (the table above, the paired comparison with the baseline, timing), `results/summary.json`, `results/episodes.csv` (one line per episode, enough to redo every paired test) and two figures in `results/figures/`.

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

On a runtime with 2 vCPUs, `colab/worker.sh` runs two arms at once. Each worker takes the next run that no other worker has claimed (a claim is a directory made with `mkdir`, which is atomic), and a run of another arm waits for the baseline of its suite. Start it twice with the same settings:

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

**Protocol.** Checkpoint `lerobot/smolvla_libero` at revision `a24f1f380461b665158d52acd140152c79e8b36d`, LeRobot 0.7.0. Four suites (Spatial, Object, Goal, Long = `libero_10`), 10 tasks each, 10 episodes per task. One environment per task (`--eval.batch_size=1`): episode j gets seed 1000 + j and init state j in every arm, which I read in the LeRobot 0.7.0 source and which the smoke test checks through the start pose. Hard resets. Images are rendered at 256 by 256, the training size of the checkpoint; LeRobot 0.7.0 renders 360 by 360 unless told otherwise. Episodes end at the LeRobot step limits (280, 280, 300 and 520).

**Two changes made after a first attempt.** The first attempt used what the install gives: the 50 actions of each chunk played before the next look at the scene (the checkpoint ships `n_action_steps=50`) and MuJoCo 3.8.1. The `keep` baseline came out at 60% on Spatial (30 of 50 episodes) and 78% on Goal (39 of 50), against 90% and 92% in the paper's table for the 0.45B model. I changed two things before running any other arm.

- Actions per chunk. The paper's simulation protocol looks again after every action, and its ablation of this setting (Table 13 of 2506.01844, models trained without robotics pretraining, so not this checkpoint) gives an average LIBERO success of 80.3 with 1 action played per chunk, 82.8 with 10, 70.8 with 30 and 51.8 with 50. I play 10 (`--policy.n_action_steps=10`): ten times fewer policy calls than looking after every action, and five times more than the shipped value.
- MuJoCo 3.3.7. On Spatial task 5 the run with MuJoCo 3.8.1 succeeded in 0 of 5 episodes and the run with 3.3.7 in 5 of 5, from the same start pose of the arm (the printed end-effector position at reset is the same). I did not look for the cause. `colab/setup.sh` pins 3.3.7.

**First pass.** The plan above assumed 8 seconds per episode. Measured on a free T4 runtime (2 vCPUs, rendering on the GPU), an episode takes 12 seconds when it succeeds and 24 when it runs to the step limit. Four suites with 10 episodes per task and 14 arms would therefore take more than 30 hours. The first pass runs Spatial, Object and Goal, the three suites the hypotheses use, with 5 episodes per task: the baseline, the four instruction arms and the first level of each shift, which is 9 arms of 150 episodes. Seeds and init states are those of the first five episodes of the plan. Long, the second level of each shift and the other five episodes of each task are not run in this pass.

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
- H3: noise and occlusion hurt more than brightness. With D over all the episodes run (150 in the first pass) at `noise_0.1`, `occl_0.4` and `bright_0.5`, it holds if the 95% intervals of D(noise) and D(occl) lie entirely above the interval of D(bright), and is falsified if D(bright) is at least as large as either.

No direction is committed for `shuffled`, `paraphrase` and `joint`: the audits above disagree on language (LIBERO-Plus finds models largely insensitive to language variations, LIBERO-Para finds drops of 22 to 52 points under paraphrases), so these arms are reported as exploratory.

`scripts/make_tables.py` applies these thresholds (`h1_verdict`, `h2_verdict` and `h3_verdict` in `src/vlashift/report.py`) and prints the verdict next to the numbers it used. H3 is only judged when the three arms ran the same episodes.

**Statistics.** Wilson 95% interval per cell. Each arm is compared with `keep` on the same episodes in two ways. The exact McNemar test on the discordant pairs is labelled uncorrected: 8 arms times 4 scopes (three suites and all of them) are compared in the first pass and nothing is corrected. The mean difference gets a 95% interval from resampling whole tasks, because the episodes of a task share a scene and an instruction. McNemar's p-value treats episodes as independent and is therefore optimistic; the task-resampling interval is the one to trust.

**Failure modes.** The `lerobot-eval` output holds success, summed reward and maximum reward per episode, and nothing on how a failed episode went. The study reports success only. `VIDEOS=1` saves a few videos (rendered from the raw simulator, not from the shifted images) for a hand-made look.

**Code.** `vlashift.patch` builds a copy of the installed `lerobot_eval.py` with five edits, each located by a string that must occur exactly once, and stops otherwise. The hooks in `vlashift.runtime` read `VLASHIFT_*` variables and write `vlashift_applied.json` at exit; `vlashift check-applied` rejects a run whose hooks did not do what its arm needs, because a shift that silently does nothing would look like robustness.

## Status and limits

- A first pass is running on a free Colab T4 (Spatial, Object and Goal, 5 episodes per task, 9 arms); the results block above stays empty until it has finished. LIBERO needs Linux and its assets come from the Hub, and the weights could not be downloaded on the Mac this was written on, so there is no run there.
- Done on a Mac: the fast test suite, a build of the patched script against the LeRobot 0.7.0 source (it compiles, and the diff to the original is the five edits), and the five instruction arms through a SmolVLA with random weights on CPU (`results/plumbing_check.json`: shapes, token counts, the image hook on torch tensors). Random weights say nothing about success.
- One checkpoint, one training seed. A suite and arm is 50 episodes in the first pass: at 90% success the Wilson interval runs from 78.6 to 95.7, and over the three suites (150 episodes) from 84.2 to 93.8.
- The `other` arm scores success on the original task. It cannot tell a policy that follows the new instruction from one that is only confused.
- Noise is drawn from a seeded generator but follows the policy's trajectory, so it is not paired across arms; the simulator is.
- The occluder, the choice of both cameras and the paraphrases are my choices.
- LIBERO-Plus (a second environment with 10,030 variants) is not part of this code.
- `results/mutants.txt` is the output of `scripts/mutate_check.py`: 45 hand-made faults (in the statistics, the instruction transforms, the parsing of results, the image and joint shifts, the start-pose check, the patch and the verdict thresholds) are applied one at a time to a copy of the sources, and each must make a test fail. It says nothing about faults it does not contain.

## Licence

MIT, see `LICENSE`.
