# Settings shared by the Colab scripts, sourced by each of them. Export a variable before
# calling a script to override its default. Nothing here needs a credential.

REPO_DIR=${REPO_DIR:-/content/vla-shift-lab}
LR=${LR:-/content/lr}                    # virtualenv that holds LeRobot, LIBERO and vlashift
PY=${PY:-$LR/bin/python}
OUT=${OUT:-/content/vla_runs}            # a mounted Drive folder keeps results after a disconnect
PATCHED=${PATCHED:-/content/eval_patched.py}
POLICY=${POLICY:-lerobot/smolvla_libero}
REVISION=${REVISION:-a24f1f380461b665158d52acd140152c79e8b36d}
SEED=${SEED:-1000}
EPISODES=${EPISODES:-10}                 # per task; the pre-registered design uses 10
SUITES=${SUITES:-libero_spatial libero_object libero_goal libero_10}
TASK_IDS=${TASK_IDS:-all}                # all, or a list such as [0]
IMAGE_SIZE=${IMAGE_SIZE:-256}            # training size of the checkpoint; LeRobot 0.7.0 renders 360
RESET=${RESET:-hard}                     # hard: reproducible; soft: faster, not bit identical
EXTRA_ARGS=${EXTRA_ARGS:-}               # added to every lerobot-eval call, e.g. --policy.dtype=torch.float32
VIDEOS=${VIDEOS:-0}                      # episodes per task rendered to mp4; each one costs extra time
DUMP_FRAMES=${DUMP_FRAMES:-0}            # policy-input frames saved as png, for a look at the shifts
DRY_RUN=${DRY_RUN:-0}                    # 1: print the commands, run nothing

export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export PATH="$LR/bin:$PATH"

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
now() { date +%H:%M:%S; }
