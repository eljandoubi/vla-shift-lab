#!/usr/bin/env bash
# usage: run_one.sh ARM SUITE
# One lerobot-eval run of the patched script. The result directory is written as
# <OUT>/<arm>/<suite>.partial and renamed to <OUT>/<arm>/<suite> only when the run finished and
# the hook record matches the arm, so a directory without .partial or .failed is complete.
set -uo pipefail
. "$(dirname "$0")/common.sh"

arm=${1:?arm name, for example keep or noise_0.1}
suite=${2:?suite name, for example libero_goal}
dir=$OUT/$arm/$suite
work=$dir.partial

if [ -f "$dir/eval_info.json" ]; then
  echo "[$(now)] skip $arm/$suite (finished)"
  exit 0
fi

RENAME='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
args=(
  --policy.path="$POLICY" --policy.pretrained_revision="$REVISION" --policy.device=cuda
  --env.type=libero --env.task="$suite"
  --env.observation_height="$IMAGE_SIZE" --env.observation_width="$IMAGE_SIZE"
  --env.init_states=true --env.max_parallel_tasks=1
  --eval.n_episodes="$EPISODES" --eval.batch_size=1 --eval.use_async_envs=false
  --policy.n_action_steps="$N_ACTION_STEPS"
  --seed="$SEED" --rename_map="$RENAME" --output_dir="$work"
)
[ "$RESET" = soft ] && args+=(--env.hard_reset=false)
[ "$TASK_IDS" != all ] && args+=(--env.task_ids="$TASK_IDS")
if [ -n "$EXTRA_ARGS" ]; then
  read -r -a extra <<< "$EXTRA_ARGS"
  args+=("${extra[@]}")
fi

lines=$("$PY" -m vlashift.cli arm-env "$arm") || { echo "[$(now)] FAILED $arm/$suite: not an arm"; exit 2; }

if [ "$DRY_RUN" = 1 ]; then
  (
    eval "$lines"
    export VLASHIFT_VIDEOS="$VIDEOS" VLASHIFT_DUMP_FRAMES="$DUMP_FRAMES"
    echo "DRY $arm/$suite: $(env | grep '^VLASHIFT_' | sort | tr '\n' ' ')$PY $PATCHED ${args[*]}"
  )
  exit 0
fi

fail() {
  mv "$work" "$dir.failed"
  echo "[$(now)] FAILED $arm/$suite: $1 (log: $dir.failed/run.log)"
  tail -n 5 "$dir.failed/run.log" 2> /dev/null
  exit 1
}

start=$(date +%s)
rm -rf "$work" "$dir.failed"
mkdir -p "$work"
echo "[$(now)] start $arm/$suite"
(
  eval "$lines"
  export VLASHIFT_APPLIED_PATH="$work/vlashift_applied.json"
  export VLASHIFT_VIDEOS="$VIDEOS" VLASHIFT_DUMP_FRAMES="$DUMP_FRAMES"
  "$PY" "$PATCHED" "${args[@]}" > "$work/run.log" 2>&1
)
status=$?
{ [ "$status" -eq 0 ] && [ -f "$work/eval_info.json" ]; } || fail "lerobot-eval exit $status"

baseline=()
[ -f "$OUT/keep/$suite/vlashift_applied.json" ] && baseline=(--baseline "$OUT/keep/$suite")
"$PY" -m vlashift.cli check-applied "$work" "$arm" ${baseline[@]+"${baseline[@]}"} > "$work/check.log" 2>&1 \
  || { cat "$work/check.log"; fail "the hooks did not do what the arm needs"; }

(
  eval "$lines"
  export VLASHIFT_VIDEOS="$VIDEOS" VLASHIFT_DUMP_FRAMES="$DUMP_FRAMES"
  "$PY" -m vlashift.cli meta "$work" --arm "$arm" --suite "$suite" --episodes "$EPISODES" \
    --seed "$SEED" --revision "$REVISION" --image-size "$IMAGE_SIZE" --reset "$RESET" \
    --task-ids "$TASK_IDS" --extra-args="--policy.n_action_steps=$N_ACTION_STEPS${EXTRA_ARGS:+ $EXTRA_ARGS}"
) || fail "could not write meta.json"

mv "$work" "$dir"
echo "[$(now)] done $arm/$suite in $(( $(date +%s) - start )) s, $("$PY" -m vlashift.cli show "$dir" | sed 's/^[^:]*: //')"
