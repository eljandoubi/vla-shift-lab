#!/usr/bin/env bash
# usage: smoke.sh
# One episode of libero_spatial task 0 for every arm that has its own code path, about 10 to 15
# minutes. It shows whether the hooks run, how long an episode takes and how much GPU memory
# the policy uses, before hours are spent on the full grid. Results go to SMOKE_OUT.
set -uo pipefail
DUMP_FRAMES=${DUMP_FRAMES:-2}
. "$(dirname "$0")/common.sh"

export OUT=${SMOKE_OUT:-/content/vla_smoke} EPISODES=1 TASK_IDS='[0]' DUMP_FRAMES
suite=libero_spatial
smoke_arms=${SMOKE_ARMS:-keep blank other shuffled paraphrase bright_0.5 noise_0.1 occl_0.4 joint_0.2}
mkdir -p "$OUT"

monitor=
if command -v nvidia-smi > /dev/null && [ "$DRY_RUN" != 1 ]; then
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > "$OUT/gpu_memory_mib.log" &
  monitor=$!
fi

failed=()
for arm in $smoke_arms; do
  bash "$HERE/run_one.sh" "$arm" "$suite" || failed+=("$arm")
done
[ -n "$monitor" ] && kill "$monitor" 2> /dev/null

echo
echo "== runs"
for arm in $smoke_arms; do
  [ -f "$OUT/$arm/$suite/eval_info.json" ] && "$PY" -m vlashift.cli show "$OUT/$arm/$suite"
done
[ -s "$OUT/gpu_memory_mib.log" ] && echo "GPU memory peak: $(sort -n "$OUT/gpu_memory_mib.log" | tail -1) MiB"
if [ -d "$OUT/keep/$suite" ]; then
  "$PY" -m vlashift.cli check-descriptions "$OUT/keep/$suite" && echo "task text matches the instruction table"
fi
echo "policy-input frames to look at: $OUT/<arm>/$suite/frames (keep is the reference)"

if [ "${#failed[@]}" -gt 0 ]; then
  echo "SMOKE FAILED: ${failed[*]}"
  exit 1
fi
echo "SMOKE OK"
