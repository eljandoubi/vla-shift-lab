#!/usr/bin/env bash
# usage: run_all.sh
# Runs every arm of the chosen stages on every suite, one result directory per arm and suite.
# Finished runs are skipped, so after a disconnect start it again with the same settings.
#   STAGES="baseline instruction shifts"   stages in order (shifts_extra adds the other levels)
#   ARMS="blank noise_0.1"                 explicit arm list, overrides STAGES
#   SUITES, EPISODES, OUT, ...             see common.sh
# Run it under nohup so a closed tab does not stop it:
#   nohup bash colab/run_all.sh > $OUT/run_all.log 2>&1 &
set -uo pipefail
. "$(dirname "$0")/common.sh"

mkdir -p "$OUT"
STAGES=${STAGES:-baseline instruction shifts}
arms=${ARMS:-$("$PY" -m vlashift.cli arms $STAGES)} || exit 2

echo "[$(now)] arms: $(echo $arms)"
echo "[$(now)] suites: $SUITES, $EPISODES episodes per task, output in $OUT"
failed=()
for arm in $arms; do
  for suite in $SUITES; do
    bash "$HERE/run_one.sh" "$arm" "$suite" || failed+=("$arm/$suite")
  done
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "[$(now)] FAILED: ${failed[*]}"
  exit 1
fi
echo "[$(now)] all runs finished"
