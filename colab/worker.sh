#!/usr/bin/env bash
# usage: worker.sh
# One worker of a pool. It walks ARMS in order over SUITES and runs the first run that no other
# worker has claimed; a claim is a directory made with mkdir, which is atomic. Start it twice on a
# 2-vCPU runtime: a run is CPU bound, so two workers about double the throughput.
# A run of another arm waits until the baseline run of its suite has finished, so that it can be
# checked against it.
#   ARMS="keep blank other noise_0.1"   arms in priority order, the baseline first
#   nohup bash colab/worker.sh > $OUT/worker_a.log 2>&1 &
#   nohup bash colab/worker.sh > $OUT/worker_b.log 2>&1 &
set -uo pipefail
. "$(dirname "$0")/common.sh"

arms=${ARMS:?arm names in priority order, for example "keep blank other"}
claims=$OUT/.claims
mkdir -p "$claims"

for arm in $arms; do
  for suite in $SUITES; do
    [ -f "$OUT/$arm/$suite/eval_info.json" ] && continue
    mkdir "$claims/$arm.$suite" 2> /dev/null || continue
    if [ "$arm" != keep ] && [ "$DRY_RUN" != 1 ]; then
      while [ ! -f "$OUT/keep/$suite/eval_info.json" ]; do
        [ -d "$OUT/keep/$suite.failed" ] && break
        sleep 20
      done
    fi
    bash "$HERE/run_one.sh" "$arm" "$suite"
  done
done
echo "[$(now)] worker finished"
