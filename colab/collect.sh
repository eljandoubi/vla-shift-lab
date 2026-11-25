#!/usr/bin/env bash
# usage: collect.sh [ARCHIVE]
# Packs the small result files of every run (eval_info.json, task text, meta, hook record, logs,
# saved frames) into one tar.gz. Videos and checkpoints are never included.
set -euo pipefail
. "$(dirname "$0")/common.sh"

archive=${1:-$OUT/vla_results_$(date +%Y%m%d_%H%M).tar.gz}
list=$(mktemp)
cd "$OUT"
find . -mindepth 3 -type f \( -name eval_info.json -o -name task_descriptions.json \
  -o -name meta.json -o -name vlashift_applied.json -o -name check.log -o -name run.log \
  -o -path '*/frames/*.png' \) | sort > "$list"
[ -s "$list" ] || { echo "no result files under $OUT"; exit 1; }
tar -czf "$archive" -T "$list"
echo "$(wc -l < "$list") files, $(du -h "$archive" | cut -f1): $archive"
echo "download it with: from google.colab import files; files.download('$archive')"
rm -f "$list"
