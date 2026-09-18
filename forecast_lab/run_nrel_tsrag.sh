#!/usr/bin/env bash
# Train/evaluate forecast_lab on the four NREL CSVs used by timeseries-rag.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PY="${PY:-/ve-remi-dmz-dmz-sh-nas01/data/de_group/ghz_11401/conda_envs/windmllm/bin/python}"
DATA_DIR="/ve-remi-dmz-dmz-sh-nas01/data/de_group/ghz_11401/data/wind_timeseries/NREL/raw_5min_dropzero1000"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/nrel_tsrag}"
DEVICE="${DEVICE:-cuda:0}"

mapfile -t CSVS < <(ls "$DATA_DIR"/NREL_WTK_site{101275,102811,10444,110192}_*.csv)
if [[ ${#CSVS[@]} -ne 4 ]]; then
  echo "Expected 4 NREL CSVs, found ${#CSVS[@]}" >&2
  exit 1
fi

echo "Using Python: $PY"
echo "Device: $DEVICE"
echo "Output: $OUT_ROOT"
"$PY" server_checks.py

for csv in "${CSVS[@]}"; do
  echo "==== $(basename "$csv") ===="
  "$PY" -u run_suite.py \
    --data "$csv" \
    --base-config configs/nrel.json \
    --horizons 24 96 244 \
    --seeds 42 \
    --stage baseline \
    --output-root "$OUT_ROOT" \
    --device "$DEVICE" \
    --skip-complete
done

"$PY" compare_runs.py --root "$OUT_ROOT" --output "$OUT_ROOT/comparison"
echo "Comparison: $OUT_ROOT/comparison"
