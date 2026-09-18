#!/usr/bin/env bash
# Explicit server entry point; never run automatically by code installation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PY="${PY:-python}"
DATA_DIR="${DATA_DIR:-/ve-remi-dmz-dmz-sh-nas01/data/de_group/ghz_11401/data/wind_timeseries/NREL/raw_5min_dropzero1000}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/nrel_v2}"
DEVICE="${DEVICE:-cuda:0}"
read -r -a SEED_ARGS <<< "${SEEDS:-42 43 44}"
read -r -a HORIZON_ARGS <<< "${HORIZONS:-24 96 244}"
read -r -a WIDTH_ARGS <<< "${WIDTHS:-64}"
read -r -a VARIANT_ARGS <<< "${VARIANTS:-baseline residual near residual_near}"
shopt -s nullglob
CSVS=()
for site in 101275 102811 10444 110192; do
  MATCHES=("$DATA_DIR"/NREL_WTK_site"$site"_*.csv)
  if [[ ${#MATCHES[@]} -ne 1 ]]; then
    echo "Expected exactly one CSV for site $site in DATA_DIR; found ${#MATCHES[@]}." >&2
    exit 1
  fi
  CSVS+=("${MATCHES[0]}")
done
EXTRA=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA+=(--dry-run)
else
  "$PY" server_checks.py
fi
for csv in "${CSVS[@]}"; do
  "$PY" -u run_forecast_study.py --data "$csv" --config configs/nrel_v2.json \
    --horizons "${HORIZON_ARGS[@]}" --widths "${WIDTH_ARGS[@]}" --seeds "${SEED_ARGS[@]}" \
    --variants "${VARIANT_ARGS[@]}" --output-root "$OUT_ROOT" --device "$DEVICE" \
    --skip-complete "${EXTRA[@]}"
done
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "$PY" compare_runs.py --study --root "$OUT_ROOT" --output "$OUT_ROOT/comparison"
fi
