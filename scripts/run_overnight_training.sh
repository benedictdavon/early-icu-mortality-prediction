#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/usr/bin:/bin:$PATH"

# Overnight end-to-end training runner.
#
# Usage:
#   bash scripts/run_overnight_training.sh
#
# Useful overrides:
#   RUN_NAME=overnight_gpu_001 bash scripts/run_overnight_training.sh
#   PYTHON=python bash scripts/run_overnight_training.sh
#   REQUIRE_GPU=0 bash scripts/run_overnight_training.sh
#   SKIP_PREPROCESS=1 bash scripts/run_overnight_training.sh
#   SKIP_EXISTING_ARTIFACTS=1 bash scripts/run_overnight_training.sh
#   RUN_MAFNET=0 bash scripts/run_overnight_training.sh
#   MAFNET_EVENTS_PATH=data/processed/first6h_events.csv bash scripts/run_overnight_training.sh

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="${SCRIPT_PATH%/*}"
if [[ "$SCRIPT_DIR" == "$SCRIPT_PATH" ]]; then
  SCRIPT_DIR="."
fi
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

to_unix_path() {
  local path_input="$1"
  if [[ "$path_input" == [A-Za-z]:\\* ]]; then
    local drive="${path_input:0:1}"
    local rest="${path_input:2}"
    rest="${rest//\\//}"
    echo "/${drive,,}$rest"
    return
  fi
  if [[ "$path_input" == [A-Za-z]:/* ]]; then
    local drive="${path_input:0:1}"
    local rest="${path_input:2}"
    echo "/${drive,,}$rest"
    return
  fi
  echo "$path_input"
}

python_can_import_torch() {
  local candidate_python="$1"
  "$candidate_python" -c "import torch" >/dev/null 2>&1
}

try_python_candidate() {
  local candidate="$1"
  if [[ -x "$candidate" ]]; then
    if [[ "$REQUIRE_GPU" == "1" ]] && ! python_can_import_torch "$candidate"; then
      return 1
    fi
    PYTHON="$candidate"
    return 0
  fi
  return 1
}

resolve_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    PYTHON="$(to_unix_path "$PYTHON")"
    if command -v "$PYTHON" >/dev/null 2>&1; then
      if try_python_candidate "$PYTHON"; then
        return
      fi
      echo "Configured PYTHON '$PYTHON' does not satisfy requirements; probing alternatives."
    elif [[ -x "$PYTHON" ]]; then
      if try_python_candidate "$PYTHON"; then
        return
      fi
      echo "Configured PYTHON '$PYTHON' is executable but does not satisfy requirements; probing alternatives."
    else
      echo "Configured PYTHON='$PYTHON' is not executable in this shell."
    fi
  fi

  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    local conda_prefix="$(to_unix_path "$CONDA_PREFIX")"
    for candidate in \
      "$conda_prefix/bin/python" \
      "$conda_prefix/bin/python.exe" \
      "$conda_prefix/Scripts/python" \
      "$conda_prefix/Scripts/python.exe"; do
      try_python_candidate "$candidate" && return
    done
  fi

  local user="${USER:-$USERNAME}"
  for candidate in \
    "/c/Users/$user/miniconda3/envs/early-icu-mortality/python" \
    "/c/Users/$user/miniconda3/envs/early-icu-mortality/python.exe" \
    "/c/Users/$user/miniconda3/python" \
    "/c/Users/$user/miniconda3/python.exe" \
    "/c/Users/$user/anaconda3/envs/early-icu-mortality/python" \
    "/c/Users/$user/anaconda3/envs/early-icu-mortality/python.exe" \
    "/c/Users/$user/anaconda3/python" \
    "/c/Users/$user/anaconda3/python.exe"; do
    try_python_candidate "$candidate" && return
  done

  for candidate in "python" "python3" "py"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      candidate="$(command -v "$candidate")"
      try_python_candidate "$candidate" && return
    fi
  done

  local conda_env
  conda_env="$(conda env list 2>/dev/null | awk '$1 == "'"${CONDA_DEFAULT_ENV:-early-icu-mortality}"'" {print $2; exit}')"
  if [[ -n "$conda_env" ]]; then
    for candidate in "$conda_env/python" "$conda_env/python.exe"; do
      try_python_candidate "$candidate" && return
    done
  fi

  if [[ "$REQUIRE_GPU" != "1" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
      local conda_prefix="$(to_unix_path "$CONDA_PREFIX")"
      for candidate in \
        "$conda_prefix/bin/python" \
        "$conda_prefix/bin/python.exe" \
        "$conda_prefix/Scripts/python" \
        "$conda_prefix/Scripts/python.exe"; do
        if [[ -x "$candidate" ]]; then
          PYTHON="$candidate"
          return
        fi
      done
    fi
  fi

  echo "Unable to locate python. Set PYTHON explicitly, for example:"
  echo "  PYTHON=/c/Users/$user/miniconda3/envs/early-icu-mortality/python.exe bash scripts/run_overnight_training.sh"
  exit 1
}

REQUIRE_GPU="${REQUIRE_GPU:-1}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
SKIP_EXISTING_ARTIFACTS="${SKIP_EXISTING_ARTIFACTS:-1}"
RUN_MODEL_SUITE="${RUN_MODEL_SUITE:-1}"
RUN_XGBOOST_ABLATION="${RUN_XGBOOST_ABLATION:-1}"
RUN_LEGACY_XGBOOST_ENSEMBLE="${RUN_LEGACY_XGBOOST_ENSEMBLE:-1}"
RUN_MAFNET="${RUN_MAFNET:-1}"
RUN_MAFNET_ABLATIONS="${RUN_MAFNET_ABLATIONS:-1}"
RUN_FINAL_SUMMARY="${RUN_FINAL_SUMMARY:-1}"
RUN_PYTEST="${RUN_PYTEST:-1}"

PYTHON="${PYTHON:-}"
resolve_python
log "Using python: $PYTHON"

RUN_NAME="${RUN_NAME:-overnight_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/results/$RUN_NAME}"
LOG_DIR="$RESULTS_ROOT/logs"
CONFIG_DIR="$RESULTS_ROOT/model_configs_gpu"

TARGET_COL="${TARGET_COL:-mortality}"
SEED="${SEED:-42}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-200}"
XGBOOST_ABLATION_ESTIMATORS="${XGBOOST_ABLATION_ESTIMATORS:-1000}"
XGBOOST_EARLY_STOPPING_ROUNDS="${XGBOOST_EARLY_STOPPING_ROUNDS:-50}"
LEGACY_XGBOOST_ENSEMBLE_SIZE="${LEGACY_XGBOOST_ENSEMBLE_SIZE:-7}"

BASELINE_FEATURES_PATH="${BASELINE_FEATURES_PATH:-data/processed/extracted_features.csv}"
EXPANDED_FEATURES_PATH="${EXPANDED_FEATURES_PATH:-data/processed/extracted_features_expanded.csv}"
BASELINE_XGBOOST_PATH="${BASELINE_XGBOOST_PATH:-data/processed/preprocessed_xgboost_features.csv}"
EXPANDED_XGBOOST_PATH="${EXPANDED_XGBOOST_PATH:-data/processed/preprocessed_xgboost_expanded_features.csv}"
MAFNET_EVENTS_PATH="${MAFNET_EVENTS_PATH:-data/processed/first6h_events.csv}"
MAFNET_COHORT_PATH="${MAFNET_COHORT_PATH:-data/processed/final_cohort.csv}"
MAFNET_FEATURES_PATH="${MAFNET_FEATURES_PATH:-$EXPANDED_XGBOOST_PATH}"

COHORT_PATH="${COHORT_PATH:-data/processed/final_cohort.csv}"
COHORT_IDS_PATH="${COHORT_IDS_PATH:-data/processed/cohort_stay_ids.csv}"
BASELINE_FEATURE_STATS_PATH="${BASELINE_FEATURE_STATS_PATH:-$(dirname "$BASELINE_FEATURES_PATH")/feature_statistics.csv}"
EXPANDED_FEATURE_STATS_PATH="${EXPANDED_FEATURE_STATS_PATH:-$(dirname "$EXPANDED_FEATURES_PATH")/feature_statistics_expanded.csv}"

mkdir -p "$RESULTS_ROOT" "$LOG_DIR" "$CONFIG_DIR"
export TMPDIR="$RESULTS_ROOT/tmp"
export TMP="$RESULTS_ROOT/tmp"
export TEMP="$RESULTS_ROOT/tmp"
PYTEST_CACHE_DIR="$RESULTS_ROOT/.pytest_cache"
PYTEST_BASE_TEMP_DIR="$RESULTS_ROOT/pytest_tmp"
mkdir -p "$TMPDIR" "$PYTEST_CACHE_DIR" "$PYTEST_BASE_TEMP_DIR"
RUN_LOG_FILE="${RUN_LOG_FILE:-$RESULTS_ROOT/log.txt}"
GLOBAL_LOG_FILE="${GLOBAL_LOG_FILE:-$ROOT_DIR/log.txt}"
touch "$RUN_LOG_FILE"
touch "$GLOBAL_LOG_FILE"

log "Writing run output to $RUN_LOG_FILE and $GLOBAL_LOG_FILE"
exec > >(tee -a "$RUN_LOG_FILE" "$GLOBAL_LOG_FILE") 2>&1

log "Run name: $RUN_NAME"
log "Results root: $RESULTS_ROOT"

kill_descendants() {
  local parent="$1"
  local child
  local children
  children="$(ps -eo pid=,ppid= | awk -v parent_pid="$parent" '$2 == parent_pid {print $1}')"
  for child in $children; do
    kill_descendants "$child"
    kill -TERM "$child" 2>/dev/null || true
  done
}

handle_interrupt() {
  log "Received interrupt. Stopping all running jobs..."
  kill_descendants "$$"
  exit 130
}

trap 'handle_interrupt' INT TERM

run_step() {
  local name="$1"
  shift
  log "START $name"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  log "DONE  $name"
}

run_step_if_missing() {
  local name="$1"
  shift

  local -a required_files=()
  local -a cmd=()

  while (($#)); do
    if [[ "$1" == "--" ]]; then
      shift
      cmd=( "$@" )
      break
    fi
    required_files+=( "$1" )
    shift
  done

  if [[ -z "${cmd[*]-}" ]]; then
    log "Invalid invocation of run_step_if_missing for $name: command separator -- is required."
    return 1
  fi

  if [[ "$SKIP_EXISTING_ARTIFACTS" == "1" ]] && (( ${#required_files[@]} > 0 )); then
    if has_files "${required_files[@]}"; then
      log "SKIP  $name (cached artifacts already exist)"
      return 0
    fi
  fi

  run_step "$name" "${cmd[@]}"
}

csv_has_columns() {
  local path="$1"
  shift
  if [[ ! -f "$path" ]]; then
    echo "$path: file_not_found"
    return 1
  fi

  local required_columns=("$@")
  local missing
  missing="$("$PYTHON" - "$path" "${required_columns[@]}" <<'PY'
import csv
import sys

path = sys.argv[1]
required = sys.argv[2:]

with open(path, newline="", encoding="utf-8") as f:
    header = next(csv.reader(f), [])

missing = [name for name in required if name not in header]
if missing:
    print(",".join(missing))
    raise SystemExit(1)
PY
)"
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "${missing:-missing_required_columns}"
    return 1
  fi
  return 0
}

csv_has_rows() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "file_not_found"
    return 1
  fi

  local row_count
  row_count="$("$PYTHON" - "$path" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader, None)
    if header is None:
        print(0)
        raise SystemExit(1)
    for i, _ in enumerate(reader, start=1):
        if i >= 1:
            print(i)
            break
    else:
        print(0)
        raise SystemExit(1)
PY
)"
  local rc=$?
  if [[ $rc -ne 0 ]] || [[ "${row_count}" -le 0 ]]; then
    echo "${row_count:-0}"
    return 1
  fi
  return 0
}

validate_csv_columns() {
  local path="$1"
  local description="$2"
  shift 2
  local missing

  if missing="$(csv_has_columns "$path" "$@")"; then
    return 0
  fi
  log "Invalid cached file: $description is missing required column(s): $missing"
  return 1
}

validate_csv_columns_and_rows() {
  local path="$1"
  local description="$2"
  shift 2
  local missing

  if missing="$(csv_has_columns "$path" "$@")"; then
    local row_count
    if row_count="$(csv_has_rows "$path")"; then
      return 0
    fi
    log "Invalid cached file: $description has no data rows (rows: ${row_count:-0})."
    return 1
  fi
  log "Invalid cached file: $description is missing required column(s): $missing"
  return 1
}

ensure_preprocessed_dataset() {
  local path="$1"
  local description="$2"
  shift 2
  local -a required=("$@")

  if [[ "$SKIP_PREPROCESS" == "1" ]]; then
    require_file "$path" "$description"
    if ! validate_csv_columns_and_rows "$path" "$description" "${required[@]}"; then
      return 1
    fi
    return 0
  fi

  if ! validate_csv_columns_and_rows "$path" "$description" "${required[@]}"; then
    log "Regenerating $description from preprocessing step."
    return 2
  fi
  return 0
}

has_files() {
  for required in "$@"; do
    if [[ ! -f "$required" ]]; then
      return 1
    fi
  done
  return 0
}

require_file() {
  local path="$1"
  local description="$2"
  if [[ ! -f "$path" ]]; then
    log "Missing required $description: $path"
    return 1
  fi
}

if [[ "$REQUIRE_GPU" == "1" ]]; then
  run_step check_torch_cuda "$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required but torch.cuda.is_available() is False")
print("CUDA available:", torch.cuda.get_device_name(0))
PY
fi

run_step write_gpu_model_configs "$PYTHON" - "$CONFIG_DIR" <<'PY'
from pathlib import Path
import shutil
import sys
import yaml

target = Path(sys.argv[1])
source = Path("configs/models")
target.mkdir(parents=True, exist_ok=True)
for path in source.glob("*.yaml"):
    shutil.copy2(path, target / path.name)

def update_yaml(name, updater):
    path = target / name
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    updater(data.setdefault("model", {}).setdefault("params", {}))
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

update_yaml("xgboost.yaml", lambda p: (p.update({"device": "cuda", "tree_method": "hist"})))
update_yaml("catboost.yaml", lambda p: p.update({"task_type": "GPU", "devices": "0"}))
# LightGBM GPU availability depends on the local build. Enable by setting
# ENABLE_LIGHTGBM_GPU=1 before running this script if your package supports it.
import os
if os.getenv("ENABLE_LIGHTGBM_GPU", "0") == "1":
    update_yaml("lightgbm.yaml", lambda p: p.update({"device_type": "gpu"}))

print(f"Wrote GPU-aware model configs to {target}")
PY

if [[ "$RUN_PYTEST" == "1" ]]; then
  run_step pytest_preflight "$PYTHON" -m pytest \
    --ignore results \
    --basetemp "$PYTEST_BASE_TEMP_DIR" \
    -o cache_dir="$PYTEST_CACHE_DIR"
fi

if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  run_step_if_missing cohort_selection \
    "$COHORT_PATH" "$COHORT_IDS_PATH" \
    -- "$PYTHON" src/cohort_selection.py
  run_step_if_missing feature_extraction_baseline \
    "$BASELINE_FEATURES_PATH" "$BASELINE_FEATURE_STATS_PATH" \
    -- "$PYTHON" src/feature_extraction.py --disable-expanded-features
  if ! validate_csv_columns_and_rows "$BASELINE_FEATURES_PATH" "baseline extracted features" "subject_id" "hadm_id" "stay_id" "mortality"; then
    log "Regenerating baseline extracted features due to invalid cached output."
    if [[ "$SKIP_PREPROCESS" == "1" ]]; then
      exit 1
    fi
    run_step feature_extraction_baseline_repair \
      "$PYTHON" src/feature_extraction.py --disable-expanded-features
  fi
  run_step_if_missing feature_extraction_expanded \
    "$EXPANDED_FEATURES_PATH" "$EXPANDED_FEATURE_STATS_PATH" \
    -- "$PYTHON" src/feature_extraction.py --enable-expanded-features
  if ! validate_csv_columns_and_rows "$EXPANDED_FEATURES_PATH" "expanded extracted features" "subject_id" "hadm_id" "stay_id" "mortality"; then
    log "Regenerating expanded extracted features due to invalid cached output."
    if [[ "$SKIP_PREPROCESS" == "1" ]]; then
      exit 1
    fi
    run_step feature_extraction_expanded_repair \
      "$PYTHON" src/feature_extraction.py --enable-expanded-features
  fi
  run_step_if_missing preprocess_baseline_all \
    "$BASELINE_XGBOOST_PATH" \
    -- "$PYTHON" src/data_preprocessing.py \
    --input-path "$BASELINE_FEATURES_PATH" \
    --model-type xgboost \
    --output-path "$BASELINE_XGBOOST_PATH" \
    --report-dir "$RESULTS_ROOT/preprocess_reports/baseline"
  preproc_status=0
  ensure_preprocessed_dataset "$BASELINE_XGBOOST_PATH" "baseline preprocessed XGBoost features" "$TARGET_COL" || preproc_status=$?
  if [[ $preproc_status -eq 2 ]]; then
    run_step preprocess_baseline_all_repair \
      "$PYTHON" src/data_preprocessing.py \
      --input-path "$BASELINE_FEATURES_PATH" \
      --model-type xgboost \
      --output-path "$BASELINE_XGBOOST_PATH" \
      --report-dir "$RESULTS_ROOT/preprocess_reports/baseline"
  elif [[ $preproc_status -ne 0 ]]; then
    exit 1
  fi
  run_step_if_missing preprocess_expanded_xgboost \
    "$EXPANDED_XGBOOST_PATH" \
    -- "$PYTHON" src/data_preprocessing.py \
    --input-path "$EXPANDED_FEATURES_PATH" \
    --output-path "$EXPANDED_XGBOOST_PATH" \
    --model-type xgboost \
    --report-dir "$RESULTS_ROOT/preprocess_reports/expanded_xgboost"
  preproc_status=0
  ensure_preprocessed_dataset "$EXPANDED_XGBOOST_PATH" "expanded preprocessed XGBoost features" "$TARGET_COL" || preproc_status=$?
  if [[ $preproc_status -eq 2 ]]; then
    run_step preprocess_expanded_xgboost_repair \
      "$PYTHON" src/data_preprocessing.py \
      --input-path "$EXPANDED_FEATURES_PATH" \
      --output-path "$EXPANDED_XGBOOST_PATH" \
      --model-type xgboost \
      --report-dir "$RESULTS_ROOT/preprocess_reports/expanded_xgboost"
  elif [[ $preproc_status -ne 0 ]]; then
    exit 1
  fi
else
  log "Skipping preprocessing because SKIP_PREPROCESS=1"
fi

if [[ "$RUN_MODEL_SUITE" == "1" ]]; then
  require_file "$EXPANDED_XGBOOST_PATH" "expanded preprocessed XGBOOST features"
  if ! validate_csv_columns_and_rows "$EXPANDED_XGBOOST_PATH" "expanded preprocessed XGBoost features" "$TARGET_COL"; then
    if [[ "$SKIP_PREPROCESS" == "1" ]]; then
      log "SKIP_PREPROCESS=1: cannot repair missing target column."
      exit 1
    fi
    log "Repairing expanded preprocessed data before model suite."
    run_step preprocess_expanded_xgboost_repair \
      "$PYTHON" src/data_preprocessing.py \
      --input-path "$EXPANDED_FEATURES_PATH" \
      --output-path "$EXPANDED_XGBOOST_PATH" \
      --model-type xgboost \
      --report-dir "$RESULTS_ROOT/preprocess_reports/expanded_xgboost"
    if ! validate_csv_columns_and_rows "$EXPANDED_XGBOOST_PATH" "expanded preprocessed XGBoost features" "$TARGET_COL"; then
      exit 1
    fi
  fi
  run_step_if_missing model_suite_gpu \
    "$RESULTS_ROOT/model_suite/model_suite_results.csv" \
    "$RESULTS_ROOT/model_suite/model_comparison_table.csv" \
    -- "$PYTHON" src/experiments/run_model_suite.py \
    --data-path "$EXPANDED_XGBOOST_PATH" \
    --target-col "$TARGET_COL" \
    --output-dir "$RESULTS_ROOT/model_suite" \
    --config-dir "$CONFIG_DIR" \
    --seed "$SEED" \
    --bootstrap-iterations "$BOOTSTRAP_ITERATIONS"
fi

if [[ "$RUN_XGBOOST_ABLATION" == "1" ]]; then
  require_file "$BASELINE_XGBOOST_PATH" "baseline preprocessed XGBoost features"
  require_file "$EXPANDED_XGBOOST_PATH" "expanded preprocessed XGBoost features"
  if ! validate_csv_columns_and_rows "$BASELINE_XGBOOST_PATH" "baseline preprocessed XGBoost features" "$TARGET_COL" ||
    ! validate_csv_columns_and_rows "$EXPANDED_XGBOOST_PATH" "expanded preprocessed XGBoost features" "$TARGET_COL"; then
    if [[ "$SKIP_PREPROCESS" == "1" ]]; then
      log "SKIP_PREPROCESS=1: cannot repair missing target column(s) in cached files."
      exit 1
    fi
    if ! validate_csv_columns_and_rows "$BASELINE_XGBOOST_PATH" "baseline preprocessed XGBoost features" "$TARGET_COL"; then
      log "Repairing baseline preprocessed data before ablation."
      run_step preprocess_baseline_all_repair \
        "$PYTHON" src/data_preprocessing.py \
        --input-path "$BASELINE_FEATURES_PATH" \
        --model-type xgboost \
        --output-path "$BASELINE_XGBOOST_PATH" \
        --report-dir "$RESULTS_ROOT/preprocess_reports/baseline"
      if ! validate_csv_columns_and_rows "$BASELINE_XGBOOST_PATH" "baseline preprocessed XGBOOST features" "$TARGET_COL"; then
        exit 1
      fi
    fi
    if ! validate_csv_columns_and_rows "$EXPANDED_XGBOOST_PATH" "expanded preprocessed XGBoost features" "$TARGET_COL"; then
      log "Repairing expanded preprocessed data before ablation."
      run_step preprocess_expanded_xgboost_repair \
        "$PYTHON" src/data_preprocessing.py \
        --input-path "$EXPANDED_FEATURES_PATH" \
        --output-path "$EXPANDED_XGBOOST_PATH" \
        --model-type xgboost \
        --report-dir "$RESULTS_ROOT/preprocess_reports/expanded_xgboost"
      if ! validate_csv_columns_and_rows "$EXPANDED_XGBOOST_PATH" "expanded preprocessed XGBoost features" "$TARGET_COL"; then
        exit 1
      fi
    fi
  fi
  run_step_if_missing xgboost_baseline_vs_expanded_cuda \
    "$RESULTS_ROOT/xgboost_expanded_ablation/ablation_threshold_policy_results.csv" \
    -- "$PYTHON" tools/run_xgboost_ablation.py \
    --baseline-data-path "$BASELINE_XGBOOST_PATH" \
    --expanded-data-path "$EXPANDED_XGBOOST_PATH" \
    --output-dir "$RESULTS_ROOT/xgboost_expanded_ablation" \
    --device cuda \
    --n-estimators "$XGBOOST_ABLATION_ESTIMATORS" \
    --early-stopping-rounds "$XGBOOST_EARLY_STOPPING_ROUNDS"
fi

if [[ "$RUN_LEGACY_XGBOOST_ENSEMBLE" == "1" ]]; then
  require_file "$BASELINE_XGBOOST_PATH" "baseline preprocessed XGBoost features"
  if ! validate_csv_columns_and_rows "$BASELINE_XGBOOST_PATH" "baseline preprocessed XGBoost features" "$TARGET_COL"; then
    if [[ "$SKIP_PREPROCESS" == "1" ]]; then
      log "SKIP_PREPROCESS=1: cannot repair missing target column in baseline features."
      exit 1
    fi
    log "Repairing baseline preprocessed data before legacy ensemble."
    run_step preprocess_baseline_all_repair \
      "$PYTHON" src/data_preprocessing.py \
      --input-path "$BASELINE_FEATURES_PATH" \
      --model-type xgboost \
      --output-path "$BASELINE_XGBOOST_PATH" \
      --report-dir "$RESULTS_ROOT/preprocess_reports/baseline"
    if ! validate_csv_columns_and_rows "$BASELINE_XGBOOST_PATH" "baseline preprocessed XGBOOST features" "$TARGET_COL"; then
      exit 1
    fi
  fi
  run_step_if_missing legacy_xgboost_ensemble_cuda \
    "$RESULTS_ROOT/legacy_xgboost_ensemble/ensemble_threshold_results.csv" \
    -- "$PYTHON" src/main.py \
    --model xgboost_ensemble \
    --data-path "$BASELINE_XGBOOST_PATH" \
    --output-dir "$RESULTS_ROOT/legacy_xgboost_ensemble" \
    --ensemble-size "$LEGACY_XGBOOST_ENSEMBLE_SIZE" \
    --no-tune
fi

if [[ "$RUN_MAFNET" == "1" || "$RUN_MAFNET_ABLATIONS" == "1" ]]; then
  require_file "$MAFNET_EVENTS_PATH" "MAFNet long-form event CSV"
  require_file "$MAFNET_COHORT_PATH" "MAFNet cohort CSV"
  require_file "$MAFNET_FEATURES_PATH" "MAFNet feature CSV"
fi

if [[ "$RUN_MAFNET" == "1" ]]; then
  run_step_if_missing mafnet_full_cuda \
    "$RESULTS_ROOT/mafnet_full/validation_metrics.json" \
    "$RESULTS_ROOT/mafnet_full/test_metrics.json" \
    -- "$PYTHON" src/training/train_mafnet.py \
    --events-path "$MAFNET_EVENTS_PATH" \
    --cohort-path "$MAFNET_COHORT_PATH" \
    --features-path "$MAFNET_FEATURES_PATH" \
    --config configs/mafnet.yaml \
    --output-dir "$RESULTS_ROOT/mafnet_full" \
    --device cuda \
    --evaluate-test
fi

if [[ "$RUN_MAFNET_ABLATIONS" == "1" ]]; then
  run_step_if_missing mafnet_ablations_cuda \
    "$RESULTS_ROOT/mafnet_ablations/mafnet_ablation_results.csv" \
    -- "$PYTHON" src/experiments/run_mafnet_ablations.py \
    --events-path "$MAFNET_EVENTS_PATH" \
    --cohort-path "$MAFNET_COHORT_PATH" \
    --features-path "$MAFNET_FEATURES_PATH" \
    --config configs/mafnet.yaml \
    --output-dir "$RESULTS_ROOT/mafnet_ablations" \
    --device cuda \
    --evaluate-test
fi

if [[ "$RUN_FINAL_SUMMARY" == "1" && -d "$RESULTS_ROOT/model_suite" ]]; then
  run_step_if_missing final_summary \
    "$RESULTS_ROOT/final_summary.md" \
    -- "$PYTHON" src/experiments/generate_final_summary.py \
    --model-suite-dir "$RESULTS_ROOT/model_suite" \
    --output-path "$RESULTS_ROOT/final_summary.md"
fi

run_step collect_training_results "$PYTHON" - "$RESULTS_ROOT" "$RUN_NAME" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

results_root = Path(sys.argv[1])
run_name = sys.argv[2]
records: list[dict] = []

def add_csv(stage: str, path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    df.insert(0, "run_name", run_name)
    df.insert(1, "stage", stage)
    df.insert(2, "source_file", str(path))
    records.extend(df.to_dict(orient="records"))

add_csv("model_suite_results", results_root / "model_suite" / "model_suite_results.csv")
add_csv("model_suite_comparison", results_root / "model_suite" / "model_comparison_table.csv")
add_csv(
    "xgboost_expanded_ablation",
    results_root / "xgboost_expanded_ablation" / "ablation_threshold_policy_results.csv",
)
add_csv("mafnet_ablations", results_root / "mafnet_ablations" / "mafnet_ablation_results.csv")
add_csv("legacy_xgboost_ensemble", results_root / "legacy_xgboost_ensemble" / "ensemble_threshold_results.csv")

for name in ["validation_metrics.json", "test_metrics.json"]:
    path = results_root / "mafnet_full" / name
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(
            {
                "run_name": run_name,
                "stage": f"mafnet_full_{path.stem}",
                "source_file": str(path),
                "model_name": "mafnet",
            }
        )
        records.append(payload)

combined = pd.DataFrame(records)
combined_path = results_root / "all_training_results.csv"
combined.to_csv(combined_path, index=False)

index_rows = []
for csv_path in sorted(results_root.rglob("*.csv")):
    if csv_path.name == "all_training_results.csv":
        continue
    index_rows.append({"run_name": run_name, "csv_file": str(csv_path)})
pd.DataFrame(index_rows).to_csv(results_root / "result_csv_index.csv", index=False)

manifest = {
    "run_name": run_name,
    "results_root": str(results_root),
    "combined_results_csv": str(combined_path),
    "result_csv_index": str(results_root / "result_csv_index.csv"),
    "n_combined_rows": int(len(combined)),
}
(results_root / "overnight_manifest.json").write_text(
    json.dumps(manifest, indent=2, allow_nan=True),
    encoding="utf-8",
)
print(json.dumps(manifest, indent=2))
PY

log "Overnight training run finished."
log "Combined comparison CSV: $RESULTS_ROOT/all_training_results.csv"
log "CSV index: $RESULTS_ROOT/result_csv_index.csv"
