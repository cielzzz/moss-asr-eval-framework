#!/usr/bin/env sh
set -eu

# Edit the "User settings" section for each run, then execute:
#   sh run_eval.sh
#
# Supported run modes:
#   eval   -> use BENCHMARK / TTS_SYSTEM / ASR_BACKEND below
#   batch  -> run BATCH_TARGETS below sequentially
#   dummy  -> lightweight pipeline regression test
#   custom -> use CUSTOM_MANIFEST / CUSTOM_CONFIG / CUSTOM_OUT_DIR below

# ----------------------------
# User settings
# ----------------------------

# Options: dummy, eval, batch, custom
# Use dummy first to verify installation. Use eval for real ASR evaluation.
RUN_MODE="dummy"

# Options: cv3, seed-tts-eval
BENCHMARK="cv3"

# User-defined TTS system name. It is used in manifest and output paths.
TTS_SYSTEM="your-tts-system"

# Options: qwen-asr-1.7b
ASR_BACKEND="qwen-asr-1.7b"

# TTS inference output root. Set this when adding a new TTS system or regenerating a manifest.
# Expected layout: ${TTS_ROOT}/${task}/${utt_id}/pred.wav
# Example: /path/to/Your-TTS/cv3 or /path/to/Your-TTS/seed-tts
TTS_ROOT=""

# Set BUILD_MANIFEST=1 for a new TTS system. Set it back to 0 after manifest is generated.
BUILD_MANIFEST=0

# Batch target rows: TTS_SYSTEM BENCHMARK.
# Batch mode is sequential, not parallel. Keep one row per run. Lines starting with # are ignored.
BATCH_TARGETS='
your-tts-system cv3
# your-tts-system seed-tts-eval
'
SKIP_COMPLETED=1

# Optional conda auto setup. Most users can activate the env manually and keep ACTIVATE_ENV=0.
ACTIVATE_ENV=0
CONDA_ENV="asr-eval"

# Shared config.
CONFIG="config.example.yaml"

# Custom target settings. These are only used when RUN_MODE="custom".
CUSTOM_MANIFEST="data/manifests/${TTS_SYSTEM}/${BENCHMARK}.asr.jsonl"
CUSTOM_CONFIG="config.example.yaml"
CUSTOM_OUT_DIR="outputs/asr_eval/${BENCHMARK}/${TTS_SYSTEM}/${ASR_BACKEND}"

benchmark_input_jsonl() {
  case "$1" in
    cv3)
      echo "data/benchmarks/cv3-eval.jsonl"
      ;;
    seed-tts-eval)
      echo "data/benchmarks/seed-tts-eval.jsonl"
      ;;
    *)
      echo "Unknown BENCHMARK: $1" >&2
      exit 2
      ;;
  esac
}

case "${RUN_MODE}" in
  eval)
    MANIFEST="data/manifests/${TTS_SYSTEM}/${BENCHMARK}.asr.jsonl"
    OUT_DIR="outputs/asr_eval/${BENCHMARK}/${TTS_SYSTEM}/${ASR_BACKEND}"
    ;;
  batch)
    MANIFEST=""
    OUT_DIR=""
    ;;
  dummy)
    MANIFEST="test_dummy/manifest.jsonl"
    CONFIG="test_dummy/config.yaml"
    OUT_DIR="test_dummy"
    ;;
  custom)
    MANIFEST="${CUSTOM_MANIFEST}"
    CONFIG="${CUSTOM_CONFIG}"
    OUT_DIR="${CUSTOM_OUT_DIR}"
    ;;
  *)
    echo "Unknown RUN_MODE: ${RUN_MODE}" >&2
    exit 2
    ;;
esac

export CONDA_ENV
if [ "${ACTIVATE_ENV}" = "1" ] && [ "${ASR_EVAL_ENV_READY:-0}" != "1" ]; then
  export ASR_EVAL_ENV_READY=1
  exec bash -lc '
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    exec sh run_eval.sh
  '
fi

echo "RUN_MODE=${RUN_MODE}"
echo "BENCHMARK=${BENCHMARK}"
echo "TTS_SYSTEM=${TTS_SYSTEM}"
echo "ASR_BACKEND=${ASR_BACKEND}"
echo "MANIFEST=${MANIFEST}"
echo "CONFIG=${CONFIG}"
echo "TTS_ROOT=${TTS_ROOT}"
echo "BUILD_MANIFEST=${BUILD_MANIFEST}"

export TRANSFORMERS_VERBOSITY=error

build_manifest_if_needed() {
  run_tts_system="$1"
  run_benchmark="$2"
  run_manifest="$3"

  if [ "${BUILD_MANIFEST}" != "1" ]; then
    return 0
  fi

  if [ -z "${TTS_ROOT}" ]; then
    echo "BUILD_MANIFEST=1 requires TTS_ROOT." >&2
    exit 2
  fi

  input_jsonl="$(benchmark_input_jsonl "${run_benchmark}")"
  echo "Build manifest from TTS_ROOT=${TTS_ROOT}"
  python build_tts_manifest.py \
    --input-jsonl "${input_jsonl}" \
    --tts-root "${TTS_ROOT}" \
    --benchmark "${run_benchmark}" \
    --tts-system "${run_tts_system}" \
    --out "${run_manifest}"
}

run_manifest() {
  run_tts_system="$1"
  run_benchmark="$2"
  run_manifest="$3"
  run_out_dir="$4"

  build_manifest_if_needed "${run_tts_system}" "${run_benchmark}" "${run_manifest}"

  if [ ! -f "${run_manifest}" ]; then
    echo "Missing manifest: ${run_manifest}" >&2
    echo "Set BUILD_MANIFEST=1 and TTS_ROOT, or generate it first with build_tts_manifest.py." >&2
    exit 2
  fi

  mkdir -p "${run_out_dir}"

  if [ "${SKIP_COMPLETED}" = "1" ] && [ -s "${run_out_dir}/results.jsonl" ] && [ -s "${run_out_dir}/predictions.jsonl" ]; then
    echo "Skip completed: ${run_benchmark}/${run_tts_system}/${ASR_BACKEND}"
    return 0
  fi

  echo "TTS_SYSTEM=${run_tts_system}"
  echo "BENCHMARK=${run_benchmark}"
  echo "MANIFEST=${run_manifest}"
  echo "OUT_DIR=${run_out_dir}"

  python asr_eval.py \
    --manifest "${run_manifest}" \
    --config "${CONFIG}" \
    --out "${run_out_dir}/results.jsonl" \
    --summary "${run_out_dir}/summary.csv" \
    --continue-on-error

  python export_eval_outputs.py \
    --results "${run_out_dir}/results.jsonl" \
    --out-dir "${run_out_dir}"
}

run_one() {
  run_tts_system="$1"
  run_benchmark="$2"
  run_manifest="data/manifests/${run_tts_system}/${run_benchmark}.asr.jsonl"
  run_out_dir="outputs/asr_eval/${run_benchmark}/${run_tts_system}/${ASR_BACKEND}"
  run_manifest "${run_tts_system}" "${run_benchmark}" "${run_manifest}" "${run_out_dir}"
}

if [ "${RUN_MODE}" = "batch" ]; then
  echo "${BATCH_TARGETS}" | while read -r run_tts_system run_benchmark _rest; do
    case "${run_tts_system:-}" in
      ""|\#*) continue ;;
    esac
    run_one "${run_tts_system}" "${run_benchmark}"
  done
elif [ "${RUN_MODE}" = "custom" ] || [ "${RUN_MODE}" = "dummy" ]; then
  echo "OUT_DIR=${OUT_DIR}"
  run_manifest "${TTS_SYSTEM}" "${BENCHMARK}" "${MANIFEST}" "${OUT_DIR}"
else
  echo "OUT_DIR=${OUT_DIR}"
  run_one "${TTS_SYSTEM}" "${BENCHMARK}"
fi
