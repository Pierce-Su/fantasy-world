#!/usr/bin/env bash
# Batch FantasyWorld Wan 2.1 video generation over data/curated_set.
# Generates prompts on the fly via generate_prompt.py (metadata lookup).
#
# Usage (from repo root):
#   bash batch_inference_wan21.sh
#   DATA_DIR=./data/curated_set/photorealistic bash batch_inference_wan21.sh
#   MAX_IMAGES=5 bash batch_inference_wan21.sh          # smoke test
#   FORCE=1 bash batch_inference_wan21.sh               # redo existing outputs
#   DRY_RUN=1 bash batch_inference_wan21.sh             # print commands only
#   FAIL_FAST=1 bash batch_inference_wan21.sh           # abort on first failure
#   FORCE=1 overwrite existing generated video.mp4
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---------------------------------------------------------------------------
# Config (override via environment)
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-./data/curated_set}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output-wan21/curated_set}"
CAMERA_JSON="${CAMERA_JSON:-./examples/cameras/camera_data_360_orbit.json}"
WAN_CKPT="${WAN_CKPT:-./models/Wan-AI/Wan2.1-I2V-14B-480P}"
MODEL_CKPT="${MODEL_CKPT:-./models/FantasyWorld-Wan2.1-I2V-14B-480P/model.pth}"
METADATA="${METADATA:-./data/curated_set/metadata.json}"
SAMPLE_STEPS="${SAMPLE_STEPS:-50}"
USING_SCALE="${USING_SCALE:-True}"
SEED="${SEED:-1024}"
MAX_IMAGES="${MAX_IMAGES:-0}"          # 0 = all
FORCE="${FORCE:-0}"                    # 1 = overwrite existing video.mp4
DRY_RUN="${DRY_RUN:-0}"                # 1 = print commands only
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-./output-wan21/logs}"
FAIL_FAST="${FAIL_FAST:-0}"            # 1 = abort on first failure

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/batch_wan21_$(date +%Y%m%d_%H%M%S).log"

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "${msg}" | tee -a "${RUN_LOG}"
}

die() {
  log "ERROR: $*"
  exit 1
}

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
[[ -d "${DATA_DIR}" ]] || die "DATA_DIR not found: ${DATA_DIR}"
[[ -f "${CAMERA_JSON}" ]] || die "CAMERA_JSON not found: ${CAMERA_JSON}"
[[ -d "${WAN_CKPT}" ]] || die "WAN_CKPT not found: ${WAN_CKPT}"
[[ -f "${MODEL_CKPT}" ]] || die "MODEL_CKPT not found: ${MODEL_CKPT}"
[[ -f "./inference_wan21.py" ]] || die "inference_wan21.py not found (run from repo root)"
[[ -f "./generate_prompt.py" ]] || die "generate_prompt.py not found"

# Collect images (sorted, recursive)
mapfile -t IMAGES < <(
  find "${DATA_DIR}" -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) \
    | sort
)

TOTAL=${#IMAGES[@]}
[[ "${TOTAL}" -gt 0 ]] || die "No images under ${DATA_DIR}"

if [[ "${MAX_IMAGES}" -gt 0 && "${MAX_IMAGES}" -lt "${TOTAL}" ]]; then
  IMAGES=("${IMAGES[@]:0:${MAX_IMAGES}}")
fi

N=${#IMAGES[@]}
log "================================================================"
log " FantasyWorld Wan2.1 batch inference"
log " DATA_DIR     = ${DATA_DIR}"
log " OUTPUT_ROOT  = ${OUTPUT_ROOT}"
log " CAMERA_JSON  = ${CAMERA_JSON}"
log " images       = ${N} / ${TOTAL} discovered"
log " SAMPLE_STEPS = ${SAMPLE_STEPS}"
log " FORCE        = ${FORCE}"
log " DRY_RUN      = ${DRY_RUN}"
log " log          = ${RUN_LOG}"
log "================================================================"

ok=0
skip=0
fail=0
failed_list=()

for idx in "${!IMAGES[@]}"; do
  img="${IMAGES[$idx]}"
  i=$((idx + 1))

  # Preserve relative layout under DATA_DIR inside OUTPUT_ROOT
  # e.g. data/curated_set/photorealistic/0000.jpg -> output-wan21/curated_set/photorealistic/0000/
  rel="${img#${DATA_DIR}/}"
  rel="${rel#./}"
  stem="$(basename "${img}")"
  stem="${stem%.*}"
  parent="$(dirname "${rel}")"
  if [[ "${parent}" == "." ]]; then
    out_dir="${OUTPUT_ROOT}/${stem}"
  else
    out_dir="${OUTPUT_ROOT}/${parent}/${stem}"
  fi
  video_path="${out_dir}/video.mp4"

  if [[ -f "${video_path}" && "${FORCE}" != "1" ]]; then
    log "[${i}/${N}] SKIP (exists): ${rel}"
    skip=$((skip + 1))
    continue
  fi

  mkdir -p "${out_dir}"
  prompt_file="${out_dir}/prompt.txt"
  if [[ -f "${prompt_file}" && "${FORCE}" != "1" ]]; then
    prompt="$(<"${prompt_file}")"
  else
    prompt="$("${PYTHON_BIN}" ./generate_prompt.py \
      --image_path "${img}" \
      --metadata "${METADATA}")"
    printf '%s\n' "${prompt}" > "${prompt_file}"
  fi
  prompt="$(echo "${prompt}" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  [[ -n "${prompt}" ]] || die "Empty prompt for ${img}"
  printf '%s\n' "${prompt}" > "${prompt_file}"

  log "[${i}/${N}] RUN  ${rel}"
  log "         prompt: ${prompt}"
  log "         out:    ${out_dir}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "         (dry-run) skipping inference"
    ok=$((ok + 1))
    continue
  fi

  set +e
  "${PYTHON_BIN}" ./inference_wan21.py \
    --wan_ckpt_path "${WAN_CKPT}" \
    --model_ckpt "${MODEL_CKPT}" \
    --image_path "${img}" \
    --camera_json_path "${CAMERA_JSON}" \
    --prompt "${prompt}" \
    --output_dir "${out_dir}" \
    --sample_steps "${SAMPLE_STEPS}" \
    --using_scale "${USING_SCALE}" \
    --seed "${SEED}" \
    2>&1 | tee -a "${RUN_LOG}"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ "${rc}" -eq 0 && -f "${video_path}" ]]; then
    log "[${i}/${N}] OK   ${video_path}"
    ok=$((ok + 1))
  else
    log "[${i}/${N}] FAIL rc=${rc}  ${rel}"
    fail=$((fail + 1))
    failed_list+=("${rel}")
    if [[ "${FAIL_FAST}" == "1" ]]; then
      die "Fail-fast enabled; aborting after first failure."
    fi
  fi
done

log "================================================================"
log " Done. ok=${ok}  skip=${skip}  fail=${fail}  total=${N}"
if [[ "${fail}" -gt 0 ]]; then
  log " Failed images:"
  for f in "${failed_list[@]}"; do
    log "   - ${f}"
  done
fi
log "================================================================"

[[ "${fail}" -eq 0 ]]
