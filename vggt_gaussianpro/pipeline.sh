#!/usr/bin/env bash
# =============================================================================
# FantasyWorld — VGGT + GaussianPro lifting pipeline (single scene, post-Stage-1)
#
# Accepts an existing FantasyWorld output directory and orchestrates
# Stages 1b through 4.  Stage 1 (FantasyWorld inference) must be run
# separately before calling this script.
#
# Usage:
#   STAGE1_DIR=runs/forest_360/stage1 bash pipeline.sh
#
#   STAGE1_DIR=runs/forest_360/stage1 \
#   VGGT_MODE=pose_conditioned       \
#   GP_ITER=30000                    \
#   USE_DEPTH_PRIOR=1                \
#   CAMERA_JSON=../examples/cameras/camera_data_360_orbit.json \
#   bash pipeline.sh
#
# Stages:
#   1b) export_stage1.py     — extract frames + NPZs from video.mp4 / debug.pth
#   2)  run_vggt.py          — VGGT geometry estimation
#   3a) colmap_adapter.py    — VGGT NPZs → COLMAP text layout
#   3b) gaussianpro_train.py — GaussianPro optimization (30k iterations)
#   4)  gaussianpro_render.py — render trained views (optional)
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Configurable variables (override via environment)
# --------------------------------------------------------------------------- #
STAGE1_DIR="${STAGE1_DIR:?'STAGE1_DIR must be set to the Stage 1 output directory'}"
# Parent workspace; stages 2 and 3 write under here.
WORKSPACE="${WORKSPACE:-$(dirname "${STAGE1_DIR}")}"

# Camera trajectory JSON (used for JSON-fallback pose export in Stage 1b).
CAMERA_JSON="${CAMERA_JSON:-../examples/cameras/camera_data_360_orbit.json}"

# Image resolution for principal-point derivation when poses come from JSON.
IMAGE_SIZE_H="${IMAGE_SIZE_H:-480}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-832}"

# Number of frames to extract from video.mp4 (empty = all frames).
NUM_FRAMES="${NUM_FRAMES:-}"

DEVICE="${DEVICE:-cuda:0}"

# --- Stage 2 (VGGT) ---
VGGT_MODE="${VGGT_MODE:-pose_conditioned}"        # pose_free | pose_conditioned
VGGT_MAX_FRAMES="${VGGT_MAX_FRAMES:-48}"
VGGT_CHECKPOINT="${VGGT_CHECKPOINT:-}"            # empty = auto-download

# --- Stage 3b (GaussianPro) ---
GP_ITER="${GP_ITER:-30000}"
GP_LAMBDA_LPIPS="${GP_LAMBDA_LPIPS:-0.3}"
USE_DEPTH_PRIOR="${USE_DEPTH_PRIOR:-1}"           # set to "" to disable
SKIP_RENDER="${SKIP_RENDER:-}"                    # set to "1" to skip Stage 4
SKIP_EVAL="${SKIP_EVAL:-}"                        # set to "1" to omit --eval

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Derived paths
STAGE2_DIR="${WORKSPACE}/stage2"
STAGE3_DIR="${WORKSPACE}/stage3"
COLMAP_DIR="${STAGE3_DIR}/colmap"

SUFFIX=""
[[ -n "${USE_DEPTH_PRIOR}" ]] && SUFFIX="_depth_prior"
MODEL_DIR="${STAGE3_DIR}/output_${GP_ITER}_gp${SUFFIX}"

echo "============================================================"
echo "  FantasyWorld VGGT + GaussianPro Pipeline"
echo "  stage1_dir   : ${STAGE1_DIR}"
echo "  workspace    : ${WORKSPACE}"
echo "  camera_json  : ${CAMERA_JSON}"
echo "  image_size   : ${IMAGE_SIZE_H}x${IMAGE_SIZE_W}"
echo "  vggt_mode    : ${VGGT_MODE}"
echo "  max_frames   : ${VGGT_MAX_FRAMES}"
echo "  device       : ${DEVICE}"
echo "  GP iter      : ${GP_ITER}"
echo "  depth prior  : ${USE_DEPTH_PRIOR:-disabled}"
echo "  render       : ${SKIP_RENDER:+skipped}"
echo "============================================================"

# --------------------------------------------------------------------------- #
# Stage 1b — export frames + NPZs from FantasyWorld output
# --------------------------------------------------------------------------- #
echo ""
echo "[Stage 1b] Exporting Stage 1 artifacts …"

EXPORT_CMD=(
    python "${SCRIPT_DIR}/export_stage1.py"
    --stage1_dir  "${STAGE1_DIR}"
    --image_size  "${IMAGE_SIZE_H}" "${IMAGE_SIZE_W}"
)
[[ -f "${CAMERA_JSON}" ]] && EXPORT_CMD+=(--camera_json "${CAMERA_JSON}")
[[ -n "${NUM_FRAMES}" ]]  && EXPORT_CMD+=(--num_frames "${NUM_FRAMES}")

"${EXPORT_CMD[@]}"
echo "[Stage 1b] Done."

# --------------------------------------------------------------------------- #
# Stage 2 — VGGT geometry estimation
# --------------------------------------------------------------------------- #
echo ""
echo "[Stage 2] Running VGGT …"

VGGT_CMD=(
    python "${SCRIPT_DIR}/run_vggt.py"
    --stage1_dir  "${STAGE1_DIR}"
    --stage2_dir  "${STAGE2_DIR}"
    --mode        "${VGGT_MODE}"
    --device      "${DEVICE}"
    --max_frames  "${VGGT_MAX_FRAMES}"
)
[[ -n "${VGGT_CHECKPOINT}" ]] && VGGT_CMD+=(--checkpoint "${VGGT_CHECKPOINT}")

"${VGGT_CMD[@]}"
echo "[Stage 2] Done. Outputs in: ${STAGE2_DIR}"

# --------------------------------------------------------------------------- #
# Stage 3a — COLMAP adapter
# --------------------------------------------------------------------------- #
echo ""
echo "[Stage 3a] Running COLMAP adapter …"

python "${SCRIPT_DIR}/colmap_adapter.py" \
    --stage1_dir  "${STAGE1_DIR}"   \
    --stage2_dir  "${STAGE2_DIR}"   \
    --colmap_dir  "${COLMAP_DIR}"

echo "[Stage 3a] Done. COLMAP layout in: ${COLMAP_DIR}"

# --------------------------------------------------------------------------- #
# Stage 3b — GaussianPro optimisation
# --------------------------------------------------------------------------- #
echo ""
echo "[Stage 3b] Running GaussianPro …"

GP_CMD=(
    python "${SCRIPT_DIR}/gaussianpro_train.py"
    --colmap_dir  "${COLMAP_DIR}"
    --model_dir   "${MODEL_DIR}"
    --iter        "${GP_ITER}"
    --device      "${DEVICE}"
)
[[ -n "${USE_DEPTH_PRIOR}" ]] && GP_CMD+=(--use_depth_prior)
[[ -z "${SKIP_EVAL}" ]]       && GP_CMD+=(--eval)

"${GP_CMD[@]}"
echo "[Stage 3b] Done."

# --------------------------------------------------------------------------- #
# Stage 4 — render (optional)
# --------------------------------------------------------------------------- #
if [[ -z "${SKIP_RENDER}" ]]; then
    echo ""
    echo "[Stage 4] Rendering …"

    RENDER_CMD=(
        python "${SCRIPT_DIR}/gaussianpro_render.py"
        --model_dir   "${MODEL_DIR}"
        --colmap_dir  "${COLMAP_DIR}"
        --device      "${DEVICE}"
    )
    [[ -z "${SKIP_EVAL}" ]] && RENDER_CMD+=(--eval)

    "${RENDER_CMD[@]}"
    echo "[Stage 4] Done. Renders in: ${MODEL_DIR}/train/"
fi

echo ""
echo "Pipeline complete."
echo "Outputs: ${WORKSPACE}"
