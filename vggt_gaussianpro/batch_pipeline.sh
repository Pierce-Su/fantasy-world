#!/usr/bin/env bash
# =============================================================================
# FantasyWorld — VGGT + GaussianPro lifting pipeline (batch / multi-scene)
#
# Auto-discovers all scenes under DATAROOT that have a stage1/ folder,
# or processes a specific list of named scenes, and runs the pipeline
# (Stages 1b–4) sequentially.
#
# Layout expected under DATAROOT:
#   DATAROOT/
#     {scene_name}/
#       stage1/
#         video.mp4           (required)
#         debug.pth           (optional, recommended)
#
# Usage:
#   # Process named scenes
#   DATAROOT=./runs SCENE_NAMES="forest_360 castle_orbit" bash batch_pipeline.sh
#
#   # Auto-discover all scenes under ./runs that have a stage1/ folder
#   DATAROOT=./runs bash batch_pipeline.sh
#
# On completion prints:
#   Batch complete. success=N failed=M
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# Configurable variables (override via environment)
# --------------------------------------------------------------------------- #
DATAROOT="${DATAROOT:-./runs}"
# Space-separated list of scene names.  When empty, all {DATAROOT}/*/stage1/
# directories are auto-discovered.
SCENE_NAMES="${SCENE_NAMES:-}"

# Camera trajectory JSON (applied to all scenes).
CAMERA_JSON="${CAMERA_JSON:-../examples/cameras/camera_data_360_orbit.json}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-480}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-832}"
NUM_FRAMES="${NUM_FRAMES:-}"

DEVICE="${DEVICE:-cuda:0}"

# --- Stage 2 (VGGT) ---
VGGT_MODE="${VGGT_MODE:-pose_conditioned}"
VGGT_MAX_FRAMES="${VGGT_MAX_FRAMES:-48}"
VGGT_CHECKPOINT="${VGGT_CHECKPOINT:-}"

# --- Stage 3b (GaussianPro) ---
GP_ITER="${GP_ITER:-30000}"
GP_LAMBDA_LPIPS="${GP_LAMBDA_LPIPS:-0.3}"
USE_DEPTH_PRIOR="${USE_DEPTH_PRIOR:-1}"
SKIP_RENDER="${SKIP_RENDER:-1}"    # skip render by default in batch mode
SKIP_EVAL="${SKIP_EVAL:-1}"        # skip eval by default in batch mode

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "  FantasyWorld VGGT + GaussianPro Batch Pipeline"
echo "  dataroot     : ${DATAROOT}"
echo "  scene_names  : ${SCENE_NAMES:-auto-discover}"
echo "  vggt_mode    : ${VGGT_MODE}"
echo "  max_frames   : ${VGGT_MAX_FRAMES}"
echo "  device       : ${DEVICE}"
echo "  GP iter      : ${GP_ITER}"
echo "  depth prior  : ${USE_DEPTH_PRIOR:-disabled}"
echo "  render       : ${SKIP_RENDER:+skipped}"
echo "============================================================"

SUCCESS=0
FAIL=0

# --------------------------------------------------------------------------- #
# Build scene list
# --------------------------------------------------------------------------- #
declare -a SCENES=()

if [[ -n "${SCENE_NAMES}" ]]; then
    for name in ${SCENE_NAMES}; do
        STAGE1_PATH="${DATAROOT}/${name}/stage1"
        if [[ -d "${STAGE1_PATH}" ]]; then
            SCENES+=("${DATAROOT}/${name}")
        else
            echo "[batch] WARNING: stage1/ not found for '${name}' (${STAGE1_PATH}); skipping."
        fi
    done
else
    # Auto-discover: any directory under DATAROOT that contains stage1/
    for candidate in "${DATAROOT}"/*/; do
        [[ -d "${candidate}stage1" ]] && SCENES+=("${candidate%/}")
    done
fi

if [[ ${#SCENES[@]} -eq 0 ]]; then
    echo "[batch] No scenes found under ${DATAROOT}. Exiting."
    exit 0
fi

echo "[batch] Processing ${#SCENES[@]} scene(s)."

# --------------------------------------------------------------------------- #
# Per-scene pipeline
# --------------------------------------------------------------------------- #
for SCENE_WORKSPACE in "${SCENES[@]}"; do
    SCENE_NAME="$(basename "${SCENE_WORKSPACE}")"
    STAGE1_DIR="${SCENE_WORKSPACE}/stage1"

    echo ""
    echo "------------------------------------------------------------"
    echo "  Scene     : ${SCENE_NAME}"
    echo "  Workspace : ${SCENE_WORKSPACE}"
    echo "------------------------------------------------------------"

    # Pass per-scene variables to the single-scene pipeline.sh
    env \
        STAGE1_DIR="${STAGE1_DIR}"         \
        WORKSPACE="${SCENE_WORKSPACE}"     \
        CAMERA_JSON="${CAMERA_JSON}"       \
        IMAGE_SIZE_H="${IMAGE_SIZE_H}"     \
        IMAGE_SIZE_W="${IMAGE_SIZE_W}"     \
        NUM_FRAMES="${NUM_FRAMES}"         \
        DEVICE="${DEVICE}"                 \
        VGGT_MODE="${VGGT_MODE}"           \
        VGGT_MAX_FRAMES="${VGGT_MAX_FRAMES}" \
        VGGT_CHECKPOINT="${VGGT_CHECKPOINT}" \
        GP_ITER="${GP_ITER}"               \
        GP_LAMBDA_LPIPS="${GP_LAMBDA_LPIPS}" \
        USE_DEPTH_PRIOR="${USE_DEPTH_PRIOR}" \
        SKIP_RENDER="${SKIP_RENDER}"       \
        SKIP_EVAL="${SKIP_EVAL}"           \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        bash "${SCRIPT_DIR}/pipeline.sh" \
    && {
        (( SUCCESS++ ))
        echo "  ✓ ${SCENE_NAME} complete."
    } || {
        (( FAIL++ ))
        echo "  ✗ ${SCENE_NAME} FAILED."
    }
done

echo ""
echo "============================================================"
echo "  Batch complete.  success=${SUCCESS}  failed=${FAIL}"
echo "============================================================"
