"""
Stage 2 — VGGT Geometry Estimation
=====================================
Loads frames from a Stage 1b export directory, runs VGGT to estimate camera
poses, depth maps, and a dense point cloud, then writes structured outputs for
the downstream COLMAP adapter (Stage 3a) and GaussianPro (Stage 3b).

Two operating modes are supported:

  pose_free (default for purely feed-forward use)
      VGGT estimates all camera poses from RGB frames.  When stage1 poses are
      available a quality gate compares the two trajectories; thresholds are
      rotation < 5° mean and translation < 5% of scene radius.

  pose_conditioned (recommended for FantasyWorld orbit sequences)
      VGGT's estimated extrinsics are replaced with the known world-to-camera
      matrices from stage1/poses_w2c.npz.  VGGT depth and point maps are
      retained unchanged (they are geometry-estimated, not diffusion-estimated).

Outputs written to ``--stage2_dir``:

  vggt_poses_w2c.npz      (T, 4, 4) float64 — final poses used
  vggt_intrinsics.npz     (T, 4)    float64 — [fx, fy, cx, cy]
  vggt_depth.npz          (T, H, W) float32 — VGGT depth maps
  vggt_conf.npz           (T, H, W) float32 — VGGT depth confidence
  vggt_points3d.ply       ASCII PLY, coloured, ≤200k points
  depth_maps/{i}.npy      (H, W) float32 per frame
  confidence_maps/{i}.npy (H, W) float32 per frame
  normals/{i}.npy         (3, H, W) float32 per frame, values in [0, 1]
  quality_gate.json       quality gate results (when stage1 poses available)

Usage
-----
    python run_vggt.py \\
        --stage1_dir  runs/my_scene/stage1 \\
        --mode        pose_conditioned \\
        --device      cuda:0

    python run_vggt.py \\
        --stage1_dir  runs/my_scene/stage1 \\
        --stage2_dir  runs/my_scene/stage2 \\
        --mode        pose_free \\
        --max_frames  48 \\
        --checkpoint  checkpoints/vggt_1b.pt
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Make the standalone thirdparty VGGT importable.
# The FantasyWorld repository keeps VGGT at  ../thirdparty/vggt/  relative to
# this script's directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_VGGT_ROOT  = _SCRIPT_DIR.parent / "thirdparty" / "vggt"
if not _VGGT_ROOT.is_dir():
    # Fallback: if running from within the vggt_gaussianpro/ dir that has a
    # local thirdparty/ copy (unlikely for FantasyWorld but keeps script robust)
    _VGGT_ROOT = _SCRIPT_DIR / "thirdparty" / "vggt"
sys.path.insert(0, str(_VGGT_ROOT))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from utils.depth_to_normal import batch_point_map_to_normals

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VGGT_FIXED_RESOLUTION = 518
IMG_LOAD_RESOLUTION   = 1024
CONF_THRES_VALUE      = 5.0
CONF_THRES_FALLBACK_PERCENTILE = 80
MIN_POINTS_BEFORE_FALLBACK     = 1_000
MAX_POINTS_PLY                 = 200_000

# Quality gate thresholds
QG_ROT_THRESH_DEG  = 5.0
QG_TRANS_THRESH_PCT = 5.0


# ---------------------------------------------------------------------------
# PLY export helper
# ---------------------------------------------------------------------------

def save_points_ply(points: np.ndarray, colors: np.ndarray, output_path: Path) -> None:
    """Save a coloured ASCII PLY point cloud."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors).reshape(-1, 3)
    if colors.dtype.kind == "f":
        if float(np.nanmax(colors)) <= 1.0:
            colors = colors * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.7f} {y:.7f} {z:.7f} {int(r)} {int(g)} {int(b)}\n")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: VGGT geometry estimation for the FantasyWorld pipeline."
    )
    parser.add_argument(
        "--stage1_dir", type=str, required=True,
        help="Path to Stage 1b export directory containing frames/ and NPZ artifacts.",
    )
    parser.add_argument(
        "--stage2_dir", type=str, default=None,
        help="Output directory for Stage 2 artifacts "
             "(default: {stage1_dir}/../stage2).",
    )
    parser.add_argument(
        "--mode", type=str, default="pose_conditioned",
        choices=["pose_free", "pose_conditioned"],
        help="VGGT operating mode.  'pose_conditioned' replaces VGGT's estimated "
             "extrinsics with Stage 1 prior poses after the forward pass "
             "(recommended for orbit/spin sequences with debug.pth available). "
             "'pose_free' lets VGGT estimate all poses from RGB frames.",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available).",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a local VGGT-1B checkpoint (.pt).  "
             "Downloaded from HuggingFace when not provided.",
    )
    parser.add_argument(
        "--max_frames", type=int, default=48,
        help="Maximum number of frames fed to VGGT (default: 48).  "
             "Frames are uniformly subsampled when the extracted count exceeds this.",
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=CONF_THRES_VALUE,
        help=f"Absolute depth confidence threshold for PLY export (default: {CONF_THRES_VALUE}).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str | None, device: str) -> VGGT:
    model = VGGT()
    if checkpoint_path and Path(checkpoint_path).is_file():
        print(f"[run_vggt] Loading VGGT weights from: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
    else:
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        print(f"[run_vggt] Downloading VGGT weights from: {_URL}")
        state = torch.hub.load_state_dict_from_url(_URL, map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# VGGT forward pass
# ---------------------------------------------------------------------------

def run_vggt_forward(
    model: VGGT,
    images: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run VGGT aggregator + camera_head + depth_head at VGGT_FIXED_RESOLUTION.

    Args:
        images: (N, 3, H, W) float32 tensor on the correct device.
        dtype:  bfloat16 or float16.

    Returns:
        extrinsic:  (N, 3, 4) float64 — camera-to-world, OpenCV convention.
        intrinsic:  (N, 3, 3) float64 — 3×3 camera intrinsic matrix.
        depth_map:  (N, H_vggt, W_vggt) float32
        depth_conf: (N, H_vggt, W_vggt) float32
    """
    images_vggt = F.interpolate(
        images, size=(VGGT_FIXED_RESOLUTION, VGGT_FIXED_RESOLUTION),
        mode="bilinear", align_corners=False,
    )

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_b = images_vggt[None]  # (1, N, 3, H, W)
            aggregated_tokens_list, ps_idx = model.aggregator(images_b)
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, images_b.shape[-2:]
            )
            depth_map, depth_conf = model.depth_head(
                aggregated_tokens_list, images_b, ps_idx
            )

    extrinsic  = extrinsic.squeeze(0).cpu().double().numpy()   # (N, 3, 4) c2w
    intrinsic  = intrinsic.squeeze(0).cpu().double().numpy()   # (N, 3, 3)
    depth_map  = depth_map.squeeze(0).cpu().float().numpy()    # (N, H, W)
    depth_conf = depth_conf.squeeze(0).cpu().float().numpy()   # (N, H, W)

    return extrinsic, intrinsic, depth_map, depth_conf


# ---------------------------------------------------------------------------
# Pose-conditioned override
# ---------------------------------------------------------------------------

def apply_pose_conditioned(
    extrinsic_vggt: np.ndarray,
    intrinsic_vggt: np.ndarray,
    prior_poses_w2c: np.ndarray,
    prior_intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replace VGGT's estimated extrinsics/intrinsics with Stage 1 prior poses.

    VGGT's depth maps and point cloud are derived from the original forward
    pass and are not modified here.

    Args:
        extrinsic_vggt:  (T, 3, 4) c2w from VGGT (unused after substitution).
        intrinsic_vggt:  (T, 3, 3) intrinsics from VGGT.
        prior_poses_w2c: (T', 4, 4) world-to-camera from stage1.
        prior_intrinsics:(T', 4)    [fx, fy, cx, cy] from stage1.

    Returns:
        extrinsic_out:  (T, 3, 4) camera-to-world (from prior).
        intrinsic_out:  (T, 3, 3) intrinsics (from prior when available).
    """
    T = extrinsic_vggt.shape[0]
    T_prior = prior_poses_w2c.shape[0]

    # Align lengths via uniform resampling if they differ.
    if T_prior != T:
        prior_indices = [round(i * (T_prior - 1) / max(T - 1, 1)) for i in range(T)]
    else:
        prior_indices = list(range(T))

    extrinsic_out = np.zeros((T, 3, 4), dtype=np.float64)
    intrinsic_out = np.zeros((T, 3, 3), dtype=np.float64)

    for i, pi in enumerate(prior_indices):
        w2c = prior_poses_w2c[pi]
        c2w = np.linalg.inv(w2c)
        extrinsic_out[i] = c2w[:3, :]

        fx, fy, cx, cy = prior_intrinsics[pi]
        intrinsic_out[i] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    return extrinsic_out, intrinsic_out


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def run_quality_gate(
    vggt_extrinsic: np.ndarray,
    prior_poses_w2c: np.ndarray,
) -> dict:
    """
    Compare VGGT-estimated poses to Stage 1 prior poses.

    Computes mean rotation error (degrees) and mean translation error
    normalised by the scene radius (percent).

    Args:
        vggt_extrinsic:  (T, 3, 4) c2w from VGGT.
        prior_poses_w2c: (T', 4, 4) w2c from stage1 (will be aligned if T ≠ T').

    Returns:
        dict with keys: rot_error_deg_mean, trans_error_pct_mean, passed.
    """
    T = vggt_extrinsic.shape[0]
    T_prior = prior_poses_w2c.shape[0]

    if T_prior != T:
        prior_indices = [round(i * (T_prior - 1) / max(T - 1, 1)) for i in range(T)]
    else:
        prior_indices = list(range(T))

    rot_errors   = []
    trans_errors = []

    # Collect camera centres for scene radius estimation.
    centres = []
    for i, pi in enumerate(prior_indices):
        w2c = prior_poses_w2c[pi]
        c2w_prior = np.linalg.inv(w2c)
        centres.append(c2w_prior[:3, 3])
    centres = np.array(centres)
    centroid = centres.mean(axis=0)
    scene_radius = float(np.linalg.norm(centres - centroid, axis=1).max())
    scene_radius = max(scene_radius, 1e-6)

    for i, pi in enumerate(prior_indices):
        w2c_prior = prior_poses_w2c[pi]
        c2w_prior = np.linalg.inv(w2c_prior)
        R_prior = c2w_prior[:3, :3]
        t_prior = c2w_prior[:3, 3]

        c2w_vggt = np.eye(4)
        c2w_vggt[:3, :] = vggt_extrinsic[i]
        R_vggt = c2w_vggt[:3, :3]
        t_vggt = c2w_vggt[:3, 3]

        # Rotation error via trace of relative rotation.
        R_rel = R_prior @ R_vggt.T
        cos_theta = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        rot_err_deg = float(np.degrees(np.arccos(cos_theta)))
        rot_errors.append(rot_err_deg)

        trans_err = float(np.linalg.norm(t_prior - t_vggt))
        trans_errors.append(trans_err)

    mean_rot   = float(np.mean(rot_errors))
    mean_trans = float(np.mean(trans_errors)) / scene_radius * 100.0

    passed = (mean_rot < QG_ROT_THRESH_DEG) and (mean_trans < QG_TRANS_THRESH_PCT)

    result = {
        "rot_error_deg_mean":   mean_rot,
        "trans_error_pct_mean": mean_trans,
        "scene_radius":         scene_radius,
        "passed":               passed,
    }

    level = "INFO" if passed else "WARNING"
    print(
        f"[run_vggt] Quality gate [{level}]: "
        f"rot_error={mean_rot:.2f}°  "
        f"trans_error={mean_trans:.2f}%  "
        f"{'PASSED' if passed else 'FAILED (high pose drift)'}"
    )
    return result


# ---------------------------------------------------------------------------
# Frame discovery
# ---------------------------------------------------------------------------

def collect_frame_paths(frames_dir: Path) -> list[Path]:
    """
    Return frame_*.png paths sorted by the integer in their stem.
    """
    paths = sorted(
        frames_dir.glob("frame_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not paths:
        raise FileNotFoundError(f"No frame_*.png found in {frames_dir}")
    return paths


# ---------------------------------------------------------------------------
# Auxiliary outputs
# ---------------------------------------------------------------------------

def save_per_frame_outputs(
    stage2_dir: Path,
    depth_map: np.ndarray,
    depth_conf: np.ndarray,
    points_3d: np.ndarray,
) -> None:
    """
    Write per-frame depth, confidence, and normal .npy files for GaussianPro.

    Args:
        depth_map:  (N, H, W) float32
        depth_conf: (N, H, W) float32
        points_3d:  (N, H, W, 3) float32  world-space point map
    """
    depth_dir = stage2_dir / "depth_maps"
    conf_dir  = stage2_dir / "confidence_maps"
    norm_dir  = stage2_dir / "normals"
    for d in (depth_dir, conf_dir, norm_dir):
        d.mkdir(parents=True, exist_ok=True)

    normals = batch_point_map_to_normals(points_3d)  # (N, H, W, 3) in [-1, 1]

    N = depth_map.shape[0]
    for i in range(N):
        np.save(str(depth_dir / f"{i}.npy"), depth_map[i].astype(np.float32))
        np.save(str(conf_dir  / f"{i}.npy"), depth_conf[i].astype(np.float32))
        # GaussianPro expects channels-first (3, H, W) in [0, 1]; applies
        # (n - 0.5) * 2 to recover [-1, 1] unit normals.
        normal_01 = ((normals[i] + 1.0) / 2.0).astype(np.float32)      # (H, W, 3)
        np.save(str(norm_dir / f"{i}.npy"), np.transpose(normal_01, (2, 0, 1)))  # (3, H, W)

    print(f"[run_vggt] Saved per-frame depth/conf/normals for {N} frames.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    stage1_dir = Path(args.stage1_dir)
    stage2_dir = (
        Path(args.stage2_dir) if args.stage2_dir
        else stage1_dir.parent / "stage2"
    )
    stage2_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    dtype  = torch.bfloat16 if (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ) else torch.float16

    print("=" * 60)
    print("  FantasyWorld — Stage 2: VGGT Geometry Estimation")
    print(f"  stage1_dir  : {stage1_dir}")
    print(f"  stage2_dir  : {stage2_dir}")
    print(f"  mode        : {args.mode}")
    print(f"  device      : {device}  dtype: {dtype}")
    print(f"  max_frames  : {args.max_frames}")
    print("=" * 60)

    # ----------------------------------------------------------- collect frames
    frames_dir = stage1_dir / "frames"
    if not frames_dir.is_dir():
        sys.exit(
            f"[run_vggt] ERROR: frames directory not found: {frames_dir}\n"
            "Run export_stage1.py first."
        )

    frame_paths = collect_frame_paths(frames_dir)
    print(f"[run_vggt] Found {len(frame_paths)} frames in {frames_dir}")

    # --------------------------------------------------- frame subsampling
    total = len(frame_paths)
    if args.max_frames is not None and total > args.max_frames:
        indices = [round(i * (total - 1) / (args.max_frames - 1)) for i in range(args.max_frames)]
        frame_paths = [frame_paths[i] for i in indices]
        print(
            f"[run_vggt] Subsampled {total} → {len(frame_paths)} frames "
            f"(--max_frames={args.max_frames})"
        )

    # --------------------------------------------------- load + preprocess
    images, original_coords = load_and_preprocess_images_square(
        [str(p) for p in frame_paths], IMG_LOAD_RESOLUTION
    )
    images = images.to(device)

    # ----------------------------------------------------------- load model
    model = load_model(args.checkpoint, device)

    # ----------------------------------------------------------- VGGT forward
    print("[run_vggt] Running VGGT forward pass …")
    extrinsic_c2w, intrinsic_3x3, depth_map, depth_conf = run_vggt_forward(
        model, images, dtype
    )
    T = extrinsic_c2w.shape[0]

    # World-space point map (N, H_vggt, W_vggt, 3)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic_c2w, intrinsic_3x3)
    # points_3d may be a tensor; convert to numpy
    if hasattr(points_3d, "cpu"):
        points_3d = points_3d.cpu().numpy()
    points_3d = np.array(points_3d, dtype=np.float32)

    # ------------------------------------------- load stage1 prior poses
    prior_poses_w2c   = None
    prior_intrinsics  = None
    stage1_poses_path = stage1_dir / "poses_w2c.npz"
    stage1_intri_path = stage1_dir / "intrinsics.npz"

    if stage1_poses_path.is_file():
        prior_poses_w2c  = np.load(str(stage1_poses_path))["poses_w2c"]   # (T', 4, 4)
        print(f"[run_vggt] Loaded stage1 poses: {prior_poses_w2c.shape}")
    if stage1_intri_path.is_file():
        prior_intrinsics = np.load(str(stage1_intri_path))["intrinsics"]  # (T', 4)

    # ------------------------------------------------ quality gate (pose_free)
    quality_gate_result = None
    if args.mode == "pose_free" and prior_poses_w2c is not None:
        quality_gate_result = run_quality_gate(extrinsic_c2w, prior_poses_w2c)
        with open(str(stage2_dir / "quality_gate.json"), "w") as f:
            json.dump(quality_gate_result, f, indent=2)

    # ----------------------------------------- pose-conditioned override
    if args.mode == "pose_conditioned":
        if prior_poses_w2c is None:
            print(
                "[run_vggt] WARNING: pose_conditioned mode requested but "
                "stage1/poses_w2c.npz not found.  Falling back to pose_free."
            )
        else:
            if prior_intrinsics is None:
                prior_intrinsics = np.zeros((prior_poses_w2c.shape[0], 4))
                # Fill from VGGT intrinsics as fallback
                for i in range(prior_intrinsics.shape[0]):
                    idx = min(i, T - 1)
                    K = intrinsic_3x3[idx]
                    prior_intrinsics[i] = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]

            extrinsic_c2w, intrinsic_3x3 = apply_pose_conditioned(
                extrinsic_c2w, intrinsic_3x3, prior_poses_w2c, prior_intrinsics
            )
            print(f"[run_vggt] Applied pose-conditioned override for {T} frames.")

            # Run quality gate in pose_conditioned mode too, for diagnostics.
            if prior_poses_w2c is not None:
                quality_gate_result = run_quality_gate(extrinsic_c2w, prior_poses_w2c)
                with open(str(stage2_dir / "quality_gate.json"), "w") as f:
                    json.dump(quality_gate_result, f, indent=2)

    # ----------------------------------------- build final w2c poses
    poses_w2c_out = np.zeros((T, 4, 4), dtype=np.float64)
    for i in range(T):
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :] = extrinsic_c2w[i]
        poses_w2c_out[i] = np.linalg.inv(c2w)

    # Extract [fx, fy, cx, cy] for each frame.
    intrinsics_out = np.zeros((T, 4), dtype=np.float64)
    for i in range(T):
        K = intrinsic_3x3[i]
        intrinsics_out[i] = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]

    # ----------------------------------------------------------- save NPZs
    np.savez_compressed(str(stage2_dir / "vggt_poses_w2c.npz"),  poses_w2c=poses_w2c_out)
    np.savez_compressed(str(stage2_dir / "vggt_intrinsics.npz"), intrinsics=intrinsics_out)
    np.savez_compressed(str(stage2_dir / "vggt_depth.npz"),      depth=depth_map)
    np.savez_compressed(str(stage2_dir / "vggt_conf.npz"),       conf=depth_conf)
    print(f"[run_vggt] Saved NPZ outputs to {stage2_dir}")

    # -------------------------------------------- build + save PLY cloud
    N_pts, H_vggt, W_vggt, _ = points_3d.shape

    # Per-frame colours at VGGT resolution.
    imgs_vggt = F.interpolate(
        images, size=(VGGT_FIXED_RESOLUTION, VGGT_FIXED_RESOLUTION),
        mode="bilinear", align_corners=False,
    )
    imgs_np = (imgs_vggt.cpu().numpy() * 255).astype(np.uint8)
    imgs_np = imgs_np.transpose(0, 2, 3, 1)  # (N, H, W, 3)

    conf_flat = depth_conf.reshape(-1)
    n_above   = int((conf_flat >= args.conf_thres_value).sum())

    if n_above < MIN_POINTS_BEFORE_FALLBACK:
        fallback_thres = float(np.percentile(conf_flat, CONF_THRES_FALLBACK_PERCENTILE))
        print(
            f"[run_vggt] WARNING: only {n_above} pixels pass confidence threshold "
            f"({args.conf_thres_value}).  Falling back to {100-CONF_THRES_FALLBACK_PERCENTILE}th "
            f"percentile = {fallback_thres:.4f}."
        )
        conf_mask = depth_conf >= fallback_thres
    else:
        conf_mask = depth_conf >= args.conf_thres_value  # (N, H, W) bool

    pts_flat  = points_3d.reshape(-1, 3)
    rgb_flat  = imgs_np.reshape(-1, 3)
    conf_flat_bool = conf_mask.reshape(-1)

    pts_sel  = pts_flat[conf_flat_bool]
    rgb_sel  = rgb_flat[conf_flat_bool]

    if len(pts_sel) > MAX_POINTS_PLY:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(pts_sel), MAX_POINTS_PLY, replace=False)
        pts_sel = pts_sel[idx]
        rgb_sel = rgb_sel[idx]

    ply_path = stage2_dir / "vggt_points3d.ply"
    save_points_ply(pts_sel, rgb_sel, ply_path)
    print(f"[run_vggt] Saved point cloud ({len(pts_sel)} pts): {ply_path}")

    # --------------------------------------- per-frame auxiliary outputs
    save_per_frame_outputs(stage2_dir, depth_map, depth_conf, points_3d)

    # --------------------------------------------------------------------- done
    print()
    print(f"[run_vggt] Done.  Outputs at: {stage2_dir}")
    print(f"  vggt_poses_w2c.npz  {poses_w2c_out.shape}")
    print(f"  vggt_depth.npz      {depth_map.shape}")
    print(f"  vggt_points3d.ply   {len(pts_sel)} points")
    print(f"  depth_maps/  confidence_maps/  normals/  (per-frame)")
    if quality_gate_result is not None:
        status = "PASSED" if quality_gate_result["passed"] else "FAILED"
        print(
            f"  quality_gate.json   rot={quality_gate_result['rot_error_deg_mean']:.2f}°  "
            f"trans={quality_gate_result['trans_error_pct_mean']:.2f}%  {status}"
        )


if __name__ == "__main__":
    main()
