"""
Stage 1b — FantasyWorld Artifact Export
=========================================
Reads a FantasyWorld inference output directory containing ``video.mp4``
(and optionally ``debug.pth``) and writes structured artifacts consumed by
Stage 2 (run_vggt.py).

Outputs written to ``--stage1_dir``:

  frames/
      frame_{:05d}.png   — uint8 RGB frames from video.mp4
  poses_w2c.npz          — (T, 4, 4) float64 world-to-camera matrices
  intrinsics.npz         — (T, 4)    float64 [fx, fy, cx, cy]
  depth.npz              — (T, H, W) float32 metric depth        (from debug.pth)
  depth_conf.npz         — (T, H, W) float32 depth confidence    (from debug.pth)
  cameras.json           — copy of the camera trajectory JSON (when provided)

Pose source priority:
  1. debug.pth  ``pose_enc`` key  →  FantasyWorld's pose_encoding_to_extri_intri
  2. camera JSON ``cameras_interp`` list  →  fixed focal length + principal point

Usage
-----
    python export_stage1.py --stage1_dir runs/my_scene/stage1

    python export_stage1.py \\
        --stage1_dir  runs/my_scene/stage1 \\
        --camera_json ../examples/cameras/camera_data_360_orbit.json \\
        --image_size  480 832 \\
        --num_frames  48
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
# Add the fantasy-world repo root to sys.path so ``from FantasyWorld.vggt...``
# imports work (FantasyWorld is the embedded VGGT used for pose decoding).
_FW_ROOT = _SCRIPT_DIR.parent
if str(_FW_ROOT) not in sys.path:
    sys.path.insert(0, str(_FW_ROOT))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1b: Export FantasyWorld artifacts for the VGGT+GaussianPro pipeline."
    )
    parser.add_argument(
        "--stage1_dir", type=str, required=True,
        help="Path to the FantasyWorld output directory.  Frames, NPZs, and "
             "optional debug artifacts are written here.",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to video.mp4 (default: {stage1_dir}/video.mp4).",
    )
    parser.add_argument(
        "--debug_pth", type=str, default=None,
        help="Path to debug.pth (default: {stage1_dir}/debug.pth). "
             "When absent or does not contain 'pose_enc', falls back to camera JSON.",
    )
    parser.add_argument(
        "--camera_json", type=str, default=None,
        help="Path to the camera trajectory JSON used during FantasyWorld inference "
             "(e.g. examples/cameras/camera_data_360_orbit.json).  Used as pose "
             "source when debug.pth is unavailable, and always copied to stage1_dir.",
    )
    parser.add_argument(
        "--image_size", type=int, nargs=2, default=[480, 832],
        metavar=("H", "W"),
        help="Image resolution (H W) for principal-point derivation when poses "
             "come from the camera JSON (default: 480 832).",
    )
    parser.add_argument(
        "--num_frames", type=int, default=None,
        help="Uniformly subsample to this many frames from the video.  "
             "Default: extract every frame.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path, frames_dir: Path, num_frames: int | None) -> list[Path]:
    """
    Extract frames from video_path and save them as frame_{:05d}.png.
    Returns list of saved frame paths in order.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise RuntimeError(f"Video reports 0 frames: {video_path}")

    if num_frames is None or num_frames >= total:
        indices = list(range(total))
    else:
        indices = [round(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)]

    index_set = set(indices)
    saved_paths: list[Path] = []
    frame_count = 0
    extracted = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count in index_set:
            out_path = frames_dir / f"frame_{extracted:05d}.png"
            cv2.imwrite(str(out_path), frame)
            saved_paths.append(out_path)
            extracted += 1
        frame_count += 1

    cap.release()
    print(f"[export_stage1] Extracted {extracted}/{total} frames → {frames_dir}")
    return saved_paths


# ---------------------------------------------------------------------------
# Pose decoding from debug.pth
# ---------------------------------------------------------------------------

def _load_debug_pth(debug_pth_path: Path) -> dict | None:
    """Load debug.pth; return None if file does not exist."""
    if not debug_pth_path.is_file():
        return None
    print(f"[export_stage1] Loading debug.pth: {debug_pth_path}")
    data = torch.load(str(debug_pth_path), map_location="cpu", weights_only=False)
    print(f"[export_stage1] debug.pth keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    return data if isinstance(data, dict) else None


def _decode_poses_from_debug(data: dict, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Decode poses and intrinsics from debug.pth pose_enc using FantasyWorld's
    embedded VGGT pose decoder.

    Returns:
        poses_w2c:  (T, 4, 4) float64 world-to-camera
        intrinsics: (T, 4)    float64 [fx, fy, cx, cy]
    or None if pose_enc is not present.
    """
    if "pose_enc" not in data:
        print("[export_stage1] debug.pth has no 'pose_enc' key; falling back to camera JSON.")
        return None

    try:
        from FantasyWorld.vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except ImportError as exc:
        print(f"[export_stage1] WARNING: Cannot import FantasyWorld.vggt.utils.pose_enc: {exc}")
        print("[export_stage1] Falling back to camera JSON for poses.")
        return None

    pose_enc = data["pose_enc"]  # (T, D) or (1, T, D)
    if pose_enc.ndim == 3:
        pose_enc = pose_enc.squeeze(0)  # (T, D)
    pose_enc = pose_enc.float()

    H, W = image_size
    # pose_encoding_to_extri_intri returns (T, 3, 4) c2w and (T, 3, 3) K
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        pose_enc.unsqueeze(0), image_size=(H, W)
    )
    extrinsic = extrinsic.squeeze(0).detach().cpu().numpy()  # (T, 3, 4) c2w
    intrinsic = intrinsic.squeeze(0).detach().cpu().numpy()  # (T, 3, 3)

    T = extrinsic.shape[0]
    poses_w2c = np.zeros((T, 4, 4), dtype=np.float64)
    for i in range(T):
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :] = extrinsic[i].astype(np.float64)
        poses_w2c[i] = np.linalg.inv(c2w)

    # Extract [fx, fy, cx, cy] from 3x3 intrinsic matrix.
    intri_out = np.zeros((T, 4), dtype=np.float64)
    for i in range(T):
        K = intrinsic[i]
        intri_out[i] = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]

    print(f"[export_stage1] Decoded {T} poses from debug.pth pose_enc.")
    return poses_w2c, intri_out


# ---------------------------------------------------------------------------
# Pose derivation from camera JSON (fallback)
# ---------------------------------------------------------------------------

def _decode_poses_from_json(
    camera_json_path: Path,
    image_size: tuple[int, int],
    num_frames: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Derive world-to-camera poses from the ``cameras_interp`` list in a
    FantasyWorld camera trajectory JSON.

    The JSON format:
      {
        "focal_length": <float>,   # fx == fy (pixels at original resolution)
        "cameras_interp": [        # list of T 4×4 camera-to-world matrices
          [[...], [...], [...], [...]],
          ...
        ]
      }

    Returns:
        poses_w2c:  (T, 4, 4) float64 world-to-camera
        intrinsics: (T, 4)    float64 [fx, fy, cx, cy]
    """
    with open(camera_json_path, "r") as f:
        cam_data = json.load(f)

    mats_c2w: list = cam_data["cameras_interp"]
    H, W = image_size
    cx, cy = W / 2.0, H / 2.0

    # focal_length in the JSON is in pixels at the original resolution.
    # Scale to the current image resolution using the larger dimension.
    json_focal = float(cam_data.get("focal_length", 500))

    T_total = len(mats_c2w)
    if num_frames is not None and num_frames < T_total:
        indices = [round(i * (T_total - 1) / (num_frames - 1)) for i in range(num_frames)]
    else:
        indices = list(range(T_total))

    T = len(indices)
    poses_w2c = np.zeros((T, 4, 4), dtype=np.float64)
    intrinsics = np.zeros((T, 4), dtype=np.float64)

    for out_i, src_i in enumerate(indices):
        c2w = np.asarray(mats_c2w[src_i], dtype=np.float64).reshape(4, 4)
        poses_w2c[out_i] = np.linalg.inv(c2w)
        intrinsics[out_i] = [json_focal, json_focal, cx, cy]

    print(f"[export_stage1] Derived {T} poses from camera JSON: {camera_json_path}")
    return poses_w2c, intrinsics


# ---------------------------------------------------------------------------
# Depth export from debug.pth
# ---------------------------------------------------------------------------

def _export_depth(data: dict, stage1_dir: Path) -> None:
    """
    Save depth.npz and depth_conf.npz from debug.pth tensor data.
    """
    if "depth" not in data:
        print("[export_stage1] debug.pth has no 'depth' key; skipping depth export.")
        return

    depth = data["depth"]
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().float().numpy()
    depth = depth.astype(np.float32)
    # Shape may be (1, T, H, W) or (T, H, W) — squeeze batch dim.
    if depth.ndim == 4 and depth.shape[0] == 1:
        depth = depth[0]

    np.savez_compressed(str(stage1_dir / "depth.npz"), depth=depth)
    print(f"[export_stage1] Saved depth.npz  shape={depth.shape}")

    if "depth_conf" in data:
        conf = data["depth_conf"]
        if isinstance(conf, torch.Tensor):
            conf = conf.detach().cpu().float().numpy()
        conf = conf.astype(np.float32)
        if conf.ndim == 4 and conf.shape[0] == 1:
            conf = conf[0]
    else:
        print("[export_stage1] debug.pth has no 'depth_conf'; writing uniform confidence.")
        conf = np.ones_like(depth, dtype=np.float32)

    np.savez_compressed(str(stage1_dir / "depth_conf.npz"), depth_conf=conf)
    print(f"[export_stage1] Saved depth_conf.npz  shape={conf.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    stage1_dir = Path(args.stage1_dir)
    stage1_dir.mkdir(parents=True, exist_ok=True)

    video_path    = Path(args.video)    if args.video    else stage1_dir / "video.mp4"
    debug_pth_path = Path(args.debug_pth) if args.debug_pth else stage1_dir / "debug.pth"
    camera_json_path = Path(args.camera_json) if args.camera_json else None

    H, W = args.image_size

    print("=" * 60)
    print("  FantasyWorld — Stage 1b: Artifact Export")
    print(f"  stage1_dir  : {stage1_dir}")
    print(f"  video       : {video_path}")
    print(f"  debug_pth   : {debug_pth_path}")
    print(f"  camera_json : {camera_json_path or '(none)'}")
    print(f"  image_size  : {H}×{W}")
    print(f"  num_frames  : {args.num_frames or 'all'}")
    print("=" * 60)

    # ----------------------------------------------------------------- frames
    frames_dir = stage1_dir / "frames"
    if not video_path.is_file():
        print(f"[export_stage1] WARNING: video not found at {video_path}; skipping frame extraction.")
    else:
        extract_frames(video_path, frames_dir, args.num_frames)

    # ------------------------------------------------------------ debug.pth
    debug_data = _load_debug_pth(debug_pth_path)

    # ------------------------------------------------------------ poses
    poses_w2c = None
    intrinsics = None

    if debug_data is not None:
        result = _decode_poses_from_debug(debug_data, (H, W))
        if result is not None:
            poses_w2c, intrinsics = result

    if poses_w2c is None:
        if camera_json_path is None or not camera_json_path.is_file():
            print(
                "[export_stage1] WARNING: No debug.pth pose_enc and no camera JSON; "
                "cannot export poses.  Pass --camera_json to provide a fallback."
            )
        else:
            poses_w2c, intrinsics = _decode_poses_from_json(
                camera_json_path, (H, W), args.num_frames
            )

    if poses_w2c is not None:
        np.savez_compressed(str(stage1_dir / "poses_w2c.npz"), poses_w2c=poses_w2c)
        np.savez_compressed(str(stage1_dir / "intrinsics.npz"), intrinsics=intrinsics)
        print(f"[export_stage1] Saved poses_w2c.npz   shape={poses_w2c.shape}")
        print(f"[export_stage1] Saved intrinsics.npz  shape={intrinsics.shape}")

    # ----------------------------------------------------------------- depth
    if debug_data is not None:
        _export_depth(debug_data, stage1_dir)

    # --------------------------------------------------------- camera JSON copy
    if camera_json_path is not None and camera_json_path.is_file():
        dst = stage1_dir / "cameras.json"
        if not dst.exists():
            shutil.copy2(str(camera_json_path), str(dst))
            print(f"[export_stage1] Copied camera JSON → {dst}")

    # ---------------------------------------------------------------------- done
    print()
    print("[export_stage1] Done.")
    frame_count = len(list(frames_dir.glob("frame_*.png"))) if frames_dir.is_dir() else 0
    print(f"  frames/         {frame_count} files")
    if (stage1_dir / "poses_w2c.npz").exists():
        print(f"  poses_w2c.npz   ✓")
    if (stage1_dir / "intrinsics.npz").exists():
        print(f"  intrinsics.npz  ✓")
    if (stage1_dir / "depth.npz").exists():
        print(f"  depth.npz       ✓")
    if (stage1_dir / "depth_conf.npz").exists():
        print(f"  depth_conf.npz  ✓")


if __name__ == "__main__":
    main()
