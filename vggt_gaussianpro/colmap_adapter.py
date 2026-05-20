"""
Stage 3a — COLMAP Text Adapter
================================
Converts VGGT NPZ outputs into the COLMAP-text dataset layout expected by
GaussianPro's ``readColmapSceneInfo``.

This stage is specific to the FantasyWorld pipeline.  The sibling DimensionX
pipeline writes COLMAP binary directly from VGGT's ``demo_colmap.py``; here
we write text format (cameras.txt / images.txt / points3D.txt) from our NPZ
outputs.  GaussianPro's dataset reader handles both binary and text layouts.

Outputs written to ``--colmap_dir``:

  images/
      frame_{:05d}.png         copies of stage1/frames/
  sparse/0/
      cameras.txt              PINHOLE camera(s)
      images.txt               one two-line entry per frame
      points3D.txt             ≤ max_points3d points from vggt_points3d.ply
  metricdepth/                 symlink → stage2/depth_maps/
  normals/                     symlink → stage2/normals/

Usage
-----
    python colmap_adapter.py \\
        --stage1_dir  runs/my_scene/stage1 \\
        --stage2_dir  runs/my_scene/stage2

    python colmap_adapter.py \\
        --stage1_dir  runs/my_scene/stage1 \\
        --stage2_dir  runs/my_scene/stage2 \\
        --colmap_dir  runs/my_scene/stage3/colmap \\
        --prefer_stage1_poses \\
        --max_points3d 500000
"""

import argparse
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_POINTS3D_DEFAULT = 500_000
SINGLE_CAMERA_RELDIFF_THRESH = 0.01  # relative std-dev of fx for shared camera decision


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3a: Convert VGGT NPZ outputs to COLMAP text format."
    )
    parser.add_argument(
        "--stage1_dir", type=str, required=True,
        help="Path to Stage 1b export directory (contains frames/).",
    )
    parser.add_argument(
        "--stage2_dir", type=str, default=None,
        help="Path to Stage 2 VGGT output directory "
             "(default: {stage1_dir}/../stage2).",
    )
    parser.add_argument(
        "--colmap_dir", type=str, default=None,
        help="Output COLMAP directory "
             "(default: {stage2_dir}/../stage3/colmap).",
    )
    parser.add_argument(
        "--prefer_stage1_poses", action="store_true", default=False,
        help="Use stage1/poses_w2c.npz instead of stage2/vggt_poses_w2c.npz.  "
             "Useful in pose_conditioned mode or when the quality gate flagged issues.",
    )
    parser.add_argument(
        "--single_camera", action="store_true", default=None,
        help="Force a single shared PINHOLE camera for all frames.  "
             "Default: auto-detect based on relative std-dev of fx across frames "
             "(shared when rel-std < 1%%).",
    )
    parser.add_argument(
        "--max_points3d", type=int, default=MAX_POINTS3D_DEFAULT,
        help=f"Maximum points to write to points3D.txt (default: {MAX_POINTS3D_DEFAULT}).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# PLY loader (ASCII, dependency-free)
# ---------------------------------------------------------------------------

def load_ply_points(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load points and colours from an ASCII PLY file.

    Returns:
        points: (M, 3) float32
        colors: (M, 3) uint8
    """
    with open(ply_path, "r", encoding="ascii") as f:
        lines = f.readlines()

    # Parse header.
    n_verts = 0
    in_header = True
    data_start = 0
    has_color = False
    for idx, line in enumerate(lines):
        line = line.strip()
        if line == "end_header":
            data_start = idx + 1
            in_header = False
            break
        if line.startswith("element vertex"):
            n_verts = int(line.split()[-1])
        if "red" in line or "green" in line or "blue" in line:
            has_color = True

    if in_header:
        raise ValueError(f"Malformed PLY (no end_header): {ply_path}")

    points = np.zeros((n_verts, 3), dtype=np.float32)
    colors = np.zeros((n_verts, 3), dtype=np.uint8)

    for i, line in enumerate(lines[data_start: data_start + n_verts]):
        vals = line.strip().split()
        points[i] = [float(vals[0]), float(vals[1]), float(vals[2])]
        if has_color:
            colors[i] = [int(vals[3]), int(vals[4]), int(vals[5])]

    return points, colors


# ---------------------------------------------------------------------------
# cameras.txt writer
# ---------------------------------------------------------------------------

def write_cameras_txt(
    cameras_txt: Path,
    intrinsics: np.ndarray,
    single_camera: bool | None,
    image_width: int,
    image_height: int,
) -> dict[int, int]:
    """
    Write cameras.txt in COLMAP text format.

    Args:
        intrinsics:    (T, 4) float64 [fx, fy, cx, cy].
        single_camera: None = auto-detect; True = always shared; False = per-image.
        image_width:   pixel width of the training images.
        image_height:  pixel height of the training images.

    Returns:
        img_to_cam: dict mapping image_id (1-based) → camera_id (1-based).
    """
    T = intrinsics.shape[0]

    # Auto-detect shared vs per-image cameras.
    if single_camera is None:
        fx_vals = intrinsics[:, 0]
        rel_std = float(fx_vals.std() / (fx_vals.mean() + 1e-12))
        single_camera = rel_std < SINGLE_CAMERA_RELDIFF_THRESH
        print(
            f"[colmap_adapter] Camera mode: {'shared' if single_camera else 'per-image'} "
            f"(rel-std fx = {rel_std:.4f})"
        )

    cameras_txt.parent.mkdir(parents=True, exist_ok=True)
    img_to_cam: dict[int, int] = {}

    with open(cameras_txt, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {1 if single_camera else T}\n")

        if single_camera:
            fx, fy, cx, cy = intrinsics[0]
            f.write(f"1 PINHOLE {image_width} {image_height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
            for img_id in range(1, T + 1):
                img_to_cam[img_id] = 1
        else:
            for i in range(T):
                cam_id = i + 1
                img_id = i + 1
                fx, fy, cx, cy = intrinsics[i]
                f.write(
                    f"{cam_id} PINHOLE {image_width} {image_height} "
                    f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n"
                )
                img_to_cam[img_id] = cam_id

    return img_to_cam


# ---------------------------------------------------------------------------
# images.txt writer
# ---------------------------------------------------------------------------

def write_images_txt(
    images_txt: Path,
    poses_w2c: np.ndarray,
    img_to_cam: dict[int, int],
    frame_names: list[str],
) -> None:
    """
    Write images.txt in COLMAP text format.

    Each image has two lines:
      IMAGE_ID  QW  QX  QY  QZ  TX  TY  TZ  CAMERA_ID  NAME
      (empty POINTS2D line)

    Rotation quaternion is derived from poses_w2c[:3, :3] (world-to-camera).
    Translation is poses_w2c[:3, 3].
    """
    T = poses_w2c.shape[0]
    n_frames = len(frame_names)

    # Align poses to frame count via uniform resampling.
    if T != n_frames:
        pose_indices = [round(i * (T - 1) / max(n_frames - 1, 1)) for i in range(n_frames)]
    else:
        pose_indices = list(range(T))

    images_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(images_txt, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {n_frames}\n")

        for out_i, src_i in enumerate(pose_indices):
            img_id  = out_i + 1
            cam_id  = img_to_cam[img_id]
            w2c     = poses_w2c[src_i]          # (4, 4)

            R_w2c   = w2c[:3, :3]
            t_w2c   = w2c[:3, 3]

            # COLMAP uses (qw, qx, qy, qz) convention.
            rot = Rotation.from_matrix(R_w2c)
            qx, qy, qz, qw = rot.as_quat()     # scipy returns (x, y, z, w)

            name = frame_names[out_i]
            f.write(
                f"{img_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                f"{t_w2c[0]:.9f} {t_w2c[1]:.9f} {t_w2c[2]:.9f} {cam_id} {name}\n"
            )
            f.write("\n")  # empty POINTS2D line


# ---------------------------------------------------------------------------
# points3D.txt writer
# ---------------------------------------------------------------------------

def write_points3d_txt(
    points3d_txt: Path,
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    seed: int = 42,
) -> None:
    """
    Write points3D.txt in COLMAP text format.

    Track information uses a minimal placeholder ``1 0`` (image 1, point2D 0)
    since GaussianPro uses 3D coordinates for initialisation only.
    """
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    points3d_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(points3d_txt, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")

        for pt_id, (xyz, rgb) in enumerate(zip(points, colors), start=1):
            x, y, z = xyz
            r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
            f.write(f"{pt_id} {x:.7f} {y:.7f} {z:.7f} {r} {g} {b} 0.0 1 0\n")

    print(f"[colmap_adapter] Wrote {len(points)} points to {points3d_txt}")


# ---------------------------------------------------------------------------
# Symlink helpers
# ---------------------------------------------------------------------------

def _make_symlink(link_path: Path, target_path: Path) -> None:
    """Create or update a symlink at link_path pointing to target_path."""
    target_abs = target_path.resolve()
    link_path.parent.mkdir(parents=True, exist_ok=True)

    if link_path.is_symlink():
        if link_path.resolve() == target_abs:
            return
        link_path.unlink()
    elif link_path.exists():
        print(f"[colmap_adapter] WARNING: {link_path} is a real path, skipping symlink.")
        return

    link_path.symlink_to(target_abs)
    print(f"[colmap_adapter] Symlink: {link_path} → {target_abs}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    stage1_dir = Path(args.stage1_dir)
    stage2_dir = (
        Path(args.stage2_dir) if args.stage2_dir
        else stage1_dir.parent / "stage2"
    )
    colmap_dir = (
        Path(args.colmap_dir) if args.colmap_dir
        else stage2_dir.parent / "stage3" / "colmap"
    )

    print("=" * 60)
    print("  FantasyWorld — Stage 3a: COLMAP Text Adapter")
    print(f"  stage1_dir       : {stage1_dir}")
    print(f"  stage2_dir       : {stage2_dir}")
    print(f"  colmap_dir       : {colmap_dir}")
    print(f"  prefer_stage1_poses: {args.prefer_stage1_poses}")
    print(f"  max_points3d     : {args.max_points3d}")
    print("=" * 60)

    # ----------------------------------------------------------- load poses
    if args.prefer_stage1_poses:
        poses_path  = stage1_dir / "poses_w2c.npz"
        intri_path  = stage1_dir / "intrinsics.npz"
        source_label = "stage1"
    else:
        poses_path  = stage2_dir / "vggt_poses_w2c.npz"
        intri_path  = stage2_dir / "vggt_intrinsics.npz"
        source_label = "stage2 (VGGT)"

    if not poses_path.is_file():
        sys.exit(
            f"[colmap_adapter] ERROR: poses not found at {poses_path}.\n"
            "Run run_vggt.py (and export_stage1.py if using --prefer_stage1_poses) first."
        )
    if not intri_path.is_file():
        sys.exit(f"[colmap_adapter] ERROR: intrinsics not found at {intri_path}.")

    poses_w2c  = np.load(str(poses_path))["poses_w2c"]    # (T, 4, 4)
    intrinsics = np.load(str(intri_path))["intrinsics"]   # (T, 4)  [fx,fy,cx,cy]
    print(f"[colmap_adapter] Loaded poses from {source_label}: {poses_w2c.shape}")

    # ----------------------------------------- collect frame files
    frames_dir = stage1_dir / "frames"
    if not frames_dir.is_dir():
        sys.exit(
            f"[colmap_adapter] ERROR: frames directory not found: {frames_dir}.\n"
            "Run export_stage1.py first."
        )

    frame_paths = sorted(
        frames_dir.glob("frame_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not frame_paths:
        sys.exit(f"[colmap_adapter] ERROR: No frame_*.png found in {frames_dir}.")

    frame_names = [p.name for p in frame_paths]
    print(f"[colmap_adapter] Found {len(frame_names)} frames.")

    # ----------------------------------------- read image dimensions
    import cv2
    sample_img = cv2.imread(str(frame_paths[0]))
    if sample_img is None:
        sys.exit(f"[colmap_adapter] ERROR: Could not read sample image: {frame_paths[0]}")
    image_height, image_width = sample_img.shape[:2]
    print(f"[colmap_adapter] Image dimensions: {image_width}×{image_height}")

    # ----------------------------------------- copy frames to colmap images/
    images_dir = colmap_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for src in frame_paths:
        dst = images_dir / src.name
        if not dst.exists():
            shutil.copy2(str(src), str(dst))
    print(f"[colmap_adapter] Copied {len(frame_paths)} frames → {images_dir}")

    # ----------------------------------------- write cameras.txt
    sparse_dir = colmap_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    img_to_cam = write_cameras_txt(
        cameras_txt   = sparse_dir / "cameras.txt",
        intrinsics    = intrinsics,
        single_camera = args.single_camera,
        image_width   = image_width,
        image_height  = image_height,
    )
    print(f"[colmap_adapter] Wrote cameras.txt ({len(set(img_to_cam.values()))} camera(s))")

    # ----------------------------------------- write images.txt
    write_images_txt(
        images_txt = sparse_dir / "images.txt",
        poses_w2c  = poses_w2c,
        img_to_cam = img_to_cam,
        frame_names= frame_names,
    )
    print(f"[colmap_adapter] Wrote images.txt ({len(frame_names)} images)")

    # ----------------------------------------- write points3D.txt
    ply_path = stage2_dir / "vggt_points3d.ply"
    if ply_path.is_file():
        points, colors = load_ply_points(ply_path)
        write_points3d_txt(
            points3d_txt = sparse_dir / "points3D.txt",
            points       = points,
            colors       = colors,
            max_points   = args.max_points3d,
        )
    else:
        print(f"[colmap_adapter] WARNING: {ply_path} not found; writing empty points3D.txt.")
        (sparse_dir / "points3D.txt").write_text(
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            "# Number of points: 0\n"
        )

    # ----------------------------------------- depth + normal symlinks
    depth_maps_dir = stage2_dir / "depth_maps"
    normals_dir    = stage2_dir / "normals"

    if depth_maps_dir.is_dir():
        _make_symlink(colmap_dir / "metricdepth", depth_maps_dir)
    else:
        print(f"[colmap_adapter] WARNING: depth_maps/ not found at {depth_maps_dir}; "
              "skipping metricdepth symlink.")

    if normals_dir.is_dir():
        _make_symlink(colmap_dir / "normals", normals_dir)
    else:
        print(f"[colmap_adapter] WARNING: normals/ not found at {normals_dir}; "
              "skipping normals symlink.")

    # --------------------------------------------------------------------- done
    print()
    print(f"[colmap_adapter] Done.  COLMAP layout at: {colmap_dir}")
    print(f"  images/          {len(frame_names)} frames")
    print(f"  sparse/0/        cameras.txt  images.txt  points3D.txt")
    print(f"  metricdepth/     → {depth_maps_dir}")
    print(f"  normals/         → {normals_dir}")


if __name__ == "__main__":
    main()
