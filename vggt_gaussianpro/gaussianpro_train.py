"""
Stage 3b — GaussianPro Optimization
======================================
Thin wrapper around GaussianPro's train.py that manages path conventions for
the FantasyWorld pipeline and the depth-prior injection flag.

Data contract
-------------
Reads:
    {colmap_dir}/images/          PNG frames
    {colmap_dir}/sparse/0/        cameras.txt  images.txt  points3D.txt
    {colmap_dir}/metricdepth/     {i}.npy float32 metric depth   (--use_depth_prior)
    {colmap_dir}/normals/         {i}.npy float32 [0,1]-normals  (--use_depth_prior)

Writes:
    {model_dir}/
        point_cloud/iteration_*/point_cloud.ply
        cameras.json
        cfg_args
        tb_logs/

Usage
-----
    # Minimal
    python gaussianpro_train.py \\
        --colmap_dir  runs/my_scene/stage3/colmap \\
        --model_dir   runs/my_scene/stage3/output_30000_gp_depth_prior

    # With depth prior
    python gaussianpro_train.py \\
        --colmap_dir  runs/my_scene/stage3/colmap \\
        --model_dir   runs/my_scene/stage3/output_30000_gp_depth_prior \\
        --iter        30000 \\
        --use_depth_prior \\
        --device      cuda:0
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = Path(__file__).resolve().parent
_GP_TRAIN_PY = _SCRIPT_DIR.parent / "thirdparty" / "GaussianPro" / "train.py"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 3b: GaussianPro optimization wrapper for the FantasyWorld pipeline."
    )
    parser.add_argument(
        "--colmap_dir", type=str, required=True,
        help="Path to the COLMAP directory output by colmap_adapter.py "
             "(should contain images/, sparse/0/, and optionally metricdepth/ "
             "and normals/ symlinks).",
    )
    parser.add_argument(
        "--model_dir", type=str, default=None,
        help="Output directory for the trained GaussianPro model.  "
             "Defaults to {colmap_dir}/../output_{iter}_gp[_depth_prior]/.",
    )
    parser.add_argument(
        "--iter", type=int, default=30_000,
        help="Total training iterations (default: 30000).",
    )
    parser.add_argument(
        "--lambda_lpips", type=float, default=0.3,
        help="Perceptual loss weight (informational; GaussianPro uses lambda_dssim "
             "natively — this value is accepted for CLI parity).",
    )
    # --- depth prior ---
    parser.add_argument(
        "--use_depth_prior", action="store_true", default=False,
        help="Inject VGGT depth maps and normals into GaussianPro propagation "
             "(activates --load_depth --load_normal --depth_loss --normal_loss).",
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.3,
        help="Minimum normalised depth confidence for normal/depth supervision "
             "(informational; not forwarded as a native GaussianPro flag).",
    )
    # --- propagation schedule ---
    parser.add_argument(
        "--propagation_interval", type=int, default=500,
        help="Iterations between successive propagation steps (default: 500).",
    )
    parser.add_argument(
        "--propagation_start", type=int, default=1000,
        help="Warm-up iterations before the first propagation step "
             "(maps to --propagated_iteration_begin, default: 1000).",
    )
    parser.add_argument(
        "--propagation_end", type=int, default=12_000,
        help="Iteration at which propagation stops "
             "(maps to --propagated_iteration_after, default: 12000).",
    )
    parser.add_argument(
        "--max_propagation_pts", type=int, default=50_000,
        help="Cap on new Gaussians per propagation step (default: 50000).",
    )
    parser.add_argument(
        "--patch_size", type=int, default=20,
        help="Patch size for ACMH-style patch-matching (default: 20).",
    )
    # --- eval split ---
    parser.add_argument(
        "--eval", action="store_true", default=False,
        help="Hold out every 8th frame as a test set (LLFF-style).",
    )
    # --- misc ---
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Torch device string (default: cuda:0).  Sets CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--port", type=int, default=6099,
        help="Network GUI port for GaussianPro viewer (default: 6099).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_colmap(colmap_dir: Path) -> None:
    """Sanity-check the COLMAP text layout before launching GaussianPro."""
    sparse_dir = colmap_dir / "sparse" / "0"
    for fname in ("cameras.txt", "images.txt", "points3D.txt"):
        fpath = sparse_dir / fname
        if not fpath.is_file():
            sys.exit(
                f"[gaussianpro_train] ERROR: {fpath} not found.\n"
                "Run colmap_adapter.py first to generate the COLMAP layout."
            )
    images_txt = sparse_dir / "images.txt"
    # A minimal images.txt has at least the header lines + one image entry.
    if images_txt.stat().st_size < 100:
        sys.exit(
            f"[gaussianpro_train] ERROR: {images_txt} appears empty or malformed "
            f"({images_txt.stat().st_size} bytes).\n"
            "Re-run colmap_adapter.py to regenerate."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    colmap_dir = Path(args.colmap_dir).resolve()

    if not colmap_dir.is_dir():
        sys.exit(
            f"[gaussianpro_train] ERROR: colmap_dir not found: {colmap_dir}\n"
            "Run colmap_adapter.py first."
        )

    suffix     = "_depth_prior" if args.use_depth_prior else ""
    model_dir  = (
        Path(args.model_dir).resolve() if args.model_dir
        else colmap_dir.parent / f"output_{args.iter}_gp{suffix}"
    )
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  FantasyWorld — GaussianPro Training (Stage 3b)")
    print(f"  colmap_dir  : {colmap_dir}")
    print(f"  model_dir   : {model_dir}")
    print(f"  iterations  : {args.iter}")
    print(f"  depth prior : {'enabled' if args.use_depth_prior else 'disabled'}")
    print("=" * 60)

    _validate_colmap(colmap_dir)

    # GaussianPro's train.py expects the COLMAP files in
    # {source_path}/sparse/0/.  The COLMAP adapter already writes them there.
    source_path = colmap_dir

    # ----------------------------------------- build subprocess command
    save_iters = sorted({1, 7_000, min(20_000, args.iter), args.iter})
    test_iters = sorted({1, 2_000, 7_000, min(20_000, args.iter), args.iter})

    cmd = [
        sys.executable,
        str(_GP_TRAIN_PY),
        "--source_path",   str(source_path),
        "--model_path",    str(model_dir),
        "--iterations",    str(args.iter),
        "--propagation_interval",        str(args.propagation_interval),
        "--propagated_iteration_begin",  str(args.propagation_start),
        "--propagated_iteration_after",  str(args.propagation_end),
        "--patch_size",    str(args.patch_size),
        "--save_iterations", *[str(it) for it in save_iters],
        "--test_iterations", *[str(it) for it in test_iters],
        "--port",          str(args.port),
    ]

    if args.use_depth_prior:
        cmd += [
            "--load_depth",
            "--load_normal",
            "--depth_loss",
            "--normal_loss",
        ]

    if args.eval:
        cmd.append("--eval")

    # Set CUDA_VISIBLE_DEVICES from --device (e.g. "cuda:0" → "0").
    env = os.environ.copy()
    device_idx = args.device.split(":")[-1] if ":" in args.device else "0"
    env.setdefault("CUDA_VISIBLE_DEVICES", device_idx)

    # GaussianPro's train.py must run with its own directory on sys.path.
    gp_root = _GP_TRAIN_PY.parent
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(gp_root) + os.pathsep + existing_pp if existing_pp else str(gp_root)
    )

    # GaussianPro's propagation step saves debug images to a hardcoded
    # relative path "cost/"; create it up front.
    (gp_root / "cost").mkdir(exist_ok=True)

    print("\n[gaussianpro_train] Launching GaussianPro train.py …")
    print("  CMD:", " ".join(cmd))
    print()

    result = subprocess.run(cmd, cwd=str(gp_root), env=env)

    if result.returncode != 0:
        sys.exit(
            f"[gaussianpro_train] GaussianPro train.py exited with code {result.returncode}."
        )

    print(f"\n[gaussianpro_train] Done.  Outputs at: {model_dir}")


if __name__ == "__main__":
    main()
