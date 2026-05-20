"""
Stage 4 — GaussianPro Rendering
==================================
Thin wrapper around GaussianPro's render.py.  Renders train and/or test views
from a trained GaussianPro scene.

Usage
-----
    python gaussianpro_render.py \\
        --model_dir   runs/my_scene/stage3/output_30000_gp_depth_prior \\
        --colmap_dir  runs/my_scene/stage3/colmap

    # skip training renders, only test views
    python gaussianpro_render.py \\
        --model_dir   runs/my_scene/stage3/output_30000_gp/ \\
        --colmap_dir  runs/my_scene/stage3/colmap \\
        --skip_train

    # render a specific checkpoint iteration (defaults to latest)
    python gaussianpro_render.py \\
        --model_dir   runs/my_scene/stage3/output_30000_gp/ \\
        --colmap_dir  runs/my_scene/stage3/colmap \\
        --iteration   7000

Output
------
    {model_dir}/train/ours_{iter}/renders/   rendered RGB frames
    {model_dir}/train/ours_{iter}/gt/        ground-truth frames
    {model_dir}/test/ours_{iter}/renders/    (if --eval and test set is non-empty)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = Path(__file__).resolve().parent
_GP_RENDER_PY = _SCRIPT_DIR.parent / "thirdparty" / "GaussianPro" / "render.py"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 4: GaussianPro render wrapper for the FantasyWorld pipeline."
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Path to a trained GaussianPro model directory "
             "(e.g. runs/my_scene/stage3/output_30000_gp_depth_prior/).",
    )
    parser.add_argument(
        "--colmap_dir", type=str, required=True,
        help="Path to the COLMAP directory used during training "
             "(e.g. runs/my_scene/stage3/colmap/).  GaussianPro uses this "
             "as --source_path to load camera parameters.",
    )
    parser.add_argument(
        "--iteration", type=int, default=-1,
        help="Checkpoint iteration to load.  -1 (default) loads the latest "
             "saved checkpoint.",
    )
    parser.add_argument(
        "--skip_train", action="store_true", default=False,
        help="Skip rendering training views.",
    )
    parser.add_argument(
        "--skip_test", action="store_true", default=False,
        help="Skip rendering test views.",
    )
    parser.add_argument(
        "--eval", action="store_true", default=False,
        help="Hold out every 8th frame as a test set when loading the scene.  "
             "Only meaningful if the model was trained with --eval.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Torch device string (default: cuda:0).  Sets CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False,
        help="Suppress GaussianPro render.py output.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    model_dir  = Path(args.model_dir).resolve()
    colmap_dir = Path(args.colmap_dir).resolve()

    if not model_dir.is_dir():
        sys.exit(
            f"[gaussianpro_render] ERROR: model_dir not found: {model_dir}\n"
            "Run gaussianpro_train.py first."
        )
    if not colmap_dir.is_dir():
        sys.exit(
            f"[gaussianpro_render] ERROR: colmap_dir not found: {colmap_dir}"
        )

    print("=" * 60)
    print("  FantasyWorld — GaussianPro Rendering (Stage 4)")
    print(f"  model_dir   : {model_dir}")
    print(f"  colmap_dir  : {colmap_dir}")
    print(f"  iteration   : {'latest' if args.iteration == -1 else args.iteration}")
    print(f"  skip_train  : {args.skip_train}")
    print(f"  skip_test   : {args.skip_test}")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(_GP_RENDER_PY),
        "--model_path",  str(model_dir),
        "--source_path", str(colmap_dir),
        "--iteration",   str(args.iteration),
    ]

    if args.skip_train:
        cmd.append("--skip_train")
    if args.skip_test:
        cmd.append("--skip_test")
    if args.eval:
        cmd.append("--eval")
    if args.quiet:
        cmd.append("--quiet")

    # Set CUDA_VISIBLE_DEVICES from --device.
    env = os.environ.copy()
    device_idx = args.device.split(":")[-1] if ":" in args.device else "0"
    env.setdefault("CUDA_VISIBLE_DEVICES", device_idx)

    # Run render.py from its own directory so relative imports resolve.
    gp_root = _GP_RENDER_PY.parent
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(gp_root) + os.pathsep + existing_pp if existing_pp else str(gp_root)
    )

    print("\n[gaussianpro_render] Launching GaussianPro render.py …")
    print("  CMD:", " ".join(cmd))
    print()

    result = subprocess.run(cmd, cwd=str(gp_root), env=env)

    if result.returncode != 0:
        sys.exit(
            f"[gaussianpro_render] GaussianPro render.py exited with code {result.returncode}."
        )

    print(
        f"\n[gaussianpro_render] Done.  Renders at: "
        f"{model_dir}/train/  and/or  {model_dir}/test/"
    )


if __name__ == "__main__":
    main()
