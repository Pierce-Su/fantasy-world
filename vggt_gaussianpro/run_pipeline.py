"""
Full Pipeline Orchestration (end-to-end, including Stage 1)
=============================================================
Runs all stages of the FantasyWorld → VGGT + GaussianPro pipeline from a
single entry point.  Optionally calls FantasyWorld inference (Stage 1) before
running the lifting pipeline (Stages 1b → 4).

Usage
-----
    # Full run including FantasyWorld inference (Stage 1)
    python run_pipeline.py \\
        --prompt        "An enchanted forest at dusk" \\
        --image_path    ../examples/images/forest.jpg \\
        --camera_json   ../examples/cameras/camera_data_360_orbit.json \\
        --model_version wan22 \\
        --using_scale \\
        --vggt_mode     pose_conditioned \\
        --gs_iterations 30000 \\
        --use_depth_prior \\
        --workspace     runs/forest_360 \\
        --render

    # Skip Stage 1 (video.mp4 already exists in workspace/stage1/)
    python run_pipeline.py \\
        --skip_stage1 \\
        --camera_json   ../examples/cameras/camera_data_360_orbit.json \\
        --vggt_mode     pose_conditioned \\
        --gs_iterations 30000 \\
        --use_depth_prior \\
        --workspace     runs/forest_360

Workspace layout
----------------
    {workspace}/
        stage1/
            video.mp4
            debug.pth       (optional, from FantasyWorld inference)
            frames/
            poses_w2c.npz
            ...
        stage2/
            vggt_poses_w2c.npz
            ...
        stage3/
            colmap/
            output_{iter}_gp[_depth_prior]/
"""

import argparse
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_FW_ROOT    = _SCRIPT_DIR.parent   # fantasy-world repo root


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FantasyWorld VGGT+GaussianPro pipeline — full orchestration."
    )

    # ---- Stage 1 (FantasyWorld inference) ----
    stage1_grp = parser.add_argument_group("Stage 1 — FantasyWorld inference")
    stage1_grp.add_argument(
        "--skip_stage1", action="store_true", default=False,
        help="Skip FantasyWorld inference; assume {workspace}/stage1/video.mp4 "
             "(and optionally debug.pth) already exist.",
    )
    stage1_grp.add_argument(
        "--model_version", type=str, default="wan22",
        choices=["wan21", "wan22"],
        help="FantasyWorld model variant to use for Stage 1 (default: wan22).",
    )
    stage1_grp.add_argument(
        "--prompt", type=str, default="",
        help="Text prompt for FantasyWorld inference.",
    )
    stage1_grp.add_argument(
        "--image_path", type=str, default=None,
        help="Path to the reference image for FantasyWorld inference.",
    )
    stage1_grp.add_argument(
        "--end_image_path", type=str, default=None,
        help="Path to end image (Wan 2.2 only).",
    )
    stage1_grp.add_argument(
        "--wan_ckpt_path", type=str, default=None,
        help="Path to Wan model weights directory.",
    )
    stage1_grp.add_argument(
        "--model_ckpt", type=str, default=None,
        help="Path to FantasyWorld Wan 2.1 model checkpoint (wan21 only).",
    )
    stage1_grp.add_argument(
        "--model_ckpt_high", type=str, default=None,
        help="Path to FantasyWorld Wan 2.2 high-noise checkpoint (wan22 only).",
    )
    stage1_grp.add_argument(
        "--model_ckpt_low", type=str, default=None,
        help="Path to FantasyWorld Wan 2.2 low-noise checkpoint (wan22 only).",
    )
    stage1_grp.add_argument(
        "--camera_json", type=str,
        default=str(_FW_ROOT / "examples" / "cameras" / "camera_data_360_orbit.json"),
        help="Camera trajectory JSON for FantasyWorld inference and JSON-fallback "
             "pose export (default: examples/cameras/camera_data_360_orbit.json).",
    )
    stage1_grp.add_argument(
        "--sample_steps", type=int, default=50,
        help="Diffusion sampling steps for FantasyWorld inference (default: 50).",
    )
    stage1_grp.add_argument(
        "--using_scale", action="store_true", default=False,
        help="Pass --using_scale True to FantasyWorld inference.",
    )

    # ---- Stage 1b (export) ----
    export_grp = parser.add_argument_group("Stage 1b — artifact export")
    export_grp.add_argument(
        "--image_size", type=int, nargs=2, default=[480, 832],
        metavar=("H", "W"),
        help="Image resolution (H W) for principal-point derivation (default: 480 832).",
    )
    export_grp.add_argument(
        "--num_frames", type=int, default=None,
        help="Extract this many frames from video.mp4 (default: all).",
    )

    # ---- Stage 2 (VGGT) ----
    vggt_grp = parser.add_argument_group("Stage 2 — VGGT geometry")
    vggt_grp.add_argument(
        "--vggt_mode", type=str, default="pose_conditioned",
        choices=["pose_free", "pose_conditioned"],
        help="VGGT operating mode (default: pose_conditioned).",
    )
    vggt_grp.add_argument(
        "--vggt_max_frames", type=int, default=48,
        help="Max frames fed to VGGT (default: 48).",
    )
    vggt_grp.add_argument(
        "--vggt_checkpoint", type=str, default=None,
        help="Path to local VGGT-1B checkpoint (.pt).  Auto-downloaded if absent.",
    )
    vggt_grp.add_argument(
        "--conf_thres_value", type=float, default=5.0,
        help="VGGT depth confidence threshold for PLY export (default: 5.0).",
    )

    # ---- Stage 3a (COLMAP adapter) ----
    colmap_grp = parser.add_argument_group("Stage 3a — COLMAP adapter")
    colmap_grp.add_argument(
        "--prefer_stage1_poses", action="store_true", default=False,
        help="Use stage1 prior poses in COLMAP layout instead of VGGT-estimated.",
    )
    colmap_grp.add_argument(
        "--single_camera", action="store_true", default=False,
        help="Force a single shared PINHOLE camera.",
    )

    # ---- Stage 3b (GaussianPro) ----
    gp_grp = parser.add_argument_group("Stage 3b — GaussianPro training")
    gp_grp.add_argument(
        "--gs_iterations", type=int, default=30_000,
        help="GaussianPro training iterations (default: 30000).",
    )
    gp_grp.add_argument(
        "--use_depth_prior", action="store_true", default=False,
        help="Inject VGGT depth maps and normals into GaussianPro.",
    )
    gp_grp.add_argument(
        "--eval", action="store_true", default=False,
        help="Hold out every 8th frame as a test set.",
    )

    # ---- Stage 4 (render) ----
    render_grp = parser.add_argument_group("Stage 4 — rendering")
    render_grp.add_argument(
        "--render", action="store_true", default=False,
        help="Run gaussianpro_render.py after training.",
    )
    render_grp.add_argument(
        "--skip_render", action="store_true", default=False,
        help="Alias for omitting --render (render is skipped by default).",
    )

    # ---- Common ----
    parser.add_argument(
        "--workspace", type=str, required=True,
        help="Root workspace directory for this experiment "
             "(e.g. runs/forest_360).  All stages write under here.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Torch device string (default: cuda:0).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run(cmd: list[str], cwd: str | None = None, label: str = "") -> None:
    """Run a subprocess command; exit on failure."""
    print(f"\n[run_pipeline] {label}")
    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(
            f"[run_pipeline] Stage failed (exit code {result.returncode}): {label}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    workspace  = Path(args.workspace)
    stage1_dir = workspace / "stage1"
    stage2_dir = workspace / "stage2"
    stage3_dir = workspace / "stage3"
    colmap_dir = stage3_dir / "colmap"

    suffix     = "_depth_prior" if args.use_depth_prior else ""
    model_dir  = stage3_dir / f"output_{args.gs_iterations}_gp{suffix}"

    workspace.mkdir(parents=True, exist_ok=True)
    stage1_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  FantasyWorld — VGGT + GaussianPro Pipeline")
    print(f"  workspace       : {workspace}")
    print(f"  model_version   : {args.model_version}")
    print(f"  vggt_mode       : {args.vggt_mode}")
    print(f"  gs_iterations   : {args.gs_iterations}")
    print(f"  use_depth_prior : {args.use_depth_prior}")
    print(f"  render          : {args.render and not args.skip_render}")
    print("=" * 70)

    # ---------------------------------------------------------------- Stage 1
    if not args.skip_stage1:
        print("\n[Stage 1] FantasyWorld inference …")
        if args.model_version == "wan21":
            inference_script = str(_FW_ROOT / "inference_wan21.py")
            cmd = [
                sys.executable, inference_script,
                "--output_dir",      str(stage1_dir),
                "--camera_json_path", args.camera_json,
                "--prompt",          args.prompt,
                "--sample_steps",    str(args.sample_steps),
            ]
            if args.wan_ckpt_path:
                cmd += ["--wan_ckpt_path", args.wan_ckpt_path]
            if args.model_ckpt:
                cmd += ["--model_ckpt", args.model_ckpt]
            if args.image_path:
                cmd += ["--image_path", args.image_path]
            if args.using_scale:
                cmd += ["--using_scale", "True"]
        else:
            inference_script = str(_FW_ROOT / "inference_wan22.py")
            cmd = [
                sys.executable, inference_script,
                "--output_dir",      str(stage1_dir),
                "--camera_json_path", args.camera_json,
                "--prompt",          args.prompt,
                "--sample_steps",    str(args.sample_steps),
            ]
            if args.wan_ckpt_path:
                cmd += ["--wan_ckpt_path", args.wan_ckpt_path]
            if args.model_ckpt_high:
                cmd += ["--model_ckpt_high", args.model_ckpt_high]
            if args.model_ckpt_low:
                cmd += ["--model_ckpt_low", args.model_ckpt_low]
            if args.image_path:
                cmd += ["--image_path", args.image_path]
            if args.end_image_path:
                cmd += ["--end_image_path", args.end_image_path]
            if args.using_scale:
                cmd += ["--using_scale", "True"]

        run(cmd, cwd=str(_FW_ROOT), label="Stage 1 — FantasyWorld inference")

    # -------------------------------------------------------------- Stage 1b
    print("\n[Stage 1b] Exporting Stage 1 artifacts …")
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "export_stage1.py"),
        "--stage1_dir", str(stage1_dir),
        "--image_size",  str(args.image_size[0]), str(args.image_size[1]),
    ]
    if args.camera_json:
        cmd += ["--camera_json", args.camera_json]
    if args.num_frames:
        cmd += ["--num_frames", str(args.num_frames)]
    run(cmd, label="Stage 1b — export_stage1.py")

    # ---------------------------------------------------------------- Stage 2
    print("\n[Stage 2] VGGT geometry estimation …")
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "run_vggt.py"),
        "--stage1_dir", str(stage1_dir),
        "--stage2_dir", str(stage2_dir),
        "--mode",        args.vggt_mode,
        "--device",      args.device,
        "--max_frames",  str(args.vggt_max_frames),
        "--seed",        str(args.seed),
        "--conf_thres_value", str(args.conf_thres_value),
    ]
    if args.vggt_checkpoint:
        cmd += ["--checkpoint", args.vggt_checkpoint]
    run(cmd, label="Stage 2 — run_vggt.py")

    # --------------------------------------------------------------- Stage 3a
    print("\n[Stage 3a] COLMAP adapter …")
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "colmap_adapter.py"),
        "--stage1_dir", str(stage1_dir),
        "--stage2_dir", str(stage2_dir),
        "--colmap_dir", str(colmap_dir),
    ]
    if args.prefer_stage1_poses:
        cmd.append("--prefer_stage1_poses")
    if args.single_camera:
        cmd.append("--single_camera")
    run(cmd, label="Stage 3a — colmap_adapter.py")

    # --------------------------------------------------------------- Stage 3b
    print("\n[Stage 3b] GaussianPro optimization …")
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "gaussianpro_train.py"),
        "--colmap_dir", str(colmap_dir),
        "--model_dir",  str(model_dir),
        "--iter",        str(args.gs_iterations),
        "--device",      args.device,
    ]
    if args.use_depth_prior:
        cmd.append("--use_depth_prior")
    if args.eval:
        cmd.append("--eval")
    run(cmd, label="Stage 3b — gaussianpro_train.py")

    # ---------------------------------------------------------------- Stage 4
    if args.render and not args.skip_render:
        print("\n[Stage 4] Rendering …")
        cmd = [
            sys.executable,
            str(_SCRIPT_DIR / "gaussianpro_render.py"),
            "--model_dir",  str(model_dir),
            "--colmap_dir", str(colmap_dir),
            "--device",      args.device,
        ]
        if args.eval:
            cmd.append("--eval")
        run(cmd, label="Stage 4 — gaussianpro_render.py")

    # ---------------------------------------------------------------------- done
    print()
    print("=" * 70)
    print("  Pipeline complete.")
    print(f"  stage1     : {stage1_dir}")
    print(f"  stage2     : {stage2_dir}")
    print(f"  colmap     : {colmap_dir}")
    print(f"  gaussians  : {model_dir}")
    if args.render and not args.skip_render:
        print(f"  renders    : {model_dir}/train/")
    print("=" * 70)


if __name__ == "__main__":
    main()
