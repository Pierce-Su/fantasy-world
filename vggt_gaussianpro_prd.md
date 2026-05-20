# Product Requirement Document: FantasyWorld → VGGT + GaussianPro 3D Lifting Pipeline

**Status:** Active  
**Authors:** FantasyWorld / E3DQA Engineering  
**Date:** 2026-05-12  
**Related Work:** [DimensionX VGGT + GaussianPro PRD](../DimensionX/vggt_gaussianpro_prd.md) (sibling pipeline, different video generation stage)

---

## 1. Executive Summary

This document specifies the VGGT + GaussianPro 3D lifting pipeline for the **FantasyWorld** project. Starting from a FantasyWorld-generated 360-degree video clip (produced by the Wan 2.1/2.2-based unified video-and-3D model), the pipeline lifts the scene into a full 3D Gaussian Splatting representation by passing RGB frames through **VGGT** (Visual Geometry Grounded Transformer, CVPR 2025 Best Paper) for geometry estimation and **GaussianPro** (ICML 2024) for Gaussian optimization with progressive densification.

FantasyWorld is architecturally distinct from CogVideoX (the video generator used in the sibling DimensionX pipeline): it is a **unified feed-forward model** that jointly predicts video latents and explicit 3D geometry in a single pass, outputting not only `video.mp4` but also per-frame depth maps, confidence maps, and camera pose encodings stored in `debug.pth`. This native geometry output creates a unique integration opportunity — VGGT can operate in **pose-conditioned mode**, using Stage 1's known camera trajectory as a prior, which produces significantly better-conditioned geometry for a 360-degree orbit than a purely pose-free VGGT pass would.

The full pipeline lives under `vggt_gaussianpro/` and does not modify any other directory in the repository.

---

## 2. Background and Motivation

### 2.1 FantasyWorld: Unified Video and 3D Prediction

FantasyWorld (ICLR 2026) is built on the WanDiT video diffusion architecture and extends it with an asymmetric dual-branch structure:

- **Imagination Prior Branch** — synthesizes appearance/video latents.
- **Geometry-Consistent Branch** — runs explicit 3D reasoning (depth, point map, camera poses) using a VGGT-style backbone integrated directly into the denoising loop via stacked IRG (Integrated Reconstruction and Generation) blocks.

Two model variants are provided:

| Variant | Base model | Resolution | Notes |
|---|---|---|---|
| `FantasyWorld-Wan2.1-I2V-14B-480P` | Wan 2.1 I2V 14B | 480p | Paper-reproducible; image-to-video |
| `FantasyWorld-Wan2.2-Fun-A14B-Control-Camera` | Wan 2.2 Fun A14B | Higher | Camera-conditioned; enhanced quality |

Both variants accept a reference image, a text prompt, and a **camera trajectory JSON** (describing the 360-degree path), and produce:
- `video.mp4` — the rendered 360-degree sequence
- `debug.pth` — per-frame `depth`, `depth_conf`, and `pose_enc` tensors from the geometry branch

These geometry artifacts are a first-class input to the downstream VGGT + GaussianPro stages.

### 2.2 360-Degree Camera Trajectories

The FantasyWorld repository provides two trajectory templates under `examples/cameras/`:

| File | Type | Baseline | Recommended use |
|---|---|---|---|
| `camera_data_360_orbit.json` | Orbit (translational) | ~scene-radius | 3D reconstruction — provides parallax for triangulation |
| `camera_data_360_spin.json` | Spin (pure rotation) | Zero | Style transfer or novelty; requires Stage 1 depth injection for SfM |

For the VGGT + GaussianPro lifting pipeline, the **orbit** trajectory is strongly recommended because it provides the translational baseline needed for VGGT to produce well-conditioned depth and for GaussianPro's patch-matching densification.

A helper script `examples/cameras/_make_360.py` can generate custom trajectories.

### 2.3 Why Add VGGT After FantasyWorld's Built-In Geometry?

FantasyWorld's geometry branch already produces per-frame depth and poses at inference time, so a natural question is why VGGT is needed at all. There are three reasons:

1. **Scale and metric consistency**: FantasyWorld's depth predictions are produced by a diffusion-based architecture and can have per-frame scale drift. VGGT's depth-from-structure approach produces globally consistent metric depth, which GaussianPro's patch-matching propagation requires.

2. **Dense point cloud for GaussianPro initialization**: GaussianPro is initialized from a COLMAP-format `points3D` cloud. VGGT's unprojection of its metric depth maps (after multi-frame attention) produces a denser, more outlier-free point cloud than naively unprojecting FantasyWorld's per-frame depth.

3. **Cross-validation and quality gate**: Running VGGT in `pose_free` mode and comparing its estimated trajectory to FantasyWorld's prior provides an automated sanity check that the generated video has consistent parallax — a failure mode that would silently corrupt the GaussianPro reconstruction otherwise.

### 2.4 Comparison with the DimensionX Sibling Pipeline

| Aspect | DimensionX pipeline | FantasyWorld pipeline (this doc) |
|---|---|---|
| Video generator | CogVideoX | FantasyWorld (Wan 2.1/2.2) |
| Native geometry from generator | None | depth + pose_enc in `debug.pth` |
| VGGT mode | pose_free | pose_conditioned (preferred) or pose_free |
| Geometry artifact from Stage 1 | N/A | `depth.npz`, `depth_conf.npz`, `poses_w2c.npz` |
| Stage 1b (export) | `get_frame.py` (frame extraction only) | `export_stage1.py` (frames + depth + poses from debug.pth) |
| COLMAP format | Binary (`.bin`) | Text (`.txt`) — GaussianPro fork reads both |
| Extra COLMAP adapter stage | No (VGGT writes binary directly) | Yes — `colmap_adapter.py` converts VGGT NPZ outputs |
| Camera trajectory assumption | Unconstrained (any CogVideoX output) | 360-degree orbit or spin |
| VGGT quality gate | Not required | Enabled by default when prior poses available |

---

## 3. Goals and Non-Goals

### 3.1 Goals

- Implement a complete, standalone 5-stage pipeline under `vggt_gaussianpro/` that takes a FantasyWorld inference output (`video.mp4` + optional `debug.pth`) and produces a trained 3D Gaussian scene.
- Exploit FantasyWorld's native geometry output (`debug.pth`) to initialize VGGT in pose-conditioned mode, improving reconstruction quality on 360-degree orbit sequences.
- Export a COLMAP-text dataset layout from VGGT's outputs that GaussianPro can ingest without format conversion.
- Inject VGGT depth maps and surface normals into GaussianPro's progressive propagation step as priors.
- Provide `pipeline.sh`, `batch_pipeline.sh`, and a Python `run_pipeline.py` orchestrator with clear CLI conventions.
- Support both `wan21` and `wan22` FantasyWorld model variants as input sources.
- Provide a quality gate comparing VGGT-estimated poses to FantasyWorld's prior poses to detect degenerate reconstructions early.

### 3.2 Non-Goals

- Modifying any file outside `vggt_gaussianpro/` (inference scripts, FantasyWorld model code, thirdparty submodules).
- Training FantasyWorld, VGGT, or GaussianPro from scratch.
- Real-time or interactive 3D reconstruction.
- Supporting arbitrary unordered image sets as input (the pipeline assumes temporally ordered frames from a FantasyWorld 360-degree run).
- Replacing FantasyWorld's built-in geometry outputs as the primary geometry source for other downstream tasks.

---

## 4. Proposed Pipeline Architecture

### 4.1 High-Level Data Flow

```
FantasyWorld inference
  (inference_wan21.py / inference_wan22.py)
        │
        │   video.mp4
        │   debug.pth   ← depth, depth_conf, pose_enc
        │
        ▼
[Stage 1b]  export_stage1.py
        │   Extract frames → stage1/frames/frame_{:05d}.png
        │   Export poses   → stage1/poses_w2c.npz   (from debug.pth pose_enc
        │                                              or camera JSON)
        │   Export intrinsics → stage1/intrinsics.npz
        │   Export depth   → stage1/depth.npz
        │   Export conf    → stage1/depth_conf.npz
        │
        ▼
[Stage 2]   run_vggt.py
        │   VGGT forward pass (pose_free or pose_conditioned)
        │   → stage2/vggt_poses_w2c.npz     (T, 4, 4) float64
        │   → stage2/vggt_intrinsics.npz    (T, 4)    float64
        │   → stage2/vggt_depth.npz         (T, H, W) float32
        │   → stage2/vggt_conf.npz          (T, H, W) float32
        │   → stage2/vggt_points3d.ply      coloured world-space cloud
        │   → stage2/depth_maps/{i}.npy     GaussianPro-format per-frame depth
        │   → stage2/confidence_maps/{i}.npy
        │   → stage2/normals/{i}.npy        channels-first (3, H, W)
        │
        ▼
[Stage 3a]  colmap_adapter.py
        │   Convert VGGT NPZ outputs → COLMAP text layout
        │   → stage3/colmap/images/            frame copies
        │   → stage3/colmap/sparse/0/
        │       cameras.txt  (PINHOLE, single or per-image)
        │       images.txt   (IMAGE_ID qw qx qy qz tx ty tz CAM_ID NAME)
        │       points3D.txt (from vggt_points3d.ply)
        │   → stage3/colmap/metricdepth/       symlink → stage2/depth_maps/
        │   → stage3/colmap/normals/           symlink → stage2/normals/
        │
        ▼
[Stage 3b]  gaussianpro_train.py
        │   GaussianPro optimization (30k iterations)
        │   Progressive propagation guided by VGGT depth + normals
        │   → stage3/output_{iter}_gp[_depth_prior]/
        │       point_cloud/iteration_*/point_cloud.ply
        │       cameras.json
        │       cfg_args
        │       tb_logs/
        │
        ▼
[Stage 4]   gaussianpro_render.py           (optional)
            Render train/test views from trained Gaussians
            → stage3/output_{iter}_gp_depth_prior/train/
```

### 4.2 Directory Layout

```
vggt_gaussianpro/
├── run_pipeline.py          # full orchestration entry point (end-to-end)
├── pipeline.sh              # single-scene shell runner (post Stage 1)
├── batch_pipeline.sh        # multi-scene batch runner
├── export_stage1.py         # Stage 1b: frames + NPZ extraction from debug.pth
├── run_vggt.py              # Stage 2: VGGT geometry estimation
├── colmap_adapter.py        # Stage 3a: VGGT outputs → COLMAP text layout
├── gaussianpro_train.py     # Stage 3b: GaussianPro optimization wrapper
├── gaussianpro_render.py    # Stage 4: GaussianPro render wrapper
├── utils/
│   ├── __init__.py
│   └── depth_to_normal.py   # batch point-map → surface normal conversion
├── checkpoints/             # optional local VGGT weights (git-ignored)
├── runs/                    # per-experiment workspaces (git-ignored)
├── environment.yml          # conda env: fw_vggt_gp
└── README.md
```

Each experiment's workspace follows this layout:

```
runs/{scene_name}/
├── stage1/
│   ├── video.mp4
│   ├── debug.pth            (optional — recommended for quality)
│   ├── frames/
│   │   └── frame_{:05d}.png
│   ├── poses_w2c.npz
│   ├── intrinsics.npz
│   ├── depth.npz
│   ├── depth_conf.npz
│   └── cameras.json
├── stage2/
│   ├── vggt_poses_w2c.npz
│   ├── vggt_intrinsics.npz
│   ├── vggt_depth.npz
│   ├── vggt_conf.npz
│   ├── vggt_points3d.ply
│   ├── depth_maps/          {i}.npy  (float32)
│   ├── confidence_maps/     {i}.npy  (float32)
│   └── normals/             {i}.npy  (float32, channels-first)
└── stage3/
    ├── colmap/
    │   ├── images/
    │   ├── sparse/0/
    │   │   ├── cameras.txt
    │   │   ├── images.txt
    │   │   └── points3D.txt
    │   ├── metricdepth/ → (symlink)
    │   └── normals/     → (symlink)
    └── output_{iter}_gp[_depth_prior]/
        ├── point_cloud/iteration_*/point_cloud.ply
        ├── cameras.json
        ├── cfg_args
        └── tb_logs/
```

---

## 5. Component Specifications

### 5.1 Stage 1 — FantasyWorld Inference (`inference_wan21.py` / `inference_wan22.py`)

This stage is **upstream** of this pipeline and is run separately. It is documented here for context.

#### 5.1.1 Wan 2.1 Inference

```bash
python inference_wan21.py \
    --wan_ckpt_path   ./models/Wan-AI/Wan2.1-I2V-14B-480P \
    --model_ckpt      ./models/FantasyWorld-Wan2.1-I2V-14B-480P/model.pth \
    --image_path      ./examples/images/input_image.png \
    --camera_json_path ./examples/cameras/camera_data_360_orbit.json \
    --prompt          "<scene description>" \
    --output_dir      ./runs/my_scene/stage1 \
    --sample_steps    50 \
    --using_scale     True
```

#### 5.1.2 Wan 2.2 Inference

```bash
python inference_wan22.py \
    --image_path          ./examples/images/input_image.png \
    --end_image_path      ./examples/images/end_image.png \
    --wan_ckpt_path       ./models/ \
    --camera_json_path    ./examples/cameras/camera_data_360_orbit.json \
    --prompt              "<scene description>" \
    --model_ckpt_high     ./models/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera/high_noise_model.pth \
    --model_ckpt_low      ./models/FantasyWorld-Wan2.2-Fun-A14B-Control-Camera/low_noise_model.pth \
    --output_dir          ./runs/my_scene/stage1 \
    --sample_steps        50 \
    --using_scale         True
```

**Outputs** (written to `--output_dir`):

| File | Description |
|---|---|
| `video.mp4` | The rendered 360-degree sequence |
| `debug.pth` | PyTorch dict containing `depth` (T,H,W), `depth_conf` (T,H,W), `pose_enc` (T,D) |

**Camera trajectory**: Always use `camera_data_360_orbit.json` for 3D reconstruction. The spin trajectory (`camera_data_360_spin.json`) produces a zero-baseline sequence that cannot be triangulated; if it must be used, pass `--vggt_mode pose_conditioned` to force Stage 2 to use FantasyWorld's poses, and ensure `debug.pth` is present.

---

### 5.2 Stage 1b — Artifact Export (`export_stage1.py`)

Reads `video.mp4` and `debug.pth` from a FantasyWorld output directory and writes structured artifacts consumed by Stage 2.

#### 5.2.1 Frame Extraction

Frames are extracted from `video.mp4` using OpenCV and saved as `frame_{:05d}.png`. Uniform temporal subsampling is applied when `--num_frames` is specified.

#### 5.2.2 Pose Export

**From `debug.pth` (preferred):** When `debug.pth` contains a `pose_enc` key, poses are decoded via FantasyWorld's embedded VGGT `pose_encoding_to_extri_intri` using the `absT_quaR_FoV` encoding. This overrides JSON-derived poses for highest accuracy.

**From camera JSON (fallback):** When `debug.pth` is absent or does not contain `pose_enc`, poses are derived from the `cameras_interp` list in the camera JSON, using a fixed focal length and the image principal point.

**Output format:** `poses_w2c.npz` — (T, 4, 4) float64 world-to-camera matrices; `intrinsics.npz` — (T, 4) float64 `[fx, fy, cx, cy]`.

#### 5.2.3 Depth Export

When `debug.pth` is present and contains `depth` and optionally `depth_conf` keys:
- `depth.npz` — (T, H, W) float32 metric depth
- `depth_conf.npz` — (T, H, W) float32 confidence; uniform 1.0 if absent in the checkpoint

#### 5.2.4 CLI

```
python export_stage1.py \
    --stage1_dir  runs/my_scene/stage1 \
    [--video      runs/my_scene/stage1/video.mp4]      # default: {stage1_dir}/video.mp4
    [--debug_pth  runs/my_scene/stage1/debug.pth]      # default: {stage1_dir}/debug.pth
    [--camera_json examples/cameras/camera_data_360_orbit.json]
    [--image_size 480 832]
    [--num_frames 48]
```

---

### 5.3 Stage 2 — VGGT Geometry Estimation (`run_vggt.py`)

#### 5.3.1 Operating Modes

**`pose_free` (default for CogVideoX-style input; also a quality-check option)**  
VGGT estimates camera poses purely from RGB frames. When `poses_w2c.npz` is available in `stage1_dir`, a rotation-and-translation quality gate is automatically run to compare VGGT's trajectory with FantasyWorld's prior. Thresholds: rotation error < 5° mean, translation error < 5% of scene radius.

**`pose_conditioned` (recommended for production FantasyWorld runs)**  
VGGT's estimated extrinsics are replaced with Stage 1's known world-to-camera matrices (inverted to camera-to-world for VGGT's coordinate frame). VGGT's depth and point maps are retained unchanged — they are already produced under VGGT's coordinate frame. This mode is preferred when the camera trajectory is known (orbit or spin) because it eliminates the risk of VGGT pose drift on textureless or repetitive fantasy-world content.

#### 5.3.2 Model Loading

```python
from vggt.models.vggt import VGGT

model = VGGT()
# Option A: automatic download from HuggingFace
state = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
    map_location="cpu",
)
model.load_state_dict(state)
# Option B: local checkpoint
# model.load_state_dict(torch.load("checkpoints/vggt_1b.pt"))
model.eval().to(device)
```

The bundled copy at `thirdparty/vggt/` is used (not the `FantasyWorld/vggt/` embedded copy — the latter is for FantasyWorld's training pipeline).

#### 5.3.3 Inference

```python
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

images, original_coords = load_and_preprocess_images_square(
    sorted_frame_paths, resolution=1024
)
images = images.to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        images_b = images[None]                     # (1, N, 3, H, W)
        aggregated_tokens_list, ps_idx = model.aggregator(images_b)
        pose_enc  = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, images_b.shape[-2:]           # (N, 3, 4), (N, 3, 3)
        )
        depth_map, depth_conf = model.depth_head(
            aggregated_tokens_list, images_b, ps_idx   # (N, H, W) each
        )

# World-space point map for PLY export + normal computation
points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
# (N, H, W, 3)
```

Frame count is capped by `--max_frames` (default 48) via uniform temporal subsampling. When `N > max_frames`, indices are chosen as `round(i * (N-1) / (max_frames-1))` for `i ∈ [0, max_frames-1]` to preserve the first and last frames.

#### 5.3.4 Pose-Conditioned Override

In `pose_conditioned` mode, after the VGGT forward pass, the estimated extrinsics are replaced:

```python
for i in range(T):
    w2c = prior_poses_w2c[i]                     # from stage1/poses_w2c.npz
    c2w = np.linalg.inv(w2c)
    extrinsic_out[i] = c2w[:3, :]               # (3, 4) camera-to-world

    fx, fy, cx, cy = prior_intrinsics[i]         # from stage1/intrinsics.npz
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    intrinsic_out[i] = K
```

Depth maps and the derived point cloud remain from VGGT (they are geometry-estimated, not diffusion-estimated, and have better metric consistency than Stage 1's depth).

#### 5.3.5 Auxiliary Outputs

In addition to the five primary NPZ/PLY outputs, `run_vggt.py` writes per-frame files for GaussianPro:

| Path | Shape / Format | Purpose |
|---|---|---|
| `stage2/depth_maps/{i}.npy` | (H, W) float32 | Per-frame metric depth for GaussianPro prior |
| `stage2/confidence_maps/{i}.npy` | (H, W) float32 | Per-frame depth confidence (gates propagation) |
| `stage2/normals/{i}.npy` | (3, H, W) float32, range [0, 1] | Surface normals (channels-first, remapped from [-1,1]) |

Normals are computed from the unprojected point map via `utils/depth_to_normal.py` (`batch_point_map_to_normals`), which uses finite-difference cross-products on the world-space point grid.

#### 5.3.6 Point Cloud Export

High-confidence points are selected by absolute threshold `conf_thres_value = 5.0`. If fewer than 1,000 points pass, the threshold falls back to the 80th-percentile of observed confidence scores. The resulting point set is further subsampled to at most 200,000 points for the PLY file (used only for COLMAP initialisation; the full depth maps are retained for GaussianPro).

#### 5.3.7 CLI

```
python run_vggt.py \
    --stage1_dir  runs/my_scene/stage1 \
    --stage2_dir  runs/my_scene/stage2          # default: {stage1_dir}/../stage2
    --mode        pose_conditioned               # pose_free | pose_conditioned
    --device      cuda:0
    --max_frames  48
    [--checkpoint checkpoints/vggt_1b.pt]
    [--conf_thres_value 5.0]
    [--seed 42]
```

---

### 5.4 Stage 3a — COLMAP Adapter (`colmap_adapter.py`)

Converts VGGT's NPZ outputs into the COLMAP-text dataset layout expected by GaussianPro's `readColmapSceneInfo`. This stage is specific to the FantasyWorld pipeline; the DimensionX pipeline writes COLMAP binary directly from VGGT's `demo_colmap.py`.

#### 5.4.1 cameras.txt

PINHOLE model. By default, if the relative standard deviation of `fx` across frames is < 1%, a single shared camera is written; otherwise per-image cameras are written. This is controlled by `--single_camera`.

#### 5.4.2 images.txt

One two-line entry per frame:
```
IMAGE_ID  QW  QX  QY  QZ  TX  TY  TZ  CAMERA_ID  NAME
(empty POINTS2D line)
```
Rotation matrix from `poses_w2c[i, :3, :3]` is converted to `(qw, qx, qy, qz)` via `scipy.spatial.transform.Rotation`. Translation is `poses_w2c[i, :3, 3]`.

Frame count alignment: if the number of poses differs from the number of extracted frames (e.g., due to differing subsampling in export vs. VGGT), uniform resampling aligns them.

#### 5.4.3 points3D.txt

Loaded from `stage2/vggt_points3d.ply` (ASCII PLY). Subsampled to at most `--max_points3d` (default 500,000) via random selection. Track information is a minimal placeholder `1 0` (image 1, point2D index 0), since GaussianPro uses the 3D coordinates only for initialization.

#### 5.4.4 Depth and Normal Symlinks

To make GaussianPro's depth-prior path work without data duplication, two symlinks are created:

```
stage3/colmap/metricdepth → stage2/depth_maps/    (GaussianPro reads {i}.npy)
stage3/colmap/normals      → stage2/normals/       (GaussianPro reads {i}.npy)
```

#### 5.4.5 Pose Source Selection

The adapter defaults to VGGT-estimated poses (`vggt_poses_w2c.npz`). Pass `--prefer_stage1_poses` to use Stage 1's poses instead — useful in `pose_conditioned` mode where the final VGGT output already encodes Stage 1 poses, or when the VGGT quality gate flagged a potential issue.

#### 5.4.6 CLI

```
python colmap_adapter.py \
    --stage1_dir  runs/my_scene/stage1 \
    --stage2_dir  runs/my_scene/stage2 \
    --colmap_dir  runs/my_scene/stage3/colmap  # default: {stage2_dir}/../stage3/colmap
    [--prefer_stage1_poses]
    [--single_camera]
    [--max_points3d 500000]
```

---

### 5.5 Stage 3b — GaussianPro Optimization (`gaussianpro_train.py`)

A thin wrapper around `thirdparty/GaussianPro/train.py` that manages path conventions and the depth-prior injection flag.

#### 5.5.1 Overview of GaussianPro

GaussianPro extends vanilla 3DGS by adding a **progressive propagation** module that fires between standard densification steps:

1. Renders depth and normal maps from the current Gaussian set.
2. Runs patch matching (ACMH-style) on rendered and reference frames to identify neighboring surface patches.
3. Propagates existing Gaussians into poorly-covered regions, using matched-patch geometry to assign accurate position and orientation to new Gaussians.

VGGT's per-frame depth maps serve as **metric depth priors** for the propagation module:
- `metricdepth/{i}.npy` provides reference depth for patch matching.
- `normals/{i}.npy` orients newly-propagated Gaussians.
- Confidence maps (accessible via `confidence_maps/{i}.npy`) can optionally gate which pixels participate, excluding uncertain regions from normal supervision.

#### 5.5.2 Key Training Parameters

| Parameter | Default | Notes |
|---|---|---|
| `--iter` | 30000 | Matches DimensionX pipeline default |
| `--lambda_lpips` | 0.3 | Perceptual loss weight |
| `--use_depth_prior` | True (flag) | Enable VGGT depth + normal injection |
| `--propagation_interval` | 500 | Iterations between propagation steps |
| `--propagation_start` | 1000 | Warm-up before first propagation |
| `--max_propagation_pts` | 50000 | Cap on new Gaussians per propagation step |
| `--confidence_threshold` | 0.3 | Exclude low-confidence pixels from normal supervision |
| `--eval` | False | Hold out every 8th frame as test split |

#### 5.5.3 Output Structure

```
stage3/output_{iter}_gp[_depth_prior]/
├── point_cloud/
│   ├── iteration_7000/point_cloud.ply
│   ├── iteration_30000/point_cloud.ply
│   └── ...
├── cameras.json
├── cfg_args
└── tb_logs/
```

The `_depth_prior` suffix is appended when `--use_depth_prior` is active.

#### 5.5.4 CLI

```
python gaussianpro_train.py \
    --colmap_dir  runs/my_scene/stage3/colmap \
    --model_dir   runs/my_scene/stage3/output_30000_gp_depth_prior \
    --iter        30000 \
    --device      cuda:0 \
    [--use_depth_prior] \
    [--eval]
```

---

### 5.6 Stage 4 — Rendering (`gaussianpro_render.py`)

A thin wrapper around `thirdparty/GaussianPro/render.py`. Renders all training (and optionally test) views from the trained Gaussian scene.

```
python gaussianpro_render.py \
    --model_dir  runs/my_scene/stage3/output_30000_gp_depth_prior \
    --colmap_dir runs/my_scene/stage3/colmap \
    --device     cuda:0 \
    [--eval]
```

Rendered frames are written to `{model_dir}/train/` (and `test/` when `--eval`).

---

### 5.7 Full Orchestration (`run_pipeline.py` and shell scripts)

#### `run_pipeline.py` (Python, end-to-end including Stage 1)

The primary entry point for running the full pipeline from a text prompt:

```bash
python run_pipeline.py \
    --prompt        "An enchanted forest at dusk" \
    --image_path    ../examples/images/forest.jpg \
    --camera_json   ../examples/cameras/camera_data_360_orbit.json \
    --model_version wan22 \
    --using_scale \
    --vggt_mode     pose_conditioned \
    --gs_iterations 30000 \
    --use_depth_prior \
    --workspace     runs/forest_360 \
    --render
```

To skip Stage 1 when it has already been run:

```bash
python run_pipeline.py \
    --skip_stage1 \
    --camera_json   ../examples/cameras/camera_data_360_orbit.json \
    --vggt_mode     pose_conditioned \
    --gs_iterations 30000 \
    --workspace     runs/forest_360
```

#### `pipeline.sh` (single scene, post-Stage-1 shell runner)

Accepts configuration via environment variables and orchestrates Stages 1b–4:

```bash
STAGE1_DIR=runs/forest_360/stage1 \
VGGT_MODE=pose_conditioned \
GP_ITER=30000 \
USE_DEPTH_PRIOR=1 \
CAMERA_JSON=../examples/cameras/camera_data_360_orbit.json \
bash pipeline.sh
```

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `STAGE1_DIR` | (required) | Path to Stage 1 output directory |
| `WORKSPACE` | `$(dirname STAGE1_DIR)` | Parent workspace; stages write under here |
| `CAMERA_JSON` | `../examples/cameras/camera_data_360_orbit.json` | Trajectory used for pose export from JSON |
| `IMAGE_SIZE_H` / `_W` | `480` / `832` | Image resolution for principal-point derivation |
| `VGGT_MODE` | `pose_conditioned` | `pose_free` or `pose_conditioned` |
| `VGGT_MAX_FRAMES` | `48` | Max frames fed to VGGT |
| `VGGT_CHECKPOINT` | `` | Path to local `.pt` weights; empty = auto-download |
| `GP_ITER` | `30000` | GaussianPro training iterations |
| `USE_DEPTH_PRIOR` | `1` | Set to `` to disable depth injection |
| `SKIP_RENDER` | `` | Set to `1` to skip Stage 4 |

#### `batch_pipeline.sh` (multi-scene)

Auto-discovers all `{DATAROOT}/{scene}/stage1/` directories and runs the pipeline sequentially:

```bash
# Process named scenes
DATAROOT=./runs SCENE_NAMES="forest_360 castle_orbit" bash batch_pipeline.sh

# Auto-discover all scenes under ./runs that have a stage1/ folder
DATAROOT=./runs bash batch_pipeline.sh
```

On completion, a summary line `Batch complete. success=N failed=M` is printed.

---

## 6. Environment and Dependencies

The pipeline uses a dedicated conda environment `fw_vggt_gp` defined in `vggt_gaussianpro/environment.yml`, separate from the main `fantasyworld` environment.

```yaml
name: fw_vggt_gp
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch>=2.1
  - torchvision
  - pytorch-cuda=11.8
  - pip
  - pip:
    - huggingface_hub
    - transformers
    - opencv-python
    - pillow
    - numpy
    - scipy
    - trimesh
    - plyfile
    - tqdm
    - tensorboard
```

### Submodule and Extension Installation

```bash
# VGGT (standalone version)
pip install -e ../thirdparty/vggt/

# utils3d (dependency of several geometry utilities)
pip install -e ../thirdparty/utils3d/

# GaussianPro CUDA extensions
pip install --no-build-isolation \
    ../thirdparty/GaussianPro/submodules/diff-gaussian-rasterization \
    ../thirdparty/GaussianPro/submodules/simple-knn \
    ../thirdparty/GaussianPro/submodules/Propagation
```

The GaussianPro `Propagation` submodule requires a CUDA-capable build environment. Ensure `nvcc` is available and that `sm_XX` in the submodule's `CMakeLists.txt` matches the target GPU (e.g., `sm_86` for RTX 3090/A10, `sm_80` for A100).

### VGGT Model Weights

Downloaded automatically on first use. For air-gapped environments:

```bash
python - <<'PY'
import torch
url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
torch.save(state, "checkpoints/vggt_1b.pt")
PY
```

Then pass `--checkpoint checkpoints/vggt_1b.pt` to `run_vggt.py`.

### Note on FantasyWorld's Embedded VGGT

The FantasyWorld repository contains an embedded VGGT copy at `FantasyWorld/vggt/`. This is used exclusively by FantasyWorld's training pipeline (for encoding `pose_enc` in `debug.pth`). `export_stage1.py` imports `pose_encoding_to_extri_intri` from this embedded copy to decode Stage 1 poses. **`run_vggt.py` and all other Stage 2+ scripts use `thirdparty/vggt/`** (the standalone, up-to-date submodule).

---

## 7. Interface Contracts and Data Formats

### 7.1 Stage 1b Outputs

```
stage1/
├── frames/
│   └── frame_{:05d}.png   # uint8 RGB, FantasyWorld output resolution (e.g. 480×832)
├── poses_w2c.npz           # (T, 4, 4) float64 world-to-camera
├── intrinsics.npz          # (T, 4)    float64 [fx, fy, cx, cy]
├── depth.npz               # (T, H, W) float32 metric depth (from debug.pth)
├── depth_conf.npz          # (T, H, W) float32 confidence   (from debug.pth)
└── cameras.json            # copy of the camera trajectory JSON
```

### 7.2 Stage 2 Outputs (VGGT)

```
stage2/
├── vggt_poses_w2c.npz      # (T, 4, 4) float64 — final poses (VGGT or Stage-1 priors)
├── vggt_intrinsics.npz     # (T, 4)    float64 — [fx, fy, cx, cy]
├── vggt_depth.npz          # (T, H, W) float32 — VGGT depth maps (VGGT resolution)
├── vggt_conf.npz           # (T, H, W) float32 — VGGT depth confidence
├── vggt_points3d.ply       # ASCII PLY, coloured, ≤200k points
├── depth_maps/{i}.npy      # (H, W) float32 per frame — for GaussianPro
├── confidence_maps/{i}.npy # (H, W) float32 per frame
└── normals/{i}.npy         # (3, H, W) float32 per frame, values in [0,1]
```

**Resolution note:** VGGT internally processes images at 518×518 px. `depth_map` and `depth_conf` outputs are at this resolution. `depth_maps/{i}.npy` saves them at VGGT resolution; `images.txt` in Stage 3a references the original frame resolution. GaussianPro handles the resolution mismatch via bilinear resampling of the depth prior to match the training image size.

### 7.3 Stage 3a Outputs (COLMAP adapter)

```
stage3/colmap/
├── images/
│   └── frame_{:05d}.png     # uint8 RGB copies of stage1/frames/
├── sparse/0/
│   ├── cameras.txt          # PINHOLE cameras (1 shared or T per-image)
│   ├── images.txt           # T images with qw qx qy qz tx ty tz
│   └── points3D.txt         # ≤500k points from vggt_points3d.ply
├── metricdepth/             # symlink → stage2/depth_maps/
└── normals/                 # symlink → stage2/normals/
```

**Coordinate convention (OpenCV):** +X right, +Y down, +Z forward. `images.txt` rotation quaternion and translation vector are derived from `poses_w2c[:3, :]`. Both GaussianPro and the bundled `gaussian-splatting` operate in this convention; no axis flip is required.

### 7.4 Stage 3b Outputs (GaussianPro)

```
stage3/output_{iter}_gp[_depth_prior]/
├── point_cloud/
│   ├── iteration_7000/point_cloud.ply
│   └── iteration_30000/point_cloud.ply
├── cameras.json
├── cfg_args
└── tb_logs/
```

---

## 8. Performance Targets

| Stage | Expected wall-clock (48 frames, A100) | Notes |
|---|---|---|
| Stage 1 — FantasyWorld inference | ~5–10 min | 81 frames at 50 diffusion steps; most of pipeline runtime |
| Stage 1b — export_stage1.py | < 30 s | Frame decode + NPZ writes |
| Stage 2 — VGGT forward (48 frames) | < 15 s | Single bfloat16 forward pass; ~10 GB VRAM |
| Stage 3a — COLMAP adapter | < 10 s | NPZ → text conversion + symlinks |
| Stage 3b — GaussianPro (30k iter) | ~20–25 min | Includes propagation overhead; ~12 GB VRAM |
| Stage 4 — render | ~2 min | Train-view renders only |

**VRAM requirements (Stage 2 VGGT):**

| Frames | VRAM | Notes |
|---|---|---|
| 24 | ~6 GB | Fits 3090/24 GB in fp16 |
| 35 | ~8 GB | Recommended default |
| 48 | ~12 GB | Default `--max_frames` |
| 81 | ~21 GB | Full FantasyWorld sequence; needs A100/H100 |

---

## 9. Testing and Validation Plan

### 9.1 Unit Tests

- `test_export_stage1.py`: Given a sample `video.mp4` and `debug.pth`, assert that `frames/`, `poses_w2c.npz`, `intrinsics.npz`, `depth.npz`, and `depth_conf.npz` are produced with correct shapes and finite values.
- `test_run_vggt.py`: Given a pre-exported `stage1/` directory, assert that all five primary Stage 2 outputs are produced and non-empty, and that `vggt_points3d.ply` contains at least 1,000 points.
- `test_colmap_adapter.py`: Given a Stage 2 output, assert that `cameras.txt`, `images.txt`, and `points3D.txt` exist and can be parsed by the test reader, and that the symlinks resolve.
- `test_gaussianpro_train.py`: Given a minimal COLMAP directory, run 500 iterations and assert that `output_500_gp_depth_prior/point_cloud/iteration_500/point_cloud.ply` exists.

### 9.2 Integration Test (Single Scene)

Run `pipeline.sh` on an existing FantasyWorld output (`assets/fantasyworld/video_1.mp4` re-inferred to produce `debug.pth`) with:
- `VGGT_MODE=pose_conditioned`
- `GP_ITER=7000`
- `USE_DEPTH_PRIOR=1`

Compare PSNR/SSIM of `gaussianpro_render.py` output against:
1. A vanilla 3DGS baseline (no depth prior, no propagation) on the same input.
2. The DimensionX pipeline output on equivalent CogVideoX content, as a cross-model reference.

### 9.3 Regression Gate

- PSNR of the new pipeline must not be worse than vanilla 3DGS + DUSt3R by more than 0.2 dB on the integration test scene.
- VGGT + export wall-clock time must be ≤ 60 s for 48 frames on a single A100/H100.
- The VGGT quality gate must not emit a WARNING on a standard orbit sequence with a well-behaved reference image.

### 9.4 Trajectory-Specific Tests

- **Orbit trajectory (`camera_data_360_orbit.json`)**: Full reconstruction pipeline in `pose_conditioned` mode. Expected: VGGT quality gate passes, GaussianPro converges without floaters.
- **Spin trajectory (`camera_data_360_spin.json`)**: Run with `pose_conditioned` + `prefer_stage1_poses`. Expected: reconstruction quality is lower (zero baseline) but does not crash; `debug.pth` must be present for this path to work.

---

## 10. Open Questions and Risks

| # | Question / Risk | Owner | Resolution Path |
|---|---|---|---|
| 1 | `debug.pth` may not be written by default in some FantasyWorld inference invocations (it is a debug artifact). Silently falling back to JSON-derived poses degrades pose quality. | Engineering | Add a `--save_debug_pth` flag to `inference_wan21.py` / `inference_wan22.py` or document it prominently in the README. |
| 2 | FantasyWorld's embedded `FantasyWorld/vggt/` may diverge from `thirdparty/vggt/` over time (API changes). `export_stage1.py` imports `pose_encoding_to_extri_intri` from the former. | Engineering | Add a version-check assertion at import time; pin both submodules to compatible commits. |
| 3 | GaussianPro's CUDA `Propagation` extension requires manual `sm_XX` editing in `CMakeLists.txt`. Incorrect compute capability produces silent wrong results on some GPUs. | DevOps | Add an automated GPU-capability probe in `environment.yml` setup script; document per-GPU sm values in README. |
| 4 | VGGT attention memory scales as O(N²). At 81 frames (full FantasyWorld sequence), ~21 GB VRAM is needed. Using `--max_frames 48` reduces this but discards frames. | Engineering | Evaluate chunked-inference mode (sliding window) for sequences > 48 frames; expose `--chunk_size` flag. |
| 5 | Spin trajectory (zero baseline) produces a degenerate SfM point cloud because there is no parallax for triangulation. GaussianPro's propagation may diverge or produce a flat reconstruction. | Research | Document that spin trajectories require `--vggt_mode pose_conditioned` + a present `debug.pth`; evaluate whether MoGe (already in `thirdparty/MoGe`) could provide single-frame depth as a fallback initialization. |
| 6 | VGGT's depth output resolution is 518×518 px (fixed internal resolution), while FantasyWorld frames are 480×832. The depth-to-image aspect-ratio mismatch requires careful bilinear resampling in GaussianPro's depth prior path. | Engineering | Validate that GaussianPro's depth loader applies aspect-preserving resampling; add an assertion checking that depth and image shapes are compatible before training starts. |
| 7 | VGGT non-commercial license restricts production deployment. The commercial checkpoint (`VGGT-1B-Commercial`) requires an approval form. | Legal/Product | Proceed with non-commercial for research; file commercial application if productising. |
| 8 | The quality gate in `run_vggt.py` warns but does not abort when pose error is high. Downstream GaussianPro may silently produce a bad model. | Engineering | Add a `--abort_on_quality_gate_fail` flag; log quality-gate metrics to a `stage2/quality_gate.json` for batch-mode reporting. |

---

## 11. Migration and Coexistence Notes

- All pipeline code is contained under `vggt_gaussianpro/`. No files elsewhere in the repository are modified.
- The `fw_vggt_gp` conda environment is separate from `fantasyworld`. They share `thirdparty/vggt/` and `thirdparty/GaussianPro/` as submodules but install them into separate Python environments.
- Data directories are fully scoped to per-experiment `runs/{scene}/` workspaces. There is no global `data/` directory shared across experiments.
- The sibling DimensionX pipeline (`DimensionX/vggt_gaussianpro/`) is structurally similar but targets CogVideoX MP4 input. Its `data/` layout differs (flat `data/images/{dataset}/` and `data/scenes/{dataset}/`) and it writes COLMAP binary rather than text. Scripts are not shared between the two pipelines.
- `thirdparty/MoGe/` (single-image monocular geometry estimation) is present in the repository and could serve as a fallback geometry source for spin trajectories in a future iteration.

---

## 12. References

- FantasyWorld paper: [Dai et al., ICLR 2026](https://openreview.net/forum?id=3q9vHEqsNx)
- FantasyWorld repository: [Fantasy-AMAP/fantasy-world](https://github.com/Fantasy-AMAP/fantasy-world)
- VGGT paper: [Wang et al., CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.pdf)
- VGGT repository: [facebookresearch/vggt](https://github.com/facebookresearch/vggt)
- GaussianPro paper: [Cheng et al., ICML 2024](https://arxiv.org/abs/2402.14650)
- GaussianPro repository: [kcheng1021/GaussianPro](https://github.com/kcheng1021/GaussianPro)
- DimensionX sibling PRD: [`../DimensionX/vggt_gaussianpro_prd.md`](../DimensionX/vggt_gaussianpro_prd.md)
- Pipeline README: [`vggt_gaussianpro/README.md`](vggt_gaussianpro/README.md)
- Camera trajectory helper: [`examples/cameras/_make_360.py`](examples/cameras/_make_360.py)
- Wan video generation: [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1), [alibaba-pai/Wan2.2-Fun](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control-Camera)
- MoGe (potential fallback): [`thirdparty/MoGe/`](thirdparty/MoGe/)
