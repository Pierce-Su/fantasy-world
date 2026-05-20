# FantasyWorld — VGGT + GaussianPro 3D Lifting Pipeline

Lifts a FantasyWorld-generated 360-degree video clip into a full 3D Gaussian
Splatting scene using **[VGGT](https://github.com/facebookresearch/vggt)**
(CVPR 2025 Best Paper) for geometry estimation and
**[GaussianPro](https://github.com/kcheng1021/GaussianPro)** (ICML 2024)
for progressive Gaussian optimisation.

See [`../vggt_gaussianpro_prd.md`](../vggt_gaussianpro_prd.md) for the full
Product Requirement Document.

---

## Pipeline overview

```
FantasyWorld inference (inference_wan21.py / inference_wan22.py)
    │  video.mp4  +  debug.pth (depth, depth_conf, pose_enc)
    ▼
[Stage 1b]  export_stage1.py
    │  stage1/frames/  +  poses_w2c.npz  +  intrinsics.npz
    │                  +  depth.npz  +  depth_conf.npz
    ▼
[Stage 2]   run_vggt.py          (pose_free | pose_conditioned)
    │  stage2/vggt_poses_w2c.npz  vggt_depth.npz  vggt_points3d.ply
    │         depth_maps/  confidence_maps/  normals/
    ▼
[Stage 3a]  colmap_adapter.py
    │  stage3/colmap/images/  sparse/0/{cameras,images,points3D}.txt
    │                         metricdepth/ → stage2/depth_maps/
    │                         normals/     → stage2/normals/
    ▼
[Stage 3b]  gaussianpro_train.py
    │  stage3/output_{iter}_gp[_depth_prior]/point_cloud/…
    ▼
[Stage 4]   gaussianpro_render.py   (optional)
            stage3/output_*/train/ours_{iter}/renders/
```

---

## Quick start

### 1. Environment

```bash
conda env create -f vggt_gaussianpro/environment.yml
conda activate fw_vggt_gp
```

Install editable packages from submodules:

```bash
pip install -e thirdparty/vggt/
pip install -e thirdparty/utils3d/
pip install --no-build-isolation \
    thirdparty/GaussianPro/submodules/diff-gaussian-rasterization \
    thirdparty/GaussianPro/submodules/simple-knn \
    thirdparty/GaussianPro/submodules/Propagation
```

> **Note:** The GaussianPro `Propagation` CUDA extension requires `nvcc`.
> Edit `sm_XX` in its `CMakeLists.txt` to match your GPU (e.g. `sm_86` for
> RTX 3090/A100).

### 2. VGGT model weights

Downloaded automatically on first run.  For offline environments:

```bash
python - <<'PY'
import torch
url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
torch.save(state, "vggt_gaussianpro/checkpoints/vggt_1b.pt")
PY
```

Then pass `--vggt_checkpoint vggt_gaussianpro/checkpoints/vggt_1b.pt` to
`run_pipeline.py` or `VGGT_CHECKPOINT=...` to the shell scripts.

### 3. Run the full pipeline (single scene, including Stage 1)

```bash
python vggt_gaussianpro/run_pipeline.py \
    --prompt        "An enchanted forest at dusk" \
    --image_path    examples/images/forest.jpg \
    --camera_json   examples/cameras/camera_data_360_orbit.json \
    --model_version wan22 \
    --using_scale \
    --vggt_mode     pose_conditioned \
    --gs_iterations 30000 \
    --use_depth_prior \
    --workspace     vggt_gaussianpro/runs/forest_360 \
    --render
```

### 4. Run post-Stage-1 only (shell runner)

After running FantasyWorld inference manually and saving outputs to
`runs/forest_360/stage1/`:

```bash
cd vggt_gaussianpro
STAGE1_DIR=runs/forest_360/stage1 \
VGGT_MODE=pose_conditioned        \
GP_ITER=30000                     \
USE_DEPTH_PRIOR=1                 \
CAMERA_JSON=../examples/cameras/camera_data_360_orbit.json \
bash pipeline.sh
```

### 5. Batch mode (multiple scenes)

```bash
cd vggt_gaussianpro
# Named scenes
DATAROOT=./runs SCENE_NAMES="forest_360 castle_orbit" bash batch_pipeline.sh

# Auto-discover all scenes under ./runs that have a stage1/ folder
DATAROOT=./runs bash batch_pipeline.sh
```

---

## Running stages individually

### Stage 1b — frame and artifact export

```bash
python vggt_gaussianpro/export_stage1.py \
    --stage1_dir  vggt_gaussianpro/runs/forest_360/stage1 \
    --camera_json examples/cameras/camera_data_360_orbit.json \
    --image_size  480 832
```

### Stage 2 — VGGT geometry estimation

```bash
python vggt_gaussianpro/run_vggt.py \
    --stage1_dir  vggt_gaussianpro/runs/forest_360/stage1 \
    --mode        pose_conditioned \
    --device      cuda:0
```

Key outputs under `stage2/`:

| Path | Description |
|---|---|
| `vggt_poses_w2c.npz` | (T, 4, 4) world-to-camera matrices |
| `vggt_depth.npz` | (T, H, W) VGGT depth maps |
| `vggt_points3d.ply` | Coloured world-space point cloud |
| `depth_maps/{i}.npy` | Per-frame metric depth for GaussianPro |
| `confidence_maps/{i}.npy` | Per-frame depth confidence |
| `normals/{i}.npy` | Per-frame surface normals (3, H, W) in [0, 1] |
| `quality_gate.json` | Pose comparison results (when stage1 poses available) |

### Stage 3a — COLMAP adapter

```bash
python vggt_gaussianpro/colmap_adapter.py \
    --stage1_dir  vggt_gaussianpro/runs/forest_360/stage1 \
    --stage2_dir  vggt_gaussianpro/runs/forest_360/stage2 \
    --colmap_dir  vggt_gaussianpro/runs/forest_360/stage3/colmap
```

### Stage 3b — GaussianPro optimization

```bash
python vggt_gaussianpro/gaussianpro_train.py \
    --colmap_dir  vggt_gaussianpro/runs/forest_360/stage3/colmap \
    --model_dir   vggt_gaussianpro/runs/forest_360/stage3/output_30000_gp_depth_prior \
    --iter        30000 \
    --use_depth_prior
```

### Stage 4 — rendering

```bash
python vggt_gaussianpro/gaussianpro_render.py \
    --model_dir   vggt_gaussianpro/runs/forest_360/stage3/output_30000_gp_depth_prior \
    --colmap_dir  vggt_gaussianpro/runs/forest_360/stage3/colmap
```

---

## Directory layout

```
vggt_gaussianpro/
├── run_pipeline.py          full orchestration (end-to-end)
├── pipeline.sh              single-scene shell runner (post Stage 1)
├── batch_pipeline.sh        multi-scene batch runner
├── export_stage1.py         Stage 1b: frames + NPZ extraction from debug.pth
├── run_vggt.py              Stage 2: VGGT geometry estimation
├── colmap_adapter.py        Stage 3a: VGGT outputs → COLMAP text layout
├── gaussianpro_train.py     Stage 3b: GaussianPro optimization wrapper
├── gaussianpro_render.py    Stage 4: GaussianPro render wrapper
├── utils/
│   ├── __init__.py
│   └── depth_to_normal.py   batch point-map → surface normal conversion
├── checkpoints/             optional local VGGT weights (git-ignored)
├── runs/                    per-experiment workspaces (git-ignored)
├── environment.yml          conda env: fw_vggt_gp
└── README.md
```

Per-experiment workspace:

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
│   ├── depth_maps/
│   ├── confidence_maps/
│   ├── normals/
│   └── quality_gate.json
└── stage3/
    ├── colmap/
    │   ├── images/
    │   ├── sparse/0/
    │   │   ├── cameras.txt
    │   │   ├── images.txt
    │   │   └── points3D.txt
    │   ├── metricdepth/ → (symlink → stage2/depth_maps/)
    │   └── normals/     → (symlink → stage2/normals/)
    └── output_{iter}_gp[_depth_prior]/
        ├── point_cloud/iteration_*/point_cloud.ply
        ├── cameras.json
        ├── cfg_args
        └── tb_logs/
```

---

## VGGT operating modes

| Mode | When to use | Behaviour |
|---|---|---|
| `pose_conditioned` | Orbit/spin sequences with `debug.pth` | VGGT depth retained; extrinsics replaced with Stage 1 prior poses from `debug.pth` or camera JSON |
| `pose_free` | Generic input; cross-validation | VGGT estimates all poses from RGB; quality gate compares to Stage 1 priors if available |

The quality gate in `run_vggt.py` writes `stage2/quality_gate.json` with:
- `rot_error_deg_mean` — mean rotation error (degrees) vs Stage 1 priors
- `trans_error_pct_mean` — mean translation error (% of scene radius)
- `passed` — True when both thresholds are met (< 5° / < 5%)

---

## GPU memory requirements

| Frames | VGGT VRAM | Notes |
|---|---|---|
| 24 | ~6 GB | Fits on 3090/24 GB in fp16 |
| 35 | ~8 GB | Recommended default |
| 48 | ~12 GB | Default `--vggt_max_frames` |
| 81 | ~21 GB | Full FantasyWorld sequence; needs A100/H100 |

---

## Key differences from the DimensionX sibling pipeline

| Aspect | DimensionX | FantasyWorld (this pipeline) |
|---|---|---|
| Video generator | CogVideoX | FantasyWorld (Wan 2.1/2.2) |
| Native geometry from generator | None | depth + pose_enc in `debug.pth` |
| VGGT mode | pose_free | pose_conditioned (preferred) |
| Stage 1b | `get_frame.py` (frames only) | `export_stage1.py` (frames + NPZs) |
| COLMAP format | Binary (`.bin`) | Text (`.txt`) |
| COLMAP adapter | None (VGGT writes binary) | `colmap_adapter.py` |
| GaussianPro interface | `--dataset` name | `--colmap_dir` + `--model_dir` paths |

---

## References

- FantasyWorld: [Dai et al., ICLR 2026](https://openreview.net/forum?id=3q9vHEqsNx) · [Fantasy-AMAP/fantasy-world](https://github.com/Fantasy-AMAP/fantasy-world)
- VGGT: [Wang et al., CVPR 2025](https://vgg-t.github.io/) · [facebookresearch/vggt](https://github.com/facebookresearch/vggt)
- GaussianPro: [Cheng et al., ICML 2024](https://arxiv.org/abs/2402.14650) · [kcheng1021/GaussianPro](https://github.com/kcheng1021/GaussianPro)
- PRD: [`../vggt_gaussianpro_prd.md`](../vggt_gaussianpro_prd.md)
