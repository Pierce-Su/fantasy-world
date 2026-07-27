# Feasibility: Longer Video Sequences (Wan 2.1 & Wan 2.2)

**Date:** 2026-07-27  
**Scope:** Can FantasyWorld expose a usable parameter for generating longer-than-default videos on both Wan 2.1 and Wan 2.2?  
**Verdict:** **Yes — already partially supported.** Both inference entrypoints already accept `--frames` (default `81`). Making longer sequences reliable is mostly an integration / validation problem (camera JSON length, divisibility constraint, VRAM), not a model-architecture blocker.

---

## 1. Executive summary

| Question | Answer |
|---|---|
| Is longer generation architecturally possible? | **Yes** for both Wan 2.1 and Wan 2.2 |
| Is a CLI parameter already present? | **Yes** — `--frames` on both `inference_wan21.py` and `inference_wan22.py` |
| Can you just pass `--frames 161` today? | **Only if** the camera JSON has ≥ that many `cameras_interp` poses and `frames % 4 == 1` |
| Hard model upper bound? | Soft RoPE table of **1024 latent frames** (~4093 pixel frames) — far above practical VRAM limits |
| Main risks | Camera/frame mismatch, silent `% 4` rounding bugs, VRAM growth, quality drop beyond training length (~81) |

---

## 2. What already exists

### 2.1 Wan 2.1 — `inference_wan21.py`

```text
--frames  (default: 81)
```

Wired through `FantasyWorldSampler(..., frames=args.frames)` into:

- `RealEstate10KPoseProcessor(sample_n_frames=frames)`
- `pipe.encode_image(..., self.num_frames, ...)`
- `model.generate_video(..., num_frames=self.num_frames)`

The fusion model itself also enforces Wan’s temporal packing rule:

```python
# FantasyWorld/fusion/model_wan21.py
if num_frames % 4 != 1:
    num_frames = (num_frames + 2) // 4 * 4 + 1
```

Help text for Wan 2.1 does **not** currently document the `% 4 == 1` rule or camera-length coupling.

### 2.2 Wan 2.2 — `inference_wan22.py`

```text
--frames  (default: 81)
help: must satisfy frames % 4 == 1; e.g. 81, 97, 113, 129, 161
      Must match the number of entries in cameras_interp
```

Same flow: sampler stores `self.num_frames`, pose processor uses it, dual-model denoising allocates noise as:

```text
(1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8)
```

Wan 2.2’s help already anticipates longer lengths (up to **161** in the examples).

### 2.3 Camera trajectory generator — `examples/cameras/_make_360.py`

Already has `--num_frames` with the same `% 4 == 1` auto-correction and documents the same example ladder (`81, 97, 113, 129, 161`). This is the intended companion for longer runs.

### 2.4 Stock camera JSONs (all fixed at 81)

| File | `cameras_interp` length |
|---|---|
| `camera_data.json` | 81 |
| `camera_data_360.json` | 81 |
| `camera_data_360_orbit.json` | 81 |
| `camera_data_360_spin.json` | 81 |

An existing curated output (`output-wan21/.../0000/video.mp4`) is **81 frames @ 16 fps** (~5.06 s), matching the default.

### 2.5 Batch path gap

`batch_inference_wan21.sh` does **not** pass `--frames` and always uses the stock 81-frame camera JSON. Longer sequences are not wired into batch yet.

---

## 3. Why longer frames are technically allowed

### 3.1 Temporal latent packing (both models)

Wan compresses time by 4× in the VAE / DiT latent:

```text
F_latent = (num_frames - 1) // 4 + 1
```

Valid pixel lengths therefore satisfy `num_frames ≡ 1 (mod 4)`:

| Pixel frames | Latent frames `F` | Duration @ 16 fps |
|---|---|---|
| 81 (default) | 21 | ~5.1 s |
| 97 | 25 | ~6.1 s |
| 113 | 29 | ~7.1 s |
| 129 | 33 | ~8.1 s |
| 161 | 41 | ~10.1 s |
| 241 | 61 | ~15.1 s |

### 3.2 RoPE / positional tables

Both DiffSynth Wan stacks precompute 3D RoPE with `end=1024`:

```python
# diffsynth_wan21|22/.../wan_video_dit.py
def precompute_freqs_cis_3d(dim, end=1024, ...):
```

Temporal freqs are sliced as `freqs[0][:f]` where `f = F_latent`. So the table supports up to **1024 latent frames** (~4093 pixel frames). That is not a practical limit; VRAM is.

The older `FantasyWorld/wan/modules/model.py` path similarly builds `rope_params(1024, ...)`.

### 3.3 No fixed-length embedding that hard-codes 81

Frame count flows as a runtime tensor shape through:

1. Noise / latents  
2. Plücker camera control features  
3. DiT patchify + RoPE  
4. Geometry (VGGT-style) branch inside the fusion IRG blocks  

There is no “always 81” weight tensor that would forbid longer sequences at inference.

---

## 4. Hard requirements to make longer sequences work

### 4.1 Camera JSON length must cover `--frames`

Pose processing asserts:

```python
assert len(cam_params) >= self.sample_n_frames
```

(`diffsynth_wan21|22/data/dataset_re10k.py` → `get_plucker_embedding_direct_from_cam_params`)

Then it samples `sample_n_frames` poses via `np.linspace`. Consequences:

- **Too few cameras** → assertion failure (cannot run).
- **Exactly matching** → one pose per generated frame (correct).
- **More cameras than frames** → temporally subsampled (works, but unusual).

For a 360° orbit/spin at length `N`, regenerate:

```bash
python examples/cameras/_make_360.py orbit --num_frames 161
# or: spin / both
```

Then pass the new JSON + matching `--frames 161`.

### 4.2 Enforce `frames % 4 == 1` at the CLI

Both fusion paths round invalid values upward. Plücker embeddings are built from the **unrounded** `self.num_frames`. If a user passes e.g. `80`:

- Plücker temporal size stays 80  
- Noise / latents become 81 after rounding  
- Shape mismatch / crash or silent misalignment  

Wan 2.2’s help already warns; Wan 2.1 should get the same validation (prefer hard fail over silent round).

### 4.3 VRAM scales with temporal length

Approximate latent token count ∝ `F × (H/patch) × (W/patch)`. Attention is quadratic in that product.

On this instance (NVIDIA RTX PRO 6000, ~96 GB), headroom above 81 frames is likely large at default resolutions:

- Wan 2.1 defaults: **336×592**  
- Wan 2.2 defaults: **480×832**

Still expect roughly linear–superlinear growth in activation memory and wall-clock as frames increase. Empirically validate before batching long clips.

### 4.4 Training-distribution / quality caveat

Public configs and docs treat **81 frames** as the operating point (paper-reproducible Wan 2.1; README examples; PRD timing tables). Longer lengths are **supported by the architecture** but are out-of-distribution relative to that default. Possible degradation:

- Temporal consistency / drifting texture  
- Weaker camera adherence late in the clip  
- Geometry branch (`depth` / `pose_enc` in `debug.pth`) noisier on long horizons  

This is a quality risk, not a hard “impossible.”

---

## 5. Downstream pipeline impact (VGGT / GaussianPro)

Longer FantasyWorld videos are compatible with the Stage 1b+ lifting stack, with caveats:

| Stage | Effect of longer video |
|---|---|
| `export_stage1.py` | Fine — can export all frames or subsample via `--num_frames` |
| `run_vggt.py` | Caps via `--max_frames` (default **48**); longer source videos are subsampled |
| VRAM at full length | PRD notes ~21 GB VGGT at 81 frames; longer needs chunking or higher VRAM |
| Geometry quality | More frames can help coverage **if** VGGT sees enough of them; default 48 may discard the extra motion |

So: longer generation is useful even if VGGT still consumes a 48-frame subset; unlocking the full longer sequence in 3D lifting is a separate follow-up.

---

## 6. Gap analysis: “add a parameter” vs current state

| Need | Current state | Gap |
|---|---|---|
| Expose frame count on Wan 2.1 | `--frames` exists | Document + validate `% 4 == 1` |
| Expose frame count on Wan 2.2 | `--frames` exists + better help | Optional stricter preflight |
| Match cameras to length | `_make_360.py --num_frames` exists; stock JSONs are 81-only | Auto-generate or assert match in inference |
| Batch / env wiring | Not present | Add `FRAMES` to `batch_inference_wan21.sh` (and any Wan 2.2 batch) |
| Duration UX | Only raw frame count | Optional `--duration_sec` → `round_to_valid(fps * sec)` |
| README | Does not list `--frames` | Document for both models |

**Bottom line:** You do not need a new diffusion mechanism. You need a **coherent longer-sequence knob** that keeps `--frames`, camera JSON, and (optionally) batch scripts in sync.

---

## 7. Recommended design (if implementing)

Minimal, shared behavior for both models:

1. Keep `--frames` as the canonical parameter (already present).  
2. Add preflight checks in both `inference_wan21.py` and `inference_wan22.py`:
   - `frames % 4 == 1` (error, don’t silent-round at the CLI)  
   - `len(camera_data["cameras_interp"]) == frames` (or `>=` with an explicit `--allow_camera_subsample` flag)  
3. Document companion camera generation:
   - `python examples/cameras/_make_360.py orbit --num_frames $FRAMES`  
4. Optionally add `--duration_sec` that derives frames via `fps` and snaps to nearest valid `4k+1`.  
5. Wire `FRAMES` into batch scripts and regenerate / select a matching camera JSON.  
6. Smoke-test ladder on this GPU: `81 → 97 → 129 → 161` for both Wan 2.1 and Wan 2.2; record VRAM peak and qualitative notes.

Suggested first “supported” longer targets (already named in Wan 2.2 help / `_make_360.py`):

- **97** (~6 s) — small step above default  
- **129** (~8 s) — moderate  
- **161** (~10 s) — current documented upper example  

Going past 161 is likely still runnable here given ~96 GB VRAM, but should be treated as experimental until quality is checked.

---

## 8. Conclusion

**Possible for both Wan 2.1 and Wan 2.2: yes.**

The parameter already exists as `--frames`. Longer sequences are allowed by RoPE tables and latent packing; the practical blockers are:

1. Regenerating / matching camera trajectories to the new length  
2. Keeping `frames % 4 == 1`  
3. VRAM and wall-clock cost  
4. Possible quality drop beyond the ~81-frame training regime  

No fusion-model rewrite is required to try longer clips. The useful product work is validation, camera sync, batch/README wiring, and empirical quality/VRAM checks — not inventing a new generation mode.
