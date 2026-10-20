# Wan2.2 CCW 360 relay (CPU-tested prototype)

GPU imports, model inference and visual/geometry quality: **NOT_RUN** on the mini PC.
This is an appearance-endpoint experiment, not a geometry-only anchor or memory implementation.
90 degrees is the chosen default, **not** a guarantee of non-OOD behavior.

For the separate C four-call protocol (two half-orbits, L2 endpoint conditioning
R2, disk-spooled CW reversal), see [C protocol](bidirectional-half-orbit.md).
C does not import A/B outputs or modify their manifests. The shared media source
extension changes its source hash: older manifests retain their recorded hashes
and will reject resume under changed source, as designed; use the preserved old
source or a new root, never rewrite an old fingerprint.

## A/B protocol

- A: four 90-degree orbit segments, each 81 frames by default.
- B: exactly the same first three segments (artifact references to A, not regenerated), then a separate final segment whose **only changed condition** is `end_image_path=original starting image`.
- Every segment starts from the previous segment's **raw generated final uint8 frame**, saved losslessly as `last_frame.png`, not a keyframe or MP4 decode. No generated last frame is overwritten to manufacture closure.
- Per-segment seed is `base_seed + zero_based_segment_index`; A/B final segments use the same seed and same input PNG hash. Shared-prefix input/camera hashes and seeds are checked before reuse. Shared output receipts are retained in both manifest branches.
- Same prompt, negative prompt, checkpoint paths, steps, frames, dimensions and camera trajectory for A/B (separate camera JSON paths, identical bytes per corresponding segment). The existing sampler CFG=5.0 and timestep boundary=900 defaults remain shared, captured by the inference source hash.
- Non-default segment angles must divide 360; frames must be >=5 and `4k+1`. Dimensions are positive multiples of 16. `--angle 120` gives 3 segments, `--angle 45` gives 8.

### Coordinates and fixed scale

OpenCV c2w matrices: camera +X right, +Y down, +Z forward. World up is -Y.
View CCW **from negative world Y toward +Y**, with screen right=+X, screen up=+Z.
For increasing theta, center O=(0,0,r), position p=(r sin(theta),0,r(1-cos(theta))),
forward=(-sin(theta),0,cos(theta)). At 0 degrees c2w=identity. At 360 degrees the
entire position and orientation close numerically. This is orbit, not stationary spin.

Each camera JSON uses the existing `focal_length`, `cameras`, `cameras_interp` schema
and contains the globally accumulated c2w samples, plus `anchor_c2w=G_s=C_s(0)`
and diagnostic `local_c2w=inv(G_s) C_s(t)`. Tests verify `G_s @ local_c2w == global_c2w`,
radius, look-at orientation, SO(3), positive CCW area and segment boundaries.
The existing pose processor legitimately re-expresses each segment relative to its
first frame. Identical local quarter orbits are therefore legal; the global ledger
is what distinguishes the accumulated world trajectory.

All relay commands explicitly set `--using_scale False`. Otherwise the existing
sampler would independently infer a new MoGe scale from each generated segment's
first image and divide the translations by different scales. Radius here is a
**fixed numerical unit**, not meters or a learned/training-scale equivalence.
The JSON `scale` field is ignored by the legacy loader. MoGe still loads/runs in
the existing sampler even with scale normalization disabled; this prototype does
not remove it or promise lower GPU requirements.

### Important legacy ray limitation

This implementation deliberately leaves the common A/B model ray path unchanged.
The pose decoder returns pixel K, but `dataset_re10k.py`'s direct path multiplies
fx/cx by W and fy/cy by H again before `ray_condition`. The decoder also changes
principal point from `(W-1)/2,(H-1)/2` to `W/2,H/2`. Manifest `ray_policy` is
`legacy_pixel_K_rescaled_again_UNVERIFIED_FOV`. Numeric global pose closure and
loader tests **do not verify correct model FoV**. The CPU tests extract and execute
the actual parser/loader/call expressions without loading GPU modules; they are
not a full tensor/Pluecker integration run. Any future correction must be shared
by A/B, versioned and validated at the actual ray input, not applied to just B.

## Isolated CPU environment

Preferred: an approved Python 3.11 interpreter (3.10 also matches upstream guidance).
No apt/system Python changes, CUDA packages or model downloads are needed.

```bash
PYTHON_BIN=python3.11 bash scripts/setup_relay_dev.sh
.venv/bin/python -m pytest
.venv/bin/python orbit_relay.py --help
.venv/bin/python relay_media.py --help
```

On this mini PC, Python 3.11 was unavailable. Installing uv 0.9.4 into repository
`.tools/uv-package` succeeded; uv-managed CPython download requests were refused
by the headless package threat-intelligence safety gate (incomplete verification,
not a finding of maliciousness). No bypass or security configuration change was
attempted. The working fallback is a repo `.venv` using existing **Python 3.14.7**:

```bash
PYTHON_BIN=python bash scripts/setup_relay_dev.sh
```

The script preserves an existing `.venv` and prints the interpreter actually used;
`PYTHON_BIN` only selects the interpreter when creating a new environment.
Pinned CPU packages are in `requirements-relay-dev.txt`. Python >=3.14 uses
numpy 2.2.6 (locally built from source); older Python uses numpy 1.26.4.
This CPU environment is **not** the GPU runtime (which requires numpy<2 plus
PyTorch/CUDA and other upstream dependencies). The .tools and .venv directories
are ignored by git; no full requirements.txt installation was performed here.

Tests cover geometry, nondefault angles/frames, real camera-loader schema inversion,
actual conditioning call forwarding of num_frames, frame validation before model
initialization, raw PNG export from the actual inference main with a test sampler,
shared-prefix branching, partial resume, changed input/settings, corrupted artifacts,
and real H.264 video concatenation. Test backends are Python callbacks only;
results are **TEST_ONLY**, never GPU COMPLETE.

For a retained, clearly synthetic end-to-end fixture (five backend invocations,
four 81-frame clips per branch with shared prefix, 321-frame joins):

```bash
.venv/bin/python -m scripts.relay_cpu_smoke --output output-relay/cpu-integration
```

Use a new empty root for each smoke run. This writes uniform-color CPU fixtures,
**not model-generated scene videos**. The normal full suite passes, but an extra
`pytest -W error` diagnostic fails three media tests because imageio-ffmpeg 0.6.0
leaves subprocess stdin/stdout handles unclosed when the decoder process has
already exited. We call reader.close() in finally blocks; the dependency's
cleanup only closes these handles inside `if process.poll() is None`.
This upstream ResourceWarning/PytestUnraisableExceptionWarning remains unresolved;
we did not modify site-packages or silently suppress it. It does not invalidate
the separately checked frame counts, but strict warning-clean verification is red.

## CPU plan / dry run

Run from the repo root. Use a fresh output root. This writes cameras and manifest,
not videos or completed inference records. There are no model imports.

```bash
.venv/bin/python orbit_relay.py plan \
  --image examples/images/input_image.png \
  --prompt 'A continuous counter-clockwise orbit around the same scene.' \
  --output output-relay/plan
```

The output is `status=NOT_RUN`, `gpu_effects=NOT_RUN`. Every manifest segment
contains its command argv, camera hash, input path, parent, seed, output directory,
raw endpoint path and reuse source. After execution it also contains input hash,
validated receipt and status. Config/input/source hashes and repository HEAD are
recorded. Local checkpoint inventory records path/size/mtime, **not full weight
content hashes**; treat weight files as immutable. Do not replace weights in place
while preserving size/mtime. The plan cannot attest to missing model files.

## Future GPU runtime and execution (not performed)

Provision a separate Python 3.10/3.11 GPU environment on an authorized GPU host,
following the upstream README and compatible CUDA/PyTorch requirements. Install
`requirements.txt` and `thirdparty/utils3d` there; do not install the Python 3.14
CPU requirements into it. Stage the required Wan2.2, FantasyWorld high/low,
LoRA, tokenizer/text encoder, VAE and MoGe assets separately after authorization.
This prototype preflights the three explicit checkpoint paths, but it does not
validate completeness of all nested assets. Upstream loaders may attempt network
access if assets are absent. Use a prepopulated cache and HF_HUB_OFFLINE=1 plus
TRANSFORMERS_OFFLINE=1 if no network/download is authorized. Model imports remain
untested here. Start with a separately authorized smoke run before a full pair.

Use the CPU orchestration interpreter and explicitly select the GPU interpreter
for subprocesses (absolute paths recommended):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python orbit_relay.py run \
  --image /absolute/starting.png --prompt 'A continuous counter-clockwise orbit around the same scene.' \
  --output /absolute/experiment-pair \
  --python /absolute/gpu-env/bin/python \
  --wan-ckpt-path /absolute/Wan2.2 \
  --model-ckpt-high /absolute/FantasyWorld/high.pth \
  --model-ckpt-low /absolute/FantasyWorld/low.pth
```

For four segments this invokes `inference_wan22.py` five times (shared A prefix,
A final, B final), sequentially, with repo cwd and per-segment logs. The existing
inference code still loads the two large models on every invocation. It now passes
`num_frames` into the real conditioning pipe, checks frame/camera counts before
sampler construction, saves lossless endpoints and receipt, and writes its legacy
`debug.pth` tuple inside the segment output (schema unchanged; not a lifting fix).
Raw endpoints/receipt are saved before the legacy PLY export. A later PLY failure
still causes subprocess failure; a receipt alone is not a successful process.

### Resume and safety

Repeat **exactly the same command** and root to resume. Same-root planning does
not reset statuses. Changes in settings, input image bytes, relevant source hashes,
commit or checkpoint inventory reject reuse. Use a new root for another protocol.
A file lock prevents concurrent writers. JSON manifest writes are atomic.

Completed segments are skipped only after endpoint/video hashes, receipt dimensions,
actual video frame count and fps are validated. A video file alone is never enough.
Corrupt committed artifacts fail rather than regenerate silently. A failed segment
with no media artifacts can be retried, retaining all previous completed segments.
If interrupted after media files were written but before a successful manifest commit,
the runner conservatively refuses to overwrite or adopt these **partial artifacts**;
inspect/preserve them and use a new root for a clean retry. Automatic recovery of that
crash window is intentionally not implemented. Nonempty roots without a manifest are
also refused. Do not hand-edit manifests to force a result to COMPLETE.

### Video assembly

The manifest includes complete `stitch_commands` argv for A and B, including shared
prefix paths. Execute each using its recorded interpreter or equivalent CPU env:

```bash
.venv/bin/python relay_media.py --fps 16 --output output-relay/plan/A/joined.mp4 \
  output-relay/plan/A/segment_000/video.mp4 \
  output-relay/plan/A/segment_001/video.mp4 \
  output-relay/plan/A/segment_002/video.mp4 \
  output-relay/plan/A/segment_003/video.mp4
```

The tool keeps all first-segment frames and removes only the first frame of every
subsequent segment: four 81-frame segments give 321 frames. Equal boundary poses
do not guarantee equal generated RGB; this is overlapping-time-sample removal,
not visual seam repair. Original per-segment files remain intact. Join uses a
streaming decode/re-encode and checks dimensions/fps; it does not copy encoded
packets or guarantee lossless concatenation. Existing joined output is not overwritten.
A final-image resemblance can support appearance closure, not geometry ground truth,
pose tracking accuracy or world consistency. Inspect the full final transition too.
