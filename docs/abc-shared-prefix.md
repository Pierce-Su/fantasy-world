# ABC shared-prefix legacy v1 — task-local orchestration

This version explicitly supersedes C's standalone/no-external-import restriction for this task. It does not modify `orbit_relay.py`, `bidirectional_relay.py`, `relay_media.py`, the inference entrypoint, loader, ray processor, adapter or model implementation. Original standalone A/B/C interfaces remain unchanged.

## Source isolation

Production inference always uses `/workspace/fantasy-world/inference_wan22.py`, cwd and PYTHONPATH `/workspace/fantasy-world`, baseline HEAD `3fba9e3b4fba73ea18c395fff777cb5ad57081d8`. The task-local orchestrator is deployed separately under `data/deployment/fw56901598-t_7cce7aa5-abc-shared-legacy-v1`. It calls the baseline command builder and camera generator. No kfix source is imported or reverted.

All original eight core source hashes must match imported evidence. Identity additionally records all tracked Python file hashes, repository HEAD, the orchestrator hash, C camera source hash, config/input hash, and original checkpoint path/size/mtime_ns inventory. HF `.cache` bookkeeping files are excluded from immutable identity because the unchanged offline loader may touch lock/metadata files; the original full inventory remains in provenance. This does not remove any weight/config file from validation and is not a new full checksum of all weights.

Policy remains `legacy_pixel_K_rescaled_again_UNVERIFIED_FOV`. Defaults CFG5/boundary900 are preserved through the unchanged entrypoint and source hashes. Do not interpret nominal camera pose closure as calibrated video FoV or geometric success.

## Authoritative DAG

    I0 -> S0 [external validated legacy reference, seed1024]
           -> S1 [CCW90..180, seed1025]
              -> S2 [CCW180..270, seed1026]
                 -> A3 [CCW270..360, seed1027, no end]
                 -> B3 [same start/camera/seed as A3; only end=I0]
    I0 -> R1 [CW0..-90, seed1024]
           -> R2 [CW-90..-180, seed1025, end=S1 raw final PNG]

R2 depends on both R1 and S1. Global poses accumulate; each segment retains its global anchor and legal local relative gauge. Fixed radius1/using_scaleFalse avoids per-generated-input scale normalization. Input is the original prepared PNG, never generated first_frame.png. Relays and end conditions use generated lossless raw uint8 PNGs, never decoded MP4 endpoints.

A = S0,S1,S2,A3. B = S0,S1,S2,B3. C tour = S0,S1,reverse(R2),reverse(R1). C_CCW and C_CW retain the two original forward-time branches. Reversal changes both clip order and frame order. Only later first time samples are dropped; no PNG is pasted to fake a seam or closure. CW events run backwards in C's tour. C changes direction, chain depth and endpoint conditioning: not a pure chain-depth ablation.

## Import and resume safety

`prepare` creates a new immutable plan/fingerprint, verifies exact baseline config/argv, original camera, source hashes, weights inventory, started/log/runtime/exit0, real full media decode and raw endpoint hashes. Production import is pinned to the original instance/HEAD/input/prompt/video/raw-final hashes. It references the original output directory without copying videos or altering old manifests. CPU fixtures explicitly label their provenance as test; they cannot become GPU imports/results.

`execute` reconstructs the plan, checks dependency receipts and both input/end-image hashes, then runs only unique uncompleted nodes. Recorded argv, routing, seeds, camera and stitch commands cannot be changed on resume. A started/partial segment without a committed receipt fails closed; it is never silently replayed. File locks and exclusive started markers protect execution. Each new GPU subprocess preserves PID, argv, log, exit, timing and sampled VRAM. Sampling is ~1s nvidia-smi, not torch allocator peak; direct host/container PID mapping is not asserted.

## Execution and verification

Production job is bounded, serial, and owns only `hermes:fw-0000-abc` plus its read-only progress window `hermes:fw-0000-abc-progress`. Logs persist independently of tmux. Do not rerun `pipeline.py`: its exclusive job marker intentionally rejects replay. If recovering, first inspect existing job/PIDs/markers and the authoritative manifest; never assume a worker interruption means the GPU stopped.

CPU regression commands (from this isolated source tree):

    /home/pierce/Documents/research/E3DQA/fantasy-world/.venv/bin/python -m pytest -q
    /home/pierce/Documents/research/E3DQA/fantasy-world/.venv/bin/python -m pytest -q -W error --tb=short

The normal suite passed59 tests locally and remotely; new real-media tests cover shared DAG, gauge/meeting poses, import mismatch refusal, TEST_ONLY/GPU separation, immutable resume/routing, crash no-replay, end-condition corruption and actual reverse pixel order. A separately retained synthetic shared-DAG smoke imported one test fixture and made six test-backend calls, then zero on resume, with no torch import. These are control-flow tests, not GPU readiness or model-quality evidence.

The strict local diagnostic remains red: 15 failed,44 passed,1 error. Existing imageio-ffmpeg unclosed subprocess handles and fixture/NumPy warnings are retained, not suppressed. Historical decord metadata failure is also retained. See evidence logs for exact outputs.

Final acceptance requires all unique real inference exits0; independent receipt/hash/full-decode/frame/fps checks on segments and manifest-produced joins; actual downloaded A/B/C and both C branches; current source/weight checks and explicit visual limitations. Starting the detached pipeline is not completion. See task handoff/validation for measured results, not this design document.
