# Scene 0000 experiment tools and preservation

This branch publishes the legacy ABC shared-prefix orchestrator plus portable
condition-only observation, bounded single-clip planning/execution, and archive
verification. It does not promote the K-only experiment to the default.

## Scientific scope

Same prepared café image, original prompt, legacy ray policy, 81 frames at
832×480/16 fps, 50 steps, radius 1, focal 500, using_scale=False, CFG 5 and
boundary 900 (the latter two remain production defaults). Nominal 60/90/360
camera matrices are not measurements of rendered-video angle, FoV or 3D closure.
The 60° single clip was less visibly degraded in this one seed/image; its tail
still ghosts. The second 60° clip redraws the seam and degrades within the clip.

A/B/C share genuine identical prefixes, not independent reruns. A has no endpoint;
B differs only in its final end image; C displays reversed CW branches, including
backwards dynamic time, and uses an already degraded teacher. It is not a pure
chain-depth ablation. CONDITION_ONLY changes the last mask/latent group and keeps
y finite, but has no denoise or model-efficacy result. Historical HIGH missing436,
strict imageio handle warnings and decord metadata limitations remain unresolved.

## Existing ABC runner

See [ABC shared prefix](abc-shared-prefix.md), [orbit relay](ccw-orbit-relay.md)
and [bidirectional relay](bidirectional-half-orbit.md). `shared_prefix_relay.py`
retains the exact original deployment/head/hash pin for GPU reference import;
it intentionally rejects a reference from a new checkout/commit. To reproduce
that historical import byte-for-byte, restore the archived original source and
runroots at their recorded paths. Do not edit an old manifest or weaken its
identity gates to force a new checkout into an old experiment. A new checkout can
plan bounded clips with the tools below; that is a new run, not a historical
resume. CPU real-media fixture tests are TEST_ONLY, never GPU proof.

## Bounded first/second 60° clips

1. Copy `examples/experiments/scene-0000-single60.template.json` outside the repo;
   set absolute prepared input, Python and model paths. Keep the paired prompt.
2. Run from repository root:

       python scripts/plan_single_clip.py --config /path/config.json --output-root /new/run01 --segment-index 0

   This only writes camera/plan JSON, no model loads. It refuses an existing root.
   For the second segment set image to run01/output/last_frame.png (the true raw
   PNG, not a decoded MP4 frame) in a separate config, preserving angle60 and
   base seed1024, then use `--segment-index 1`. The result is global60→120,
   seed1025 and an empty endpoint; this is not a same-seed ablation. Do not launch
   the full relay and interrupt it to simulate a bounded run.
3. After reviewing the generated camera, argv, pinned checkpoint manifest,
   source identity and GPU/disk availability, obtain authorization for generation.
   Explicit execution only, on a suitable GPU, in a dedicated tmux window:

       HF_HOME=/absolute/existing/cache python scripts/run_single_clip.py --root /new/run01 --execute

   Exclusive started marker prevents replay, offline flags are enforced, and raw
   logs/PID/exit/runtime/NVML samples are retained. No automatic retry/parameter
   reduction. It intentionally refuses input/source/commit/checkpoint-stat changes.
   If a launch fails, retain its partial root and diagnose before a new run.
   Stat identity is not a substitute for initial full checkpoint SHA256 checks.

## Isolated endpoint observer

Requires the production Python environment (torch2.7.1 runtime used historically),
not the lightweight relay-only environment. No DiT/text weights are loaded.

       CUDA_VISIBLE_DEVICES='' python scripts/condition_observer.py fixture --output-root /new/cpu-observer

The fixture runs real torch and the production image unit with a small test-only
VAE implementation. It checks OFF/ON hook parity and NaN detection; it is not a
real VAE inference. Actual CONDITION_ONLY measurement additionally requires all
explicit paths, HF_HOME and `--allow-vae`:

       HF_HOME=/existing/cache python scripts/condition_observer.py measure --allow-vae --output-root /new/condition --start-image /raw/S2-last.png --end-image /input/I0.png --loader-log /original/inference.log --vae-path /models/Wan2.1_VAE.pth --prompt-file /input/prompt.txt

This traverses production return_condition/image units, deliberately omits
unrelated text/noise/camera units, constructs DiT flags on meta from saved loader
kwargs, and performs OFF→ON→OFF-repeat at bf16. It does not observe a loaded
HIGH/LOW denoise forward. Difference arithmetic is bf16 subtraction then fp32.
It refuses an existing output root. Historical machine-specific launchers and
all exact argv/config/receipts are preserved in the evidence backup, not silently
replaced by these portable wrappers. No new GPU measurement was run for publication.

## Archives and results

`python scripts/evidence_archive.py create --root /results --paths-json /file-list.json --archive /new/results.tar.gz`

`python scripts/evidence_archive.py verify --archive /new/results.tar.gz --manifest /new/results.tar.gz.json --target /new/extraction`

These perform a real extraction, exact member-set and every-member byte/hash
verification, including hardlinks; no tensor deserialization. They cannot establish
visual quality or geometry. The actual delivered preservation pack has separate
human-readable experiment groups, technical evidence, complete remote/local source
mapping, and auxiliary debug/PLY archives. Public weights/cache/venv/git databases
are excluded, with exact pinned download/checksum evidence retained. No training
or new checkpoint is claimed. Results and large media are not committed here.

## Verification boundaries

The original59 CPU tests and new archive/60° planner/refusal tests run in the
relay-dev environment. Production torch observer fixture is additionally run
CPU-only in the existing production environment. Full media decode and archive
hash verification are reported independently from historical visual observations.
K-only correction remains isolated as `experiments/fantasyworld-0000-kfix-isolated`
(commit2af494b21ecaba7df76b4ac6eef5b9a2387d59d9); it was visibly worse and is not
cherry-picked here. No main merge, force push, instance stop/destroy or training.
