"""CPU-safe planner and resumable A/B CCW orbit relay for inference_wan22.py."""
import argparse
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
from relay_media import sha256, validate_segment

REPO = Path(__file__).resolve().parent
SOURCE_FILES = ['orbit_relay.py', 'relay_media.py', 'inference_wan22.py', 'utils.py',
                'FantasyWorld/diffsynth_wan22/data/dataset_re10k.py',
                'FantasyWorld/vggt/utils/pose_enc.py', 'FantasyWorld/fusion/model_wan22.py',
                'FantasyWorld/diffsynth_wan22/pipelines/wan_video_new.py']
RAY_POLICY = 'legacy_pixel_K_rescaled_again_UNVERIFIED_FOV'


@dataclass
class Config:
    image: str
    prompt: str
    angle: float = 90
    radius: float = 1
    frames: int = 81
    seed: int = 1024
    width: int = 832
    height: int = 480
    fps: int = 16
    focal: float = 500
    steps: int = 50
    negative_prompt: str = ''
    wan_ckpt_path: str = 'models/Wan2.2'
    model_ckpt_high: str = 'models/FantasyWorld/high.pth'
    model_ckpt_low: str = 'models/FantasyWorld/low.pth'
    python: str = sys.executable


def camera_segments(cfg):
    if not math.isfinite(cfg.angle) or not 0 < cfg.angle <= 360:
        raise ValueError('angle must be finite, positive and at most 360')
    count = round(360 / cfg.angle)
    if count > 360 or not math.isclose(count * cfg.angle, 360, abs_tol=1e-8):
        raise ValueError('angle must divide 360 (at most 360 segments)')
    if not math.isfinite(cfg.radius) or cfg.radius <= 0 or not math.isfinite(cfg.focal) or cfg.focal <= 0:
        raise ValueError('radius and focal must be finite and positive')
    if cfg.frames < 5 or cfg.frames % 4 != 1:
        raise ValueError('frames must be 4k+1, k>=1')
    if cfg.seed < 0 or cfg.fps <= 0 or cfg.steps <= 0:
        raise ValueError('nonnegative fixed seed and positive fps/steps required')
    if cfg.width <= 0 or cfg.height <= 0 or cfg.width % 16 or cfg.height % 16:
        raise ValueError('dimensions must be positive multiples of 16')
    segments = []
    for segment in range(count):
        poses = []
        for frame in range(cfg.frames):
            theta = math.radians(cfg.angle * (segment + frame/(cfg.frames-1)))
            c, s = math.cos(theta), math.sin(theta)
            poses.append([[c,0,-s,cfg.radius*s], [0,1,0,0],
                          [s,0,c,cfg.radius*(1-c)], [0,0,0,1]])
        poses = np.asarray(poses)
        anchor = poses[0]
        local = np.linalg.inv(anchor) @ poses
        segments.append({'focal_length': cfg.focal, 'scale': 1,
                         'cameras': [poses[0].tolist(), poses[-1].tolist()],
                         'cameras_interp': poses.tolist(), 'anchor_c2w': anchor.tolist(),
                         'local_c2w': local.tolist()})
    return segments


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def identity(cfg):
    config = asdict(cfg)
    config['image'] = str(Path(cfg.image).resolve())
    for key in ('wan_ckpt_path', 'model_ckpt_high', 'model_ckpt_low'):
        config[key] = str(Path(config[key]).resolve())
    # A venv interpreter is commonly a symlink. Resolving it loses pyvenv.cfg
    # discovery and silently runs the base environment instead of the chosen one.
    config['python'] = str(Path(cfg.python).expanduser().absolute())
    with Image.open(config['image']) as image:
        image.verify()
    # No weights are loaded or downloaded. Record local file identity when present.
    checkpoints = {}
    for key in ('wan_ckpt_path', 'model_ckpt_high', 'model_ckpt_low'):
        path = Path(config[key])
        files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
        checkpoints[key] = [{'path': str(p), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                            for p in files if p.is_file()]
    return {'config': config, 'input_sha256': sha256(config['image']),
            'source_sha256': {name: sha256(REPO/name) for name in SOURCE_FILES},
            'checkpoint_inventory': checkpoints, 'scale_policy': 'fixed_global_radius_using_scale_false',
            'ray_policy': RAY_POLICY, 'prefix_policy': 'shared_A_prefix_B_final_branch',
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()}


def command(cfg, segment):
    cmd = [str(Path(cfg.python).expanduser().absolute()), str(REPO/'inference_wan22.py')]
    options = {'image_path': segment['input'], 'prompt': cfg.prompt, 'neg_prompt': cfg.negative_prompt,
               'camera_json_path': segment['camera'], 'output_dir': segment['directory'],
               'wan_ckpt_path': str(Path(cfg.wan_ckpt_path).resolve()),
               'model_ckpt_high': str(Path(cfg.model_ckpt_high).resolve()),
               'model_ckpt_low': str(Path(cfg.model_ckpt_low).resolve()),
               'using_scale': 'False', 'frames': cfg.frames, 'seed': segment['seed'],
               'height': cfg.height, 'width': cfg.width, 'fps': cfg.fps, 'sample_steps': cfg.steps,
               'extract_frames': 'False'}
    if segment['end_image']:
        options['end_image_path'] = segment['end_image']
    for key, value in options.items():
        cmd.extend(['--'+key, str(value)])
    return cmd


def prepare(cfg, root, backend='gpu'):
    cameras = camera_segments(cfg)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = identity(cfg)
        fingerprint = digest({'identity': current, 'backend': backend})
        path = root/'manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest['fingerprint'] != fingerprint:
                raise ValueError('incompatible resume: configuration/input/source/checkpoint identity changed')
            return manifest
        if any(p.name != '.lock' for p in root.iterdir()):
            raise ValueError('nonempty output root without manifest; refusing to overwrite existing research files')
        segments = []
        count = len(cameras)
        for experiment in ('A', 'B'):
            previous = None
            for index, camera in enumerate(cameras):
                directory = root/experiment/f'segment_{index:03d}'
                directory.mkdir(parents=True, exist_ok=True)
                camera_path = directory/'camera.json'
                atomic_json(camera_path, camera)
                reused = f'A:{index}' if experiment == 'B' and index < count-1 else None
                artifact_dir = root/'A'/f'segment_{index:03d}' if reused else directory
                segment = {'id': f'{experiment}:{index}', 'experiment': experiment, 'index': index,
                           'status': 'NOT_RUN', 'reused_from': reused, 'parent': previous,
                           'directory': str(directory), 'artifact_directory': str(artifact_dir),
                           'camera': str(camera_path), 'camera_sha256': sha256(camera_path),
                           'input': str(Path(cfg.image).resolve()) if index == 0 else segments[-1]['last_frame'],
                           'last_frame': str(artifact_dir/'last_frame.png'),
                           'end_image': str(Path(cfg.image).resolve()) if experiment == 'B' and index == count-1 else None,
                           'seed': cfg.seed+index}
                segment['command'] = command(cfg, segment)
                segments.append(segment)
                previous = segment['id']
        manifest = {'schema_version': 1, 'backend': backend, 'status': 'NOT_RUN',
                    'identity': current, 'fingerprint': fingerprint, 'segments': segments,
                    'gpu_effects': 'NOT_RUN', 'seed_policy': 'base_plus_segment_index_same_for_A_B',
                    'stitch_commands': {e: [cfg.python, str(REPO/'relay_media.py'), '--fps', str(cfg.fps),
                                           '--output', str(root/e/'joined.mp4')]+
                                          [str(Path(s['artifact_directory'])/'video.mp4') for s in segments if s['experiment']==e]
                                        for e in ('A', 'B')}}
        atomic_json(path, manifest)
        return manifest


def execute(root, backend=None):
    root = Path(root).resolve()
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root/'manifest.json'
        manifest = json.loads(path.read_text())
        cfg = Config(**manifest['identity']['config'])
        current = identity(cfg)
        if (digest(current) != digest(manifest['identity']) or
                digest({'identity': current, 'backend': manifest['backend']}) != manifest['fingerprint']):
            raise ValueError('incompatible resume: configuration/input/source/checkpoint identity changed')
        if (backend is None) != (manifest['backend'] == 'gpu'):
            raise ValueError('test backend cannot be used for GPU results')
        if backend is None:
            for key in ('wan_ckpt_path', 'model_ckpt_high', 'model_ckpt_low'):
                if not Path(getattr(cfg, key)).exists():
                    raise ValueError(f'missing local checkpoint: {key}; no automatic download authorized')
        by_id = {s['id']: s for s in manifest['segments']}
        done = 'COMPLETE' if backend is None else 'TEST_ONLY'
        for segment in manifest['segments']:
            if segment['command'] != command(cfg, segment):
                raise ValueError('incompatible stored command')
            if sha256(segment['camera']) != segment['camera_sha256']:
                raise ValueError('camera artifact changed')
            input_hash = sha256(segment['input'])
            if segment.get('input_hash', input_hash) != input_hash:
                raise ValueError('input artifact changed')
            segment['input_hash'] = input_hash
            try:
                if segment['reused_from']:
                    source = by_id[segment['reused_from']]
                    # A/B may share only truly identical conditioning, camera and seed.
                    if (source['input_hash'], source['camera_sha256'], source['seed'], source['end_image']) != (input_hash, segment['camera_sha256'], segment['seed'], segment['end_image']):
                        raise ValueError('incompatible shared prefix')
                elif 'receipt' not in segment:
                    # Preserve interrupted research outputs rather than overwriting or
                    # accidentally treating stale video/receipt as this attempt's result.
                    if any((Path(segment['directory'])/name).exists() for name in
                           ('video.mp4', 'receipt.json', 'first_frame.png', 'last_frame.png')):
                        raise ValueError('partial artifact requires inspection; use a new output root to retry safely')
                    segment['status'] = 'RUNNING'
                    atomic_json(path, manifest)
                    if backend is None:
                        with (Path(segment['directory'])/'inference.log').open('w') as log:
                            subprocess.run(segment['command'], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
                    else:
                        backend(segment)
                receipt = validate_segment(segment['artifact_directory'], cfg.frames, cfg.height, cfg.width, cfg.fps)
                if 'receipt' in segment and segment['receipt'] != receipt:
                    raise ValueError('artifact receipt changed')
                segment['receipt'] = receipt
                segment['status'] = done
                segment.pop('error', None)
                atomic_json(path, manifest)
            except Exception as exc:
                segment['status'] = 'FAILED'
                segment['error'] = str(exc)
                manifest['status'] = 'FAILED'
                atomic_json(path, manifest)
                raise
        manifest['status'] = done
        # A successful process is not evidence of orbit quality or visual closure.
        manifest['gpu_effects'] = 'NOT_EVALUATED' if backend is None else 'NOT_RUN'
        atomic_json(path, manifest)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan', 'run'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--prompt', required=True)
    for name in ('angle', 'radius', 'focal'):
        parser.add_argument('--'+name, type=float, default=getattr(Config, name))
    for name in ('frames', 'seed', 'width', 'height', 'fps', 'steps'):
        parser.add_argument('--'+name, type=int, default=getattr(Config, name))
    for name in ('negative_prompt', 'wan_ckpt_path', 'model_ckpt_high', 'model_ckpt_low', 'python'):
        parser.add_argument('--'+name.replace('_','-'), default=getattr(Config, name))
    args = vars(parser.parse_args())
    mode, output = args.pop('mode'), args.pop('output')
    try:
        manifest = prepare(Config(**args), output)
        if mode == 'run':
            manifest = execute(output)
        print(json.dumps({'manifest': str(Path(output).resolve()/'manifest.json'), 'status': manifest['status'],
                          'gpu_effects': manifest['gpu_effects']}))
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
