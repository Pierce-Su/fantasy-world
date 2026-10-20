"""CPU-safe C: two half-orbits, with only R2 conditioned on completed L2.

Standalone four-call protocol. Cross-experiment A/B import is not supported.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from orbit_relay import (Config, REPO, atomic_json, camera_segments, command,
                         digest, identity as ab_identity)
from relay_media import sha256, validate_segment

PROTOCOL = 'C_directional_half_orbit_L2_teacher_R2_endpoint_v1'


def camera_paths(cfg):
    if cfg.angle != 90:
        raise ValueError('C supports exactly --angle 90; other segment angles are unsupported')
    left = camera_segments(cfg)[:2]
    # Reflection conjugation maps theta to -theta while retaining proper SO(3).
    reflection = np.diag([-1, 1, 1, 1])
    right = []
    for camera in left:
        mirrored = dict(camera)
        for key in ('cameras', 'cameras_interp', 'anchor_c2w', 'local_c2w'):
            mirrored[key] = (reflection @ np.asarray(camera[key]) @ reflection).tolist()
        right.append(mirrored)
    return left+right


def identity(cfg):
    current = ab_identity(cfg)
    current['prefix_policy'] = 'standalone_C_no_external_import'
    current['protocol'] = PROTOCOL
    current['camera_convention'] = 'OpenCV_c2w_world_up_minus_Y_CCW_view_from_minus_Y_screen_right_X_up_Z'
    current['loader_intrinsics'] = [[cfg.focal, 0, (cfg.width-1)/2],
                                    [0, cfg.focal, (cfg.height-1)/2], [0, 0, 1]]
    current['source_sha256']['bidirectional_relay.py'] = sha256(REPO/'bidirectional_relay.py')
    return current


def blueprint(cfg, root):
    """Deterministic immutable plan; regenerated to reject altered routing/commands."""
    cameras = camera_paths(cfg)
    segments = []
    for index, (name, camera) in enumerate(zip(('L1', 'L2', 'R1', 'R2'), cameras)):
        directory = root/'C'/name
        parent = {'L2': 'L1', 'R2': 'R1'}.get(name)
        source = root/'C'/parent/'last_frame.png' if parent else Path(cfg.image).resolve()
        end = root/'C'/'L2'/'last_frame.png' if name == 'R2' else None
        segment = {'id': name, 'experiment': 'C', 'index': index,
                   'parent': parent, 'depends_on': {'L1': [], 'L2': ['L1'], 'R1': [], 'R2': ['R1', 'L2']}[name],
                   'directory': str(directory), 'artifact_directory': str(directory),
                   'camera': str(directory/'camera.json'),
                   'camera_sha256': hashlib.sha256((json.dumps(camera, indent=2, allow_nan=False)+'\n').encode()).hexdigest(),
                   'input': str(source), 'input_source': parent or 'I0',
                   'end_image': str(end) if end else None, 'end_image_source': 'L2' if end else None,
                   'last_frame': str(directory/'last_frame.png'), 'reused_from': None,
                   'seed': cfg.seed+index % 2}
        segment['command'] = command(cfg, segment)
        segments.append(segment)
    videos = {s['id']: str(Path(s['directory'])/'video.mp4') for s in segments}
    orders = {'CCW': ['L1', 'L2'], 'CW': ['R1', 'R2'], 'tour': ['L1', 'L2', 'R2', 'R1']}
    stitches = {}
    for name, order in orders.items():
        cmd = [str(Path(cfg.python).expanduser().absolute()), str(REPO/'relay_media.py'),
               '--fps', str(cfg.fps), '--output', str(root/'C'/f'{name}.mp4')]
        if name == 'tour':
            cmd += ['--reverse', '2', '--reverse', '3']
        stitches[name] = cmd+[videos[key] for key in order]
    presentation = {'order': ['L1', 'L2', 'reverse(R2)', 'reverse(R1)'],
                    'cw_time_reversed': True, 'dynamic_content_warning': 'CW scene events also run backwards; not continuous forward scene time',
                    'seam_policy': 'drop_later_first_time_sample_no_pixel_repair',
                    'geometry_closure': 'NOT_GUARANTEED', 'teacher': 'L2_to_R2_asymmetric'}
    return {'segments': segments, 'stitch_commands': stitches, 'presentation': presentation}, cameras


def checked_plan(manifest, cfg, root, backend):
    if manifest['protocol'] != PROTOCOL or manifest['schema_version'] != 1:
        raise ValueError('incompatible C protocol/schema')
    if backend not in ('gpu', 'test') or manifest['backend'] != backend:
        raise ValueError('incompatible backend: TEST_ONLY cannot become GPU results')
    current = identity(cfg)
    plan, _ = blueprint(cfg, root)
    fingerprint = digest({'identity': current, 'backend': backend, 'plan': plan})
    if manifest['fingerprint'] != fingerprint or manifest['identity'] != current:
        raise ValueError('incompatible resume: configuration/input/source/checkpoint identity changed')
    if len(manifest['segments']) != len(plan['segments']):
        raise ValueError('incompatible segment plan')
    for actual, expected in zip(manifest['segments'], plan['segments']):
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError('incompatible stored segment plan')
        if sha256(actual['camera']) != expected['camera_sha256']:
            raise ValueError('camera artifact changed')
    for key in ('stitch_commands', 'presentation'):
        if manifest[key] != plan[key]:
            raise ValueError('incompatible presentation plan')


def prepare(cfg, root, backend='gpu'):
    root = Path(root).resolve()
    plan, cameras = blueprint(cfg, root)  # Reject unsupported settings before creating root.
    if backend not in ('gpu', 'test'):
        raise ValueError('unsupported backend')
    current = identity(cfg)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root/'manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            checked_plan(manifest, cfg, root, backend)
            return manifest
        if any(p.name != '.lock' for p in root.iterdir()):
            raise ValueError('nonempty output root without manifest; refusing to overwrite existing research files')
        fingerprint = digest({'identity': current, 'backend': backend, 'plan': plan})
        for segment, camera in zip(plan['segments'], cameras):
            Path(segment['directory']).mkdir(parents=True)
            atomic_json(segment['camera'], camera)
            segment['status'] = 'NOT_RUN'
        manifest = {'schema_version': 1, 'protocol': PROTOCOL, 'backend': backend,
                    'identity': current, 'fingerprint': fingerprint, **plan,
                    'status': 'NOT_RUN', 'gpu_effects': 'NOT_RUN',
                    'seed_policy': 'base_plus_branch_depth_L1_R1_base_L2_R2_base_plus_one',
                    'external_import': 'UNSUPPORTED_standalone_four_calls'}
        atomic_json(path, manifest)
        return manifest


def execute(root, backend=None):
    root = Path(root).resolve()
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root/'manifest.json'
        manifest = json.loads(path.read_text())
        cfg = Config(**manifest['identity']['config'])
        checked_plan(manifest, cfg, root, 'gpu' if backend is None else 'test')
        if backend is None:
            for key in ('wan_ckpt_path', 'model_ckpt_high', 'model_ckpt_low'):
                if not Path(getattr(cfg, key)).exists():
                    raise ValueError(f'missing local checkpoint: {key}; no automatic download authorized')
        by_id = {s['id']: s for s in manifest['segments']}
        done = 'COMPLETE' if backend is None else 'TEST_ONLY'
        for segment in manifest['segments']:
            try:
                for dependency in segment['depends_on']:
                    source = by_id[dependency]
                    if source['status'] != done:
                        raise ValueError('dependency not validly completed')
                    if validate_segment(source['directory'], cfg.frames, cfg.height, cfg.width, cfg.fps) != source['receipt']:
                        raise ValueError('dependency artifact receipt changed')
                # Hash and bind BOTH conditions before invoking R2, including resume.
                for field in ('input', 'end_image'):
                    value = sha256(segment[field]) if segment[field] else None
                    key = field+'_hash'
                    if key in segment and segment[key] != value:
                        raise ValueError(f'{field} artifact changed')
                    segment[key] = value
                if 'receipt' not in segment:
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
                receipt = validate_segment(segment['directory'], cfg.frames, cfg.height, cfg.width, cfg.fps)
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
        parser.add_argument('--'+name.replace('_', '-'), default=getattr(Config, name))
    args = vars(parser.parse_args())
    mode, output = args.pop('mode'), args.pop('output')
    try:
        manifest = prepare(Config(**args), output)
        if mode == 'run':
            manifest = execute(output)
        print(json.dumps({'manifest': str(Path(output).resolve()/'manifest.json'),
                          'status': manifest['status'], 'gpu_effects': manifest['gpu_effects']}))
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
