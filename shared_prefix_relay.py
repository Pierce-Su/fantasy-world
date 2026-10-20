"""Task-local ABC shared-prefix v1 orchestration; baseline numerical code untouched."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import orbit_relay as orbit
from bidirectional_relay import camera_paths
from relay_media import sha256, validate_segment, stitch

PROTOCOL = 'ABC_shared_prefix_legacy_v1'
BASELINE_HEAD = '3fba9e3b4fba73ea18c395fff777cb5ad57081d8'
INPUT_HASH = '9ce8d6672260b4c9a2c4416cbbe661376cb7cf11519a99f3fd27e4405fc7f73e'
PROMPT_HASH = '42817c9716a414573624f419372944591f477c9cee147a00d48fac1974c6d45e'
VIDEO_HASH = '4b7f9f8bf9617277fe47dbed3c1be155236af5264f31618699d17bc59f31a514'
LAST_HASH = '9555df9d0e65b1dca0e6ec9168de36cb0898807797213bfaffa0d30b5ba73f7e'


def inventory(cfg):
    data = orbit.identity(cfg)
    # HF lock/metadata timestamps are not weights and may be touched by the
    # unchanged offline loader. Preserve them in original evidence, not identity.
    data['checkpoint_inventory'] = {k: [x for x in rows if '.cache' not in Path(x['path']).parts]
                                    for k, rows in data['checkpoint_inventory'].items()}
    tracked = subprocess.check_output(['git','ls-files','--recurse-submodules'],cwd=orbit.REPO,text=True).splitlines()
    data['tracked_python_sha256'] = {name: sha256(orbit.REPO/name) for name in tracked
                                     if name.endswith('.py') and (orbit.REPO/name).is_file()}
    data['tracked_diff'] = subprocess.check_output(['git','diff','HEAD','--stat'],cwd=orbit.REPO,text=True)
    data['orchestration_sha256'] = sha256(__file__)
    data['c_camera_source_sha256'] = sha256(orbit.REPO/'bidirectional_relay.py')
    return data


def blueprint(cfg, root, baseline):
    ccw = orbit.camera_segments(cfg)
    if cfg.angle != 90:
        raise ValueError('shared ABC requires 90 degree quarters')
    cameras = ccw + [ccw[3]] + camera_paths(cfg)[2:]
    names = ['S0', 'S1', 'S2', 'A3', 'B3', 'R1', 'R2']
    parents = [None, 'S0', 'S1', 'S2', 'S2', None, 'R1']
    result = []
    for name, parent, camera, depth in zip(names, parents, cameras, [0,1,2,3,3,0,1]):
        directory = baseline/'output' if name == 'S0' else root/'segments'/name
        camera_path = baseline/'camera.json' if name == 'S0' else directory/'camera.json'
        end = cfg.image if name == 'B3' else str(root/'segments/S1/last_frame.png') if name == 'R2' else None
        segment = {'id': name, 'parent': parent, 'depends_on': ([parent] if parent else []) + (['S1'] if name == 'R2' else []),
                   'directory': str(directory), 'artifact_directory': str(directory), 'camera': str(camera_path),
                   'camera_sha256': orbit.digest(camera), 'input': str(Path(cfg.image).resolve()) if not parent else str((baseline/'output' if parent == 'S0' else root/'segments'/parent)/'last_frame.png'),
                   'end_image': end, 'seed': cfg.seed+depth, 'last_frame': str(directory/'last_frame.png'),
                   'import_reference': str(baseline) if name == 'S0' else None}
        segment['command'] = orbit.command(cfg, segment)
        result.append(segment)
    orders = {'A': ['S0','S1','S2','A3'], 'B': ['S0','S1','S2','B3'],
              'C': ['S0','S1','R2','R1'], 'C_CCW': ['S0','S1'], 'C_CW': ['R1','R2']}
    by_id = {s['id']: s for s in result}
    stitches = {name: [cfg.python, str(orbit.REPO/'relay_media.py'), '--fps', str(cfg.fps), '--output', str(root/(name+'.mp4'))] +
                (['--reverse','2','--reverse','3'] if name == 'C' else []) + [str(Path(by_id[s]['directory'])/'video.mp4') for s in order]
                for name, order in orders.items()}
    plan = {'segments': result, 'orders': orders, 'stitch_commands': stitches,
            'presentation': {'version': 'shared-prefix new C, supersedes standalone no-import policy',
                             'cw_time_reversed': True, 'dynamic_content_warning': 'CW events run backwards in C tour',
                             'not_pure_chain_depth_ablation': True, 'geometry_closure': 'NOT_GUARANTEED',
                             'seams': 'drop later first time sample; no PNG pixel repair'}}
    return plan, dict(zip(names, cameras))


def options(argv):
    if len(argv) < 2 or len(argv[2:]) % 2:
        raise ValueError('invalid argv')
    pairs = list(zip(argv[2::2], argv[3::2]))
    if len(dict(pairs)) != len(pairs):
        raise ValueError('duplicate argv option')
    return argv[:2], dict(pairs)


def import_proof(cfg, baseline, segment, backend):
    """Reference the original root, never copy outputs or alter its manifest."""
    provenance = json.loads((baseline/'provenance.json').read_text())
    old = provenance['identity']
    current = inventory(cfg)
    if old['config'] != current['config']:
        raise ValueError('import config mismatch')
    for field in ('input_sha256', 'source_sha256', 'scale_policy', 'ray_policy', 'commit'):
        if old[field] != current[field]:
            raise ValueError('import identity mismatch: '+field)
    expected_inventory = {k: [x for x in rows if '.cache' not in Path(x['path']).parts] for k, rows in old['checkpoint_inventory'].items()}
    if expected_inventory != current['checkpoint_inventory']:
        raise ValueError('import checkpoint mismatch')
    expected = json.loads((baseline/'argv.json').read_text())
    if segment['command'] != expected:
        raise ValueError('import argv mismatch')
    if orbit.digest(json.loads((baseline/'camera.json').read_text())) != segment['camera_sha256']:
        raise ValueError('import camera mismatch')
    if int((baseline/'inference.exit').read_text()) != 0 or json.loads((baseline/'runtime.json').read_text())['exit_code'] != 0:
        raise ValueError('import did not exit 0')
    started = json.loads((baseline/'started.json').read_text())
    if not started.get('pid') or not started.get('jobid'):
        raise ValueError('import missing started provenance')
    if not (baseline/'inference.log').is_file():
        raise ValueError('import missing original log')
    if provenance.get('backend', 'gpu') != backend:
        raise ValueError('TEST_ONLY cannot be imported as GPU')
    receipt = validate_segment(segment['directory'], cfg.frames, cfg.height, cfg.width, cfg.fps)
    if backend == 'gpu':
        if old['commit'] != BASELINE_HEAD or provenance.get('instance_id') != '56901598' or current['tracked_diff']:
            raise ValueError('unexpected/dirty GPU baseline')
        import hashlib
        if (old['input_sha256'], hashlib.sha256(cfg.prompt.encode()).hexdigest(), receipt['hashes']['video.mp4'], receipt['hashes']['last_frame.png']) != (INPUT_HASH, PROMPT_HASH, VIDEO_HASH, LAST_HASH):
            raise ValueError('original production hash mismatch')
        if (cfg.angle,cfg.radius,cfg.frames,cfg.seed,cfg.width,cfg.height,cfg.fps,cfg.focal,cfg.steps,cfg.negative_prompt) != (90,1,81,1024,832,480,16,500,50,''):
            raise ValueError('production contract mismatch')
    return {'kind': 'validated_external_reference', 'backend': backend, 'source_root': str(baseline),
            'receipt': receipt, 'evidence_hashes': {name: sha256(baseline/name) for name in
                ('provenance.json','argv.json','camera.json','runtime.json','started.json','inference.exit','inference.log')}}


def prepare(cfg, root, baseline, backend='gpu'):
    root, baseline = Path(root).resolve(), Path(baseline).resolve()
    if backend not in ('gpu','test'):
        raise ValueError('unsupported backend')
    identity = inventory(cfg)
    plan, cameras = blueprint(cfg, root, baseline)
    proof = import_proof(cfg, baseline, plan['segments'][0], backend)
    fingerprint = orbit.digest({'identity': identity, 'plan': plan, 'proof': proof, 'backend': backend, 'protocol': PROTOCOL})
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root/'manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            check(manifest, cfg, root, baseline, backend)
            return manifest
        if any(p.name != '.lock' for p in root.iterdir()):
            raise ValueError('nonempty root without manifest')
        for segment in plan['segments'][1:]:
            Path(segment['directory']).mkdir(parents=True)
            orbit.atomic_json(segment['camera'], cameras[segment['id']])
        manifest = {'protocol': PROTOCOL, 'schema_version': 1, 'backend': backend, 'identity': identity,
                    'baseline': str(baseline), 'import_proof': proof, 'fingerprint': fingerprint, **plan,
                    'status': 'NOT_RUN', 'gpu_effects': 'NOT_RUN'}
        for segment in manifest['segments']:
            segment['status'] = 'NOT_RUN'
        orbit.atomic_json(path, manifest)
        return manifest


def check(manifest, cfg, root, baseline, backend):
    current = inventory(cfg)
    plan, _ = blueprint(cfg, root, baseline)
    proof = import_proof(cfg, baseline, plan['segments'][0], backend)
    expected = orbit.digest({'identity': current, 'plan': plan, 'proof': proof, 'backend': backend, 'protocol': PROTOCOL})
    if manifest['backend'] != backend or manifest['protocol'] != PROTOCOL or manifest['fingerprint'] != expected or manifest['identity'] != current or manifest['import_proof'] != proof:
        raise ValueError('incompatible resume/import/source/backend')
    if len(manifest['segments']) != len(plan['segments']):
        raise ValueError('changed segment count')
    for actual, wanted in zip(manifest['segments'], plan['segments']):
        if any(actual.get(k) != v for k,v in wanted.items()):
            raise ValueError('changed routing/command')
        if orbit.digest(json.loads(Path(actual['camera']).read_text())) != wanted['camera_sha256']:
            raise ValueError('changed camera')
    for field in ('orders','stitch_commands','presentation'):
        if manifest[field] != plan[field]:
            raise ValueError('changed presentation')


def gpu_run(segment):
    directory = Path(segment['directory'])
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPU occupied')
    if shutil.disk_usage(directory).free < 10_000_000_000:
        raise ValueError('disk free below 10GB')
    start = time.time()
    with (directory/'started.json').open('x') as out:
        json.dump({'jobid': directory.parent.parent.name+':'+segment['id'], 'launcher_pid': os.getpid(), 'time': start},out)
    orbit.atomic_json(directory/'argv.json',segment['command'])
    with (directory/'inference.log').open('xb') as log:
        process = subprocess.Popen(segment['command'],cwd=orbit.REPO,stdout=log,stderr=subprocess.STDOUT)
        (directory/'inference.pid').write_text(str(process.pid))
        peaks = {}
        with (directory/'vram-samples.jsonl').open('x') as out:
            while True:
                text = (directory/'inference.log').read_text(errors='replace')
                phase = 'decode_export' if 'Dual-model video generation complete' in text else 'inference' if 'Generating video ...' in text else 'loading'
                raw = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip()
                rows = [dict(pid=int(a),used_MiB=int(b)) for a,b in (row.split(',') for row in raw.splitlines())]
                value = sum(row['used_MiB'] for row in rows)
                peaks[phase] = max(peaks.get(phase,0),value)
                out.write(json.dumps({'elapsed_s':time.time()-start,'phase':phase,'processes':rows})+'\n'); out.flush()
                if process.poll() is not None:
                    break
                time.sleep(1)
        rc = process.wait()
    (directory/'inference.exit').write_text(str(rc)+'\n')
    orbit.atomic_json(directory/'runtime.json',{'exit_code':rc,'elapsed_s':time.time()-start,'sampled_peak_MiB':peaks,'method':'nvidia-smi ~1s sampling; NOT allocator peak; host/container PID mapping not independently verified'})
    if rc:
        raise RuntimeError('GPU subprocess failed: '+str(rc))


def execute(root, backend=None):
    root = Path(root).resolve()
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root/'manifest.json'
        manifest = json.loads(path.read_text())
        cfg = orbit.Config(**manifest['identity']['config'])
        mode = 'gpu' if backend is None else 'test'
        check(manifest,cfg,root,Path(manifest['baseline']),mode)
        done = 'COMPLETE' if backend is None else 'TEST_ONLY'
        by_id = {s['id']:s for s in manifest['segments']}
        for segment in manifest['segments']:
            try:
                for dep in segment['depends_on']:
                    source = by_id[dep]
                    if source['status'] != done or validate_segment(source['directory'],cfg.frames,cfg.height,cfg.width,cfg.fps) != source['receipt']:
                        raise ValueError('invalid dependency')
                for field in ('input','end_image'):
                    value = sha256(segment[field]) if segment[field] else None
                    if field+'_hash' in segment and segment[field+'_hash'] != value:
                        raise ValueError('changed condition: '+field)
                    segment[field+'_hash'] = value
                if segment['import_reference']:
                    receipt = manifest['import_proof']['receipt']
                else:
                    directory = Path(segment['directory'])
                    if 'receipt' not in segment:
                        if segment['status'] != 'NOT_RUN' or any((directory/name).exists() for name in ('started.json','video.mp4','receipt.json','inference.log','inference.exit')):
                            raise ValueError('partial/started segment: inspect; never replay')
                        segment['status'] = 'RUNNING'
                        orbit.atomic_json(path,manifest)
                        print('START '+segment['id']+' '+str(directory/'inference.log'),flush=True)
                        (gpu_run if backend is None else backend)(segment)
                    if backend is None and int((directory/'inference.exit').read_text()) != 0:
                        raise ValueError('missing successful exit')
                    receipt = validate_segment(directory,cfg.frames,cfg.height,cfg.width,cfg.fps)
                if 'receipt' in segment and segment['receipt'] != receipt:
                    raise ValueError('changed receipt')
                segment['receipt'],segment['status'] = receipt,done
                orbit.atomic_json(path,manifest)
                print(done+' '+segment['id'],flush=True)
            except Exception as exc:
                segment['status'],segment['error'],manifest['status'] = 'FAILED',str(exc),'FAILED'
                orbit.atomic_json(path,manifest)
                raise
        manifest['status'],manifest['gpu_effects'] = done, 'NOT_EVALUATED' if backend is None else 'NOT_RUN'
        orbit.atomic_json(path,manifest)
        return manifest


def assemble(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    cfg = orbit.Config(**manifest['identity']['config'])
    check(manifest,cfg,root,Path(manifest['baseline']),manifest['backend'])
    done = 'COMPLETE' if manifest['backend'] == 'gpu' else 'TEST_ONLY'
    for s in manifest['segments']:
        if s['status'] != done or validate_segment(s['directory'],cfg.frames,cfg.height,cfg.width,cfg.fps) != s['receipt']:
            raise ValueError('cannot stitch incomplete or changed artifact')
    for name, argv in manifest['stitch_commands'].items():
        subprocess.run(argv,check=True,cwd=orbit.REPO)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['plan','run','stitch'])
    parser.add_argument('--root',required=True)
    parser.add_argument('--baseline',required=True)
    args = parser.parse_args()
    baseline = Path(args.baseline).resolve()
    cfg = orbit.Config(**json.loads((baseline/'provenance.json').read_text())['identity']['config'])
    prepare(cfg,args.root,baseline)
    if args.mode == 'run':
        os.environ.update(HF_HOME=str(orbit.REPO/'.hf-cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
        execute(args.root)
    elif args.mode == 'stitch':
        assemble(args.root)

if __name__ == '__main__':
    main()
