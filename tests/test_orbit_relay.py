"""CPU contracts: global orbit, conditioning, provenance and real media IO."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


def test_global_ccw_orbit_closes_and_rebases_without_resetting():
    from orbit_relay import Config, camera_segments
    cfg = Config(image='unused.png', prompt='room', frames=5)
    segments = camera_segments(cfg)
    assert len(segments) == 4
    poses = np.concatenate([s['cameras_interp'][:-1] for s in segments])
    center = np.array([0, 0, 1])
    positions = poses[:, :3, 3]
    np.testing.assert_allclose(np.linalg.norm(positions-center, axis=1), 1)
    np.testing.assert_allclose(poses[:, :3, 2], center-positions, atol=1e-12)
    np.testing.assert_allclose(poses[:, :3, :3].transpose(0, 2, 1) @ poses[:, :3, :3], np.broadcast_to(np.eye(3), (16, 3, 3)), atol=1e-12)
    assert np.all(np.linalg.det(poses[:, :3, :3]) > .999)
    # View from -Y, image right=+X and up=+Z: positive oriented area is CCW.
    x, z = positions[:, 0], positions[:, 2]-1
    assert np.sum(x*np.roll(z, -1)-z*np.roll(x, -1)) > 0
    np.testing.assert_allclose(segments[0]['cameras_interp'][0], np.eye(4), atol=1e-12)
    np.testing.assert_allclose(segments[-1]['cameras_interp'][-1], np.eye(4), atol=1e-12)
    np.testing.assert_allclose(segments[1]['cameras_interp'][0], [[0,0,-1,1],[0,1,0,0],[1,0,0,1],[0,0,0,1]], atol=1e-12)
    for i, segment in enumerate(segments):
        world = np.array(segment['cameras_interp'])
        anchor = np.array(segment['anchor_c2w'])
        local = np.array(segment['local_c2w'])
        np.testing.assert_allclose(anchor @ local, world, atol=1e-12)
        if i:
            np.testing.assert_allclose(segments[i-1]['cameras_interp'][-1], world[0], atol=1e-12)


@pytest.mark.parametrize('kw', [{'angle':0}, {'angle':100}, {'angle':float('nan')}, {'radius':0}, {'frames':1}, {'frames':6}, {'seed':-1}, {'width':17}])
def test_invalid_configuration_rejected(kw):
    from orbit_relay import Config, camera_segments
    with pytest.raises(ValueError):
        camera_segments(Config(image='x', prompt='room', **kw))


def make_config(tmp_path):
    from orbit_relay import Config
    image = tmp_path/'start.png'
    Image.new('RGB', (32, 32), (15, 25, 35)).save(image)
    return Config(image=str(image), prompt='room', frames=5, width=32, height=32)


def test_plan_has_no_model_import_and_never_claims_execution(tmp_path):
    cfg = make_config(tmp_path)
    result = subprocess.run([sys.executable, 'orbit_relay.py', 'plan', '--image', cfg.image, '--prompt', 'room', '--output', str(tmp_path/'plan'), '--frames', '5', '--width', '32', '--height', '32'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path/'plan/manifest.json').read_text())
    assert manifest['status'] == 'NOT_RUN'
    assert all(s['status'] == 'NOT_RUN' for s in manifest['segments'])
    assert len(manifest['segments']) == 8
    commands = [s['command'] for s in manifest['segments']]
    assert sum('--end_image_path' in c for c in commands) == 1
    assert all(c[c.index('--using_scale')+1] == 'False' for c in commands)
    assert all(c[c.index('--frames')+1] == '5' for c in commands)


def test_relay_uses_exact_raw_last_frame_and_identical_prefix_seeds(tmp_path):
    from orbit_relay import execute, prepare
    from relay_media import write_segment
    cfg = make_config(tmp_path)
    root = tmp_path/'run'
    manifest = prepare(cfg, root, backend='test')
    observed = []
    def backend(segment):
        pixels = np.array(Image.open(segment['input']))
        observed.append((segment['experiment'], segment['index'], pixels[0,0].tolist(), segment['end_image'], segment['seed']))
        frames = [np.full((32,32,3), segment['index']*40+j, dtype=np.uint8) for j in range(5)]
        write_segment(frames, Path(segment['directory']), fps=16)
    result = execute(root, backend)
    assert result['status'] == 'TEST_ONLY'
    assert [x[2] for x in observed] == [[15,25,35], [4,4,4], [44,44,44], [84,84,84], [84,84,84]]
    assert [x[4] for x in observed] == [1024,1025,1026,1027,1027]
    assert [bool(x[3]) for x in observed] == [False]*4+[True]
    assert all(s['reused_from'] for s in result['segments'][4:7])
    assert result['segments'][3]['input_hash'] == result['segments'][7]['input_hash']
    assert observed[-1][3] == str(Path(cfg.image).resolve())
    # Fully verified resume executes nothing; corruption must not silently skip.
    execute(root, lambda s: pytest.fail('completed segment rerun'))
    Path(result['segments'][0]['last_frame']).write_bytes(b'broken')
    with pytest.raises(ValueError, match='artifact'):
        execute(root, backend)


def test_resume_rejects_changed_configuration_input_and_missing_artifacts(tmp_path):
    from orbit_relay import prepare, execute
    cfg = make_config(tmp_path)
    root = tmp_path/'run'
    original = prepare(cfg, root, backend='test')
    cfg.prompt = 'changed'
    with pytest.raises(ValueError, match='incompatible'):
        prepare(cfg, root, backend='test')
    cfg.prompt = 'room'
    Image.new('RGB', (32,32), 'red').save(cfg.image)
    with pytest.raises(ValueError, match='incompatible'):
        prepare(cfg, root, backend='test')
    # Merely having video.mp4 is not completion.
    segdir = Path(original['segments'][0]['directory'])
    (segdir/'video.mp4').write_bytes(b'not a video')
    with pytest.raises(ValueError):
        execute(root, lambda s: None)


def test_real_video_stitch_drops_only_later_first_frames(tmp_path):
    from relay_media import write_segment, stitch
    import imageio.v2 as imageio
    dirs = [tmp_path/'a', tmp_path/'b']
    for directory, values in zip(dirs, [[10,20,30,40,50], [50,60,70,80,90]]):
        write_segment([np.full((32,32,3), v, np.uint8) for v in values], directory, fps=16)
    output = tmp_path/'joined.mp4'
    stitch([d/'video.mp4' for d in dirs], output, fps=16)
    reader = imageio.get_reader(output)
    decoded = list(reader.iter_data())
    reader.close()
    assert len(decoded) == 9
    np.testing.assert_allclose([np.asarray(f).mean() for f in decoded], [10,20,30,40,50,60,70,80,90], atol=3)


def test_partial_run_resumes_without_regenerating_successful_segments(tmp_path):
    from orbit_relay import prepare, execute
    from relay_media import write_segment
    cfg = make_config(tmp_path)
    root = tmp_path/'partial'
    prepare(cfg, root, backend='test')
    def fail_second(segment):
        if segment['index'] == 1:
            raise RuntimeError('interrupted')
        write_segment(np.zeros((5,32,32,3), dtype=np.uint8), segment['directory'], fps=16)
    with pytest.raises(RuntimeError, match='interrupted'):
        execute(root, fail_second)
    assert json.loads((root/'manifest.json').read_text())['status'] == 'FAILED'
    def finish(segment):
        assert segment['id'] != 'A:0'
        write_segment(np.zeros((5,32,32,3), dtype=np.uint8), segment['directory'], fps=16)
    assert execute(root, finish)['status'] == 'TEST_ONLY'


def test_tampered_manifest_is_rejected_before_backend(tmp_path):
    from orbit_relay import prepare, execute
    cfg = make_config(tmp_path)
    root = tmp_path/'tampered'
    manifest = prepare(cfg, root, backend='test')
    manifest['identity']['config']['prompt'] = 'silently changed'
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='incompatible'):
        execute(root, lambda s: pytest.fail('tampered settings executed'))


def test_fixed_scale_survives_other_segment_sizes():
    from orbit_relay import Config, camera_segments
    segments = camera_segments(Config(image='unused', prompt='room', frames=9, angle=120, radius=2))
    assert len(segments) == 3
    for segment in segments:
        points = np.array(segment['cameras_interp'])[:, :3, 3]
        np.testing.assert_allclose(np.linalg.norm(points-[0,0,2], axis=1), 2, atol=1e-12)
    np.testing.assert_allclose(segments[-1]['cameras_interp'][-1], np.eye(4), atol=1e-12)


def test_prepare_preserves_unrelated_files_in_existing_output(tmp_path):
    from orbit_relay import prepare
    cfg = make_config(tmp_path)
    root = tmp_path/'existing'
    root.mkdir()
    keep = root/'research.txt'
    keep.write_text('KEEP')
    with pytest.raises(ValueError, match='nonempty'):
        prepare(cfg, root)
    assert keep.read_text() == 'KEEP'


def test_cpu_smoke_cli_executes_shared_prefix_and_321_frame_join(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'scripts.relay_cpu_smoke', '--output', str(tmp_path/'smoke')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary['status'] == 'TEST_ONLY'
    assert summary['gpu_effects'] == 'NOT_RUN'
    assert summary['backend_calls'] == ['A:0','A:1','A:2','A:3','B:3']
    assert summary['joined_decoded_counts'] == {'A':321, 'B':321}
    assert summary['torch_imported'] is False


def test_subprocess_preserves_selected_virtualenv_interpreter(tmp_path):
    from orbit_relay import prepare
    cfg = make_config(tmp_path)
    manifest = prepare(cfg, tmp_path/'venv-command')
    interpreter = manifest['segments'][0]['command'][0]
    result = subprocess.run([interpreter, '-c', 'import sys; print(sys.prefix)'], capture_output=True, text=True, check=True)
    # Resolving the venv Python symlink executes the base interpreter and loses
    # the selected GPU/CPU environment, even though the binary file is identical.
    assert result.stdout.strip() == sys.prefix
    assert manifest['identity']['config']['python'] == str(Path(sys.executable).absolute())
