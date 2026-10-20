"""C contracts, exercised entirely on CPU with deliberately nonmatching endpoints."""
import json
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import pytest

from test_orbit_relay import make_config


def synthetic(segment):
    from relay_media import write_segment
    base = {'L1': 10, 'L2': 60, 'R1': 110, 'R2': 160}[segment['id']]
    write_segment([np.full((32, 32, 3), base+j*5, np.uint8) for j in range(5)],
                  segment['directory'], 16)


def test_c_global_pose_sign_gauge_scale_and_meeting(tmp_path):
    from bidirectional_relay import prepare
    from orbit_relay import camera_segments
    cfg = make_config(tmp_path)
    cfg.radius = 2
    m = prepare(cfg, tmp_path/'c', backend='test')
    cams = [json.loads(Path(s['camera']).read_text()) for s in m['segments']]
    np.testing.assert_allclose(cams[1]['cameras_interp'][-1], cams[3]['cameras_interp'][-1], atol=1e-12)
    assert cams[0]['cameras_interp'][1][0][3] > 0
    assert cams[2]['cameras_interp'][1][0][3] < 0
    for c in cams:
        world = np.array(c['cameras_interp'])
        np.testing.assert_allclose(np.array(c['anchor_c2w']) @ c['local_c2w'], world, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(world[:, :3, 3]-[0, 0, 2], axis=1), 2)
        np.testing.assert_allclose(world[:, :3, 2], ([0, 0, 2]-world[:, :3, 3])/2, atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(world[:, :3, :3]), 1)
        assert c['focal_length'] == cfg.focal
    # L prefix uses the unchanged A/B camera representation, not a second gauge.
    assert cams[:2] == camera_segments(cfg)[:2]
    tour_poses = [cams[0]['cameras_interp'], cams[1]['cameras_interp'],
                  cams[3]['cameras_interp'][::-1], cams[2]['cameras_interp'][::-1]]
    for actual, expected in zip(tour_poses, camera_segments(cfg)):
        np.testing.assert_allclose(actual, expected['cameras_interp'], atol=1e-12)
    assert all('--using_scale' in s['command'] and s['command'][s['command'].index('--using_scale')+1] == 'False' for s in m['segments'])
    assert m['identity']['ray_policy'] == 'legacy_pixel_K_rescaled_again_UNVERIFIED_FOV'


def test_c_raw_routing_dependency_end_hash_resume_and_backend_separation(tmp_path):
    from bidirectional_relay import prepare, execute
    from relay_media import sha256
    cfg = make_config(tmp_path)
    root = tmp_path/'c'
    prepare(cfg, root, backend='test')
    calls = []
    def backend(s):
        calls.append(s['id'])
        if s['id'] in ('L1', 'R1'):
            assert s['input'] == str(Path(cfg.image).resolve())
        if s['id'] == 'L2':
            assert np.asarray(Image.open(s['input']))[0, 0, 0] == 30
        if s['id'] == 'R2':
            assert np.asarray(Image.open(s['input']))[0, 0, 0] == 130
            assert np.asarray(Image.open(s['end_image']))[0, 0, 0] == 80
            disk = json.loads((root/'manifest.json').read_text())
            assert disk['segments'][1]['status'] == 'TEST_ONLY'
            assert s['end_image_hash'] == sha256(s['end_image'])
        else:
            assert s['end_image'] is None
        synthetic(s)
    m = execute(root, backend)
    assert calls == ['L1', 'L2', 'R1', 'R2']
    assert [s['seed'] for s in m['segments']] == [1024, 1025, 1024, 1025]
    assert m['status'] == 'TEST_ONLY' and m['gpu_effects'] == 'NOT_RUN'
    assert m['segments'][3]['receipt']['hashes']['last_frame.png'] != m['segments'][1]['receipt']['hashes']['last_frame.png']
    execute(root, lambda s: pytest.fail('completed C rerun'))
    assert prepare(cfg, root, backend='test')['status'] == 'TEST_ONLY'
    with pytest.raises(ValueError, match='backend'):
        execute(root)
    Path(m['segments'][1]['last_frame']).write_bytes(b'broken')
    with pytest.raises(ValueError, match='artifact'):
        execute(root, lambda s: pytest.fail('corrupt teacher executed'))


def test_c_failure_before_teacher_completion_and_safe_retry(tmp_path):
    from bidirectional_relay import prepare, execute
    cfg = make_config(tmp_path)
    root = tmp_path/'c'
    prepare(cfg, root, backend='test')
    def fail(s):
        if s['id'] == 'L2':
            raise RuntimeError('teacher failed')
        synthetic(s)
    with pytest.raises(RuntimeError, match='teacher failed'):
        execute(root, fail)
    calls = []
    def finish(s):
        calls.append(s['id'])
        synthetic(s)
    execute(root, finish)
    assert calls == ['L2', 'R1', 'R2']


def test_c_partial_outputs_are_not_overwritten(tmp_path):
    from bidirectional_relay import prepare, execute
    cfg = make_config(tmp_path)
    root = tmp_path/'c'
    m = prepare(cfg, root, backend='test')
    target = Path(m['segments'][0]['directory'])/'last_frame.png'
    target.write_bytes(b'KEEP')
    with pytest.raises(ValueError, match='partial artifact'):
        execute(root, synthetic)
    assert target.read_bytes() == b'KEEP'


@pytest.mark.parametrize('change', ['prompt', 'input', 'camera', 'command', 'dependency', 'end_source', 'backend'])
def test_c_rejects_incompatible_reuse_or_plan_tampering(tmp_path, change):
    from bidirectional_relay import prepare, execute
    cfg = make_config(tmp_path)
    root = tmp_path/'c'
    m = prepare(cfg, root, backend='test')
    if change == 'prompt':
        cfg.prompt = 'changed'
        with pytest.raises(ValueError, match='incompatible'):
            prepare(cfg, root, backend='test')
        return
    if change == 'input':
        Image.new('RGB', (32, 32), 'red').save(cfg.image)
    elif change == 'camera':
        Path(m['segments'][0]['camera']).write_text('{}')
    else:
        if change == 'command':
            m['segments'][0]['command'].append('--evil')
        elif change == 'dependency':
            m['segments'][3]['depends_on'] = ['R1']
        elif change == 'end_source':
            m['segments'][3]['end_image'] = cfg.image
        else:
            m['backend'] = 'gpu'
        (root/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):
        execute(root, lambda s: pytest.fail('incompatible C executed'))


def test_c_real_reverse_pixels_and_assembly_counts(tmp_path):
    from bidirectional_relay import prepare, execute
    cfg = make_config(tmp_path)
    root = tmp_path/'c'
    prepare(cfg, root, backend='test')
    m = execute(root, synthetic)
    expected = {'CCW': [10,15,20,25,30,65,70,75,80],
                'CW': [110,115,120,125,130,165,170,175,180],
                'tour': [10,15,20,25,30,65,70,75,80,175,170,165,160,125,120,115,110]}
    for name, cmd in m['stitch_commands'].items():
        subprocess.run(cmd, check=True)
        reader = imageio.get_reader(str(root/'C'/f'{name}.mp4'))
        try:
            actual = [np.asarray(frame).mean() for frame in reader]
        finally:
            reader.close()
        np.testing.assert_allclose(actual, expected[name], atol=3)
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.run(cmd, check=True, capture_output=True)
    assert m['presentation']['cw_time_reversed'] is True
    assert m['presentation']['seam_policy'] == 'drop_later_first_time_sample_no_pixel_repair'


def test_c_cli_rejects_nonquarter_angle_and_plan_is_cpu_safe(tmp_path):
    cfg = make_config(tmp_path)
    cmd = [sys.executable, 'bidirectional_relay.py', 'plan', '--image', cfg.image,
           '--prompt', 'room', '--output', str(tmp_path/'plan')]
    rejected = subprocess.run(cmd+['--angle', '45'], capture_output=True, text=True)
    assert rejected.returncode != 0 and '90' in rejected.stderr
    assert not (tmp_path/'plan').exists()
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['status'] == 'NOT_RUN'
    m = json.loads((tmp_path/'plan/manifest.json').read_text())
    assert len(m['segments']) == 4
    assert m['identity']['config']['frames'] == 81
    assert sum('--end_image_path' in s['command'] for s in m['segments']) == 1


def test_c_cpu_smoke_cli(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'scripts.bidirectional_cpu_smoke', '--output', str(tmp_path/'smoke')],
                            capture_output=True, text=True, check=True)
    summary = json.loads(result.stdout)
    assert summary['backend_calls'] == ['L1', 'L2', 'R1', 'R2']
    assert summary['decoded_counts'] == {'CCW': 161, 'CW': 161, 'tour': 321}
    assert summary['torch_imported'] is False
    assert summary['gpu_effects'] == 'NOT_RUN' and summary['status'] == 'TEST_ONLY'


def test_c_meeting_intrinsics_through_real_loader_without_model_imports(tmp_path):
    from bidirectional_relay import camera_paths
    from test_wan22_contract import main_namespace
    cfg = make_config(tmp_path)
    cameras = camera_paths(cfg)
    loader = main_namespace()['cameras_json_to_camera_list']
    loaded = [loader(c, image_size=(cfg.height, cfg.width)) for c in cameras]
    np.testing.assert_allclose(loaded[1][-1].c2w_mat, loaded[3][-1].c2w_mat, atol=1e-12)
    intrinsics = [(c.fx, c.fy, c.cx, c.cy) for segment in loaded for c in segment]
    assert all(k == (cfg.focal, cfg.focal, 15.5, 15.5) for k in intrinsics)
    result = subprocess.run([sys.executable, '-c',
                             "import bidirectional_relay, sys; assert 'torch' not in sys.modules"], check=True)
    assert result.returncode == 0


@pytest.mark.parametrize('name', ['video.mp4', 'first_frame.png', 'last_frame.png', 'receipt.json'])
def test_c_corrupt_committed_outputs_never_regenerate(tmp_path, name):
    from bidirectional_relay import prepare, execute
    root = tmp_path/'c'
    prepare(make_config(tmp_path), root, backend='test')
    m = execute(root, synthetic)
    target = Path(m['segments'][3]['directory'])/name
    target.write_bytes(b'KEEP_BROKEN')
    with pytest.raises(ValueError, match='artifact'):
        execute(root, lambda s: pytest.fail('corrupted committed segment regenerated'))
    assert target.read_bytes() == b'KEEP_BROKEN'


def test_c_condition_hash_binding_is_checked_on_resume(tmp_path):
    from bidirectional_relay import prepare, execute
    root = tmp_path/'c'
    prepare(make_config(tmp_path), root, backend='test')
    m = execute(root, synthetic)
    m['segments'][3]['end_image_hash'] = 'stale'
    (root/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='end_image artifact changed'):
        execute(root, lambda s: pytest.fail('changed condition executed'))


def test_c_lock_and_nonempty_root_preserve_research(tmp_path):
    import fcntl
    from bidirectional_relay import prepare, execute
    cfg = make_config(tmp_path)
    root = tmp_path/'c'
    root.mkdir()
    note = root/'research.txt'
    note.write_text('KEEP')
    with pytest.raises(ValueError, match='nonempty'):
        prepare(cfg, root)
    assert note.read_text() == 'KEEP'
    root = tmp_path/'locked'
    prepare(cfg, root, backend='test')
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            prepare(cfg, root, backend='test')
        with pytest.raises(BlockingIOError):
            execute(root, synthetic)


def test_c_checkpoint_inventory_changes_reject_resume(tmp_path):
    from bidirectional_relay import prepare, execute
    cfg = make_config(tmp_path)
    checkpoint = tmp_path/'local-model.pth'
    checkpoint.write_bytes(b'FAKE LOCAL INVENTORY ONLY')
    cfg.model_ckpt_high = str(checkpoint)
    root = tmp_path/'c'
    prepare(cfg, root, backend='test')
    checkpoint.write_bytes(b'changed size')
    with pytest.raises(ValueError, match='incompatible'):
        execute(root, synthetic)


def test_reverse_spool_keeps_only_bounded_live_frames_and_cleans_up(tmp_path):
    import weakref
    from relay_media import ordered_frames, stitch
    refs = []
    def generated():
        for index in range(100):
            frame = np.full((32,32,3), index, np.uint8)
            refs.append(weakref.ref(frame))
            assert sum(ref() is not None for ref in refs) <= 3
            yield frame
    assert [int(frame[0,0,0]) for frame in ordered_frames(generated(), True, tmp_path)] == list(range(99,-1,-1))
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match='reverse indices'):
        stitch(['unused.mp4'], tmp_path/'out.mp4', 16, reverse=[1])
    assert not (tmp_path/'out.mp4').exists()
    with pytest.raises(ValueError, match='size/dtype'):
        list(ordered_frames(iter([np.zeros((32,32,3), np.uint8),
                                  np.zeros((16,16,3), np.uint8)]), True, tmp_path))
    assert list(tmp_path.iterdir()) == []
