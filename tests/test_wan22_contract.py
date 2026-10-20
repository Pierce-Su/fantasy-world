"""Exercise inference entry-point contracts without importing GPU packages.

Only the expensive sampler/model is replaced; parser, camera loader and endpoint
writer execute their real source. This does not validate GPU imports/weights.
"""
import argparse
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace):
    tree = ast.parse((ROOT/path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), namespace)


def test_nondefault_frames_reach_actual_pipe_call():
    tree = ast.parse((ROOT/'inference_wan22.py').read_text())
    sampler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FantasyWorldSampler')
    generate = next(n for n in sampler.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_video')
    call = next(n for n in ast.walk(generate) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'self.model_high.pipe')
    captured = {}
    def pipe(**kwargs):
        captured.update(kwargs)
    namespace = {'self': SimpleNamespace(num_frames=5, base_seed=42, height=32, width=32,
                                        model_high=SimpleNamespace(pipe=pipe)),
                 'prompt': 'room', 'neg_prompt':'', 'input_image': object(), 'end_image': None}
    eval(compile(ast.Expression(call), 'pipe_call', 'eval'), namespace)
    assert captured.get('num_frames') == 5
    assert captured['end_image'] is None


def main_namespace():
    ns = {'argparse': argparse, 'os': os, 'Path': Path, 'json': json, 'np': np}
    definitions('utils.py', {'Camera', '_infer_intrinsics', 'cameras_json_to_camera_list'}, ns)
    definitions('inference_wan22.py', {'str2bool', 'main'}, ns)
    return ns


@pytest.mark.parametrize('frames,camera_count', [(6,6), (5,4), (1,1)])
def test_invalid_frames_rejected_before_expensive_sampler(tmp_path, monkeypatch, frames, camera_count):
    from PIL import Image
    image = tmp_path/'start.png'
    Image.new('RGB', (32,32)).save(image)
    camera = tmp_path/'camera.json'
    camera.write_text(json.dumps({'cameras_interp':[np.eye(4).tolist()]*camera_count}))
    monkeypatch.setattr('sys.argv', ['inference_wan22.py', '--image_path',str(image), '--prompt','room',
                                   '--camera_json_path',str(camera), '--frames',str(frames),
                                   '--wan_ckpt_path','unused', '--model_ckpt_high','unused',
                                   '--model_ckpt_low','unused', '--output_dir',str(tmp_path/'out')])
    ns = main_namespace()
    ns['FantasyWorldSampler'] = lambda **kw: pytest.fail('expensive sampler initialized for invalid input')
    with pytest.raises(ValueError):
        ns['main']()


def test_real_camera_loader_inverts_global_schema_with_pixel_intrinsics():
    from orbit_relay import Config, camera_segments
    ns = main_namespace()
    segment = camera_segments(Config(image='unused', prompt='room', frames=5))[1]
    cameras = ns['cameras_json_to_camera_list'](segment, image_size=(480,832))
    np.testing.assert_allclose(cameras[0].c2w_mat, segment['anchor_c2w'], atol=1e-12)
    np.testing.assert_allclose(cameras[0].w2c_mat @ np.array(segment['anchor_c2w']), np.eye(4), atol=1e-12)
    assert (cameras[0].fx, cameras[0].fy, cameras[0].cx, cameras[0].cy) == (500,500,415.5,239.5)


def test_inference_exports_raw_endpoint_before_lossy_video_handoff(tmp_path, monkeypatch):
    from PIL import Image
    from relay_media import write_segment
    import imageio.v2 as imageio
    image = tmp_path/'start.png'
    Image.new('RGB', (32,32)).save(image)
    camera = tmp_path/'camera.json'
    camera.write_text(json.dumps({'cameras_interp':[np.eye(4).tolist()]*5}))
    out = tmp_path/'out'
    monkeypatch.setattr('sys.argv', ['inference_wan22.py', '--image_path',str(image), '--prompt','room',
                                   '--camera_json_path',str(camera), '--frames','5', '--height','32', '--width','32',
                                   '--wan_ckpt_path','unused', '--model_ckpt_high','unused',
                                   '--model_ckpt_low','unused', '--output_dir',str(out), '--extract_frames','False'])
    video = np.arange(5*32*32*3, dtype=np.uint8).reshape(5,32,32,3)
    ns = main_namespace()
    ns['FantasyWorldSampler'] = lambda **kw: SimpleNamespace(generate_video=lambda **kwargs: (video, None))
    ns['torch'] = SimpleNamespace(save=lambda *args: None)
    ns['write_segment'] = write_segment
    ns['save_video_imageio'] = lambda v,p,fps: imageio.mimwrite(str(p), v, fps=fps)
    class StopBeforeUnrelatedLifting(Exception):
        pass
    def stop(*args, **kwargs):
        raise StopBeforeUnrelatedLifting
    ns['get_pointclouds'] = stop
    with pytest.raises(StopBeforeUnrelatedLifting):
        ns['main']()
    assert (out/'receipt.json').is_file()
    np.testing.assert_array_equal(np.array(Image.open(out/'last_frame.png')), video[-1])
