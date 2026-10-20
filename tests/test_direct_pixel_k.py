"""Real CPU tensor contracts, without constructing a diffusion model.

Dataset/adapter source modules are loaded directly to avoid package-level model
imports; pose encoder/decoder, ray_condition and adapter forward are unmodified.
The sampler's actual packing statements execute via AST, not a copied formula.
"""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from FantasyWorld.vggt.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri

ROOT = Path(__file__).resolve().parents[1]


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pack(rays):
    tree = ast.parse((ROOT/'inference_wan22.py').read_text())
    sampler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FantasyWorldSampler')
    method = next(n for n in sampler.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_video_with_dual_models')
    block = next(n for n in method.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'self.model_high.camera_control')
    ns = {'torch': torch, 'plucker_embedding': rays,
          'self': SimpleNamespace(model_high=SimpleNamespace(camera_control=True, use_info='plucker'), device='cpu', torch_dtype=torch.float32)}
    exec(compile(ast.Module(body=[block], type_ignores=[]), 'actual_sampler_camera_pack', 'exec'), ns)
    return ns['control_camera_latents_input']


@pytest.mark.parametrize('height,width,frames,segment', [(32,48,5,1), (480,832,81,0)])
def test_pixel_roundtrip_rays_file_parity_and_adapter(tmp_path, height, width, frames, segment):
    from orbit_relay import Config, camera_segments
    ds = source_module('direct_k_dataset', 'FantasyWorld/diffsynth_wan22/data/dataset_re10k.py')
    adapter_module = source_module('direct_k_adapter', 'FantasyWorld/diffsynth_wan22/models/wan_video_camera_controller.py')
    c2w = torch.tensor(camera_segments(Config(image='unused', prompt='room', frames=frames, height=height, width=width))[segment]['cameras_interp'], dtype=torch.float32)
    w2c = torch.linalg.inv(c2w)[:, :3]
    k = torch.tensor([[500.,0,(width-1)/2],[0,500.,(height-1)/2],[0,0,1]]).expand(frames,-1,-1).clone()
    encoded = extri_intri_to_pose_encoding(w2c[None], k[None], (height,width))
    decoded_ext, decoded_k = pose_encoding_to_extri_intri(encoded, (height,width))
    torch.testing.assert_close(decoded_ext[0], w2c, atol=2e-6, rtol=2e-6)
    expected_k = torch.tensor([500.,500.,width/2,height/2]).expand(1,frames,4)
    torch.testing.assert_close(decoded_k[0,:,0,2], torch.full((frames,),width/2))
    proc = ds.RealEstate10KPoseProcessor(sample_stride=1, sample_n_frames=frames, sample_size=(height,width), relative_pose=True, zero_t_first_frame=True, rescale_fxy=False)
    captured = {}
    real_ray = ds.ray_condition
    def observe(K, poses, *args, **kwargs):
        captured['K'], captured['poses'] = K.clone(), poses.clone()
        return real_ray(K, poses, *args, **kwargs)
    with patch.object(ds, 'ray_condition', observe):
        rays = proc.get_plucker_embedding_direct_from_cam_params(encoded, (height,width))
    print('ACTUAL_RAY_K', captured['K'][0,0].tolist(), 'RAY_SHAPE', tuple(rays.shape))
    torch.testing.assert_close(captured['K'], expected_k, atol=1e-4, rtol=1e-6)
    expected_pose = torch.linalg.inv(c2w[0]) @ c2w
    torch.testing.assert_close(captured['poses'][0], expected_pose, atol=2e-6, rtol=2e-6)
    directions = rays[...,3:]
    torch.testing.assert_close(directions.norm(dim=-1), torch.ones_like(directions[...,0]), atol=2e-6, rtol=2e-6)
    origins = expected_pose[None,:,None,None,:3,3].expand_as(directions)
    torch.testing.assert_close(rays[...,:3], torch.linalg.cross(origins,directions), atol=2e-6, rtol=2e-6)
    # Independent analytic pinhole ray, preserving decoder W/2,H/2 and pixel +.5.
    d = torch.tensor([.5/500,.5/500,1.]); d /= d.norm()
    torch.testing.assert_close(directions[0,0,height//2,width//2], d, atol=2e-6, rtol=2e-6)
    # File path has normalized K and must retain its one W/H conversion.
    pose_file = tmp_path/'normalized.txt'
    pose_file.write_text('\n'.join(' '.join(map(str,[i,500/width,500/height,.5,.5,0,0]+ext.flatten().tolist())) for i,ext in enumerate(decoded_ext[0].numpy())))
    file_rays = proc.get_plucker_embedding(pose_file)
    torch.testing.assert_close(file_rays, rays, atol=2e-6, rtol=2e-6)
    del file_rays
    packed = pack(rays)
    assert packed.shape == (1,24,(frames+3)//4,height,width)
    for group in range((frames+3)//4):
        indices = [0]*4 if group == 0 else list(range(group*4-3,group*4+1))
        expected = rays[0,indices].permute(3,0,1,2).reshape(24,height,width)
        torch.testing.assert_close(packed[0,:,group], expected)
    # Actual adapter forward on small CPU tensors, full production shape on meta.
    if frames == 5:
        adapter = adapter_module.SimpleAdapter(24,8,2,2)
        with torch.no_grad(): result = adapter(packed)
    else:
        with torch.device('meta'):
            adapter = adapter_module.SimpleAdapter(24,5120,2,2)
            result = adapter(torch.empty(packed.shape,device='meta'))
    assert result.shape == (1,8 if frames==5 else 5120,(frames+3)//4,height//16,width//16)


def test_policy_fingerprint_distinguishes_legacy(tmp_path, monkeypatch):
    import orbit_relay as relay
    from PIL import Image
    RAY_POLICY, digest = relay.RAY_POLICY, relay.digest
    assert RAY_POLICY == 'direct_pixel_K_v1_decoder_center_UNVERIFIED_VIDEO_FOV'
    assert digest({'ray_policy':RAY_POLICY}) != digest({'ray_policy':'legacy_pixel_K_rescaled_again_UNVERIFIED_FOV'})
    image = tmp_path/'input.png'
    Image.new('RGB',(32,32)).save(image)
    cfg = relay.Config(image=str(image),prompt='room',frames=5,width=32,height=32)
    root = tmp_path/'legacy'
    with monkeypatch.context() as m:
        m.setattr(relay,'RAY_POLICY','legacy_pixel_K_rescaled_again_UNVERIFIED_FOV')
        relay.prepare(cfg,root,backend='synthetic')
    with pytest.raises(ValueError,match='incompatible resume'):
        relay.prepare(cfg,root,backend='synthetic')
    with pytest.raises(ValueError,match='incompatible resume'):
        relay.execute(root,backend=lambda *args: pytest.fail('must not run cross-policy'))
