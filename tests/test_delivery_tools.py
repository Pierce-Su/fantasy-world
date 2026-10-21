import importlib.util
import json
import os
from pathlib import Path
import sys
import pytest
from PIL import Image
import numpy as np
from dataclasses import asdict
from orbit_relay import Config,camera_segments,RAY_POLICY

def module(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

plan=module('plan_single_clip');archive=module('evidence_archive');runner=module('run_single_clip')

def config(tmp_path):
    image=tmp_path/'raw.png';Image.new('RGB',(32,32),(60,90,120)).save(image)
    return asdict(Config(image=str(image),prompt='TEST_ONLY',angle=60,frames=5,width=32,height=32))

def test_single60_and_second60_pose_seed_argv(tmp_path):
    c=config(tmp_path);a=plan.prepare(c,tmp_path/'first',0);b=plan.prepare(c,tmp_path/'second',1)
    first=json.loads((tmp_path/'first/camera.json').read_text());second=json.loads((tmp_path/'second/camera.json').read_text())
    np.testing.assert_array_equal(first['cameras_interp'][-1],second['cameras_interp'][0])
    for cam in [first,second]:np.testing.assert_allclose(np.array(cam['anchor_c2w'])@np.array(cam['local_c2w']),cam['cameras_interp'],atol=1e-12)
    assert a['status']=='PREPARED_NOT_GENERATED' and b['ray_policy']==RAY_POLICY
    assert b['command'][b['command'].index('--seed')+1]=='1025'
    assert '--end_image_path' not in b['command']
    assert not (tmp_path/'second/output').exists()
    with pytest.raises(FileExistsError):plan.prepare(c,tmp_path/'first')
    with pytest.raises(ValueError):plan.prepare(c,tmp_path/'invalid',99)

def test_runner_rejects_tamper_before_gpu_query(tmp_path):
    c=config(tmp_path);root=tmp_path/'clip';p=plan.prepare(c,root)
    p['command'].append('--malicious');(root/'plan.json').write_text(json.dumps(p))
    with pytest.raises(ValueError,match='command changed'):runner.gate(root)
    p['command'].pop();(root/'plan.json').write_text(json.dumps(p));(root/'started.json').write_text('{}')
    with pytest.raises(ValueError,match='never replay'):runner.gate(root)

def test_real_archive_extract_hardlink_hash_and_corruption(tmp_path):
    root=tmp_path/'input';root.mkdir();(root/'a.txt').write_text('real bytes\n');os.link(root/'a.txt',root/'b.txt')
    m=archive.create(root,['a.txt','b.txt'],tmp_path/'result.tar.gz')
    v=archive.verify(tmp_path/'result.tar.gz',m,tmp_path/'extracted')
    assert v['PASS'] and v['members_extracted_and_hashed']==2
    with pytest.raises(FileExistsError):archive.verify(tmp_path/'result.tar.gz',m,tmp_path/'extracted')
    m['files'][0]['sha256']='0'*64
    with pytest.raises(ValueError,match='extracted member checksum'):archive.verify(tmp_path/'result.tar.gz',m,tmp_path/'bad')
    with pytest.raises(ValueError):archive.inventory(root,['../result.tar.gz'])
    (root/'outside').symlink_to(tmp_path/'result.tar.gz')
    with pytest.raises(ValueError):archive.inventory(root,['outside'])


def test_comparison_rejects_wrong_shape_real_media(tmp_path):
    import subprocess
    from relay_media import write_segment
    source=tmp_path/'source'
    write_segment(np.full((5,32,32,3),80,dtype=np.uint8),source,16)
    script=Path(__file__).resolve().parents[1]/'scripts/compare_orbits.py'
    result=subprocess.run([sys.executable,str(script),'--clip60',str(source/'video.mp4'),'--clip90',str(source/'video.mp4'),'--output-root',str(tmp_path/'comparison')],capture_output=True,text=True)
    assert result.returncode!=0 and 'AssertionError' in result.stderr
    assert not (tmp_path/'comparison/orbit-060-vs-090_side-by-side.mp4').exists()
