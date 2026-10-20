import json
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest
import orbit_relay as orbit
import shared_prefix_relay as shared
from relay_media import write_segment, validate_segment


def fixture(tmp_path):
    baseline = tmp_path/'baseline'
    baseline.mkdir()
    image = baseline/'I0.png'
    Image.new('RGB',(32,32),(20,30,40)).save(image)
    cfg = orbit.Config(image=str(image),prompt='fixture',frames=5,width=32,height=32)
    segment = shared.blueprint(cfg,tmp_path/'new',baseline)[0]['segments'][0]
    orbit.atomic_json(baseline/'camera.json',orbit.camera_segments(cfg)[0])
    orbit.atomic_json(baseline/'argv.json',segment['command'])
    orbit.atomic_json(baseline/'provenance.json',{'identity':orbit.identity(cfg),'backend':'test'})
    orbit.atomic_json(baseline/'runtime.json',{'exit_code':0})
    orbit.atomic_json(baseline/'started.json',{'pid':1,'jobid':'TEST_ONLY'})
    (baseline/'inference.exit').write_text('0\n')
    (baseline/'inference.log').write_text('TEST_ONLY real media fixture; no inference\n')
    write_segment(np.full((5,32,32,3),40,dtype=np.uint8),baseline/'output',16)
    return cfg,baseline,tmp_path/'new'


def backend(calls):
    def run(s):
        calls.append(s['id'])
        frames=np.asarray([np.full((32,32,3),60+len(calls)*20+i*2,np.uint8) for i in range(5)])
        write_segment(frames,s['directory'],16)
    return run


def test_plan_dag_poses_and_prefix(tmp_path):
    cfg,base,root=fixture(tmp_path)
    m=shared.prepare(cfg,root,base,'test')
    assert m['status']=='NOT_RUN'
    assert len({s['id'] for s in m['segments']})==7
    assert m['orders']['C']==['S0','S1','R2','R1']
    by={s['id']:s for s in m['segments']}
    assert by['R2']['depends_on']==['R1','S1']
    assert [s['seed'] for s in m['segments']]==[1024,1025,1026,1027,1027,1024,1025]
    assert by['A3']['input']==by['B3']['input']
    assert by['B3']['end_image']==cfg.image
    assert by['R2']['end_image']==by['S1']['last_frame']
    cams=shared.blueprint(cfg,root,base)[1]
    np.testing.assert_allclose(cams['S1']['cameras_interp'][-1],cams['R2']['cameras_interp'][-1],atol=1e-12)
    for name,cam in cams.items():
        np.testing.assert_allclose(np.asarray(cam['anchor_c2w'])@np.asarray(cam['local_c2w']),cam['cameras_interp'],atol=1e-12)
    assert by['A3']['camera_sha256']==by['B3']['camera_sha256']


def test_real_media_import_execute_resume_stitch(tmp_path):
    cfg,base,root=fixture(tmp_path)
    shared.prepare(cfg,root,base,'test')
    calls=[]
    m=shared.execute(root,backend(calls))
    assert calls==['S1','S2','A3','B3','R1','R2']
    assert m['status']=='TEST_ONLY' and m['gpu_effects']=='NOT_RUN'
    assert m['segments'][0]['directory']==str(base/'output')
    calls.clear(); shared.prepare(cfg,root,base,'test'); shared.execute(root,backend(calls))
    assert calls==[]
    shared.assemble(root)
    import imageio.v2 as imageio
    for name,count in [('A',17),('B',17),('C',17),('C_CCW',9),('C_CW',9)]:
        reader=imageio.get_reader(str(root/(name+'.mp4')))
        try:
            frames=[f for f in reader]
            assert len(frames)==count
            if name=='C':
                # R2 backwards drops generated last sample, then 3,2,1,0;
                # R1 backwards likewise. Real lossy codec tolerance only.
                means=[float(f.mean()) for f in frames]
                assert means[9]>means[12] and means[13]>means[16]
        finally:reader.close()
    with pytest.raises(ValueError):shared.execute(root)


@pytest.mark.parametrize('field',['camera','argv','exit','runtime','source','input','test_only'])
def test_import_rejects_mismatch(tmp_path,field):
    cfg,base,root=fixture(tmp_path)
    if field=='camera':
        data=json.loads((base/'camera.json').read_text()); data['focal_length']=10;orbit.atomic_json(base/'camera.json',data)
    elif field=='argv':
        data=json.loads((base/'argv.json').read_text());data[-1]='True';orbit.atomic_json(base/'argv.json',data)
    elif field=='exit':(base/'inference.exit').write_text('1')
    elif field=='runtime':orbit.atomic_json(base/'runtime.json',{'exit_code':1})
    elif field=='source':
        p=json.loads((base/'provenance.json').read_text());p['identity']['source_sha256']['utils.py']='x';orbit.atomic_json(base/'provenance.json',p)
    elif field=='input':Image.new('RGB',(32,32)).save(cfg.image)
    with pytest.raises(ValueError):shared.prepare(cfg,root,base,'gpu' if field=='test_only' else 'test')


def test_resume_tamper_and_no_replay(tmp_path):
    cfg,base,root=fixture(tmp_path)
    m=shared.prepare(cfg,root,base,'test')
    m['segments'][1]['depends_on']=[]
    orbit.atomic_json(root/'manifest.json',m)
    with pytest.raises(ValueError):shared.execute(root,backend([]))
    root2=tmp_path/'new2';shared.prepare(cfg,root2,base,'test')
    def crash(s):
        (Path(s['directory'])/'started.json').write_text('{}')
        raise RuntimeError('crash fixture')
    with pytest.raises(RuntimeError):shared.execute(root2,crash)
    with pytest.raises(ValueError,match='partial/started'):shared.execute(root2,backend([]))


def test_changed_end_condition_refused(tmp_path):
    cfg,base,root=fixture(tmp_path)
    shared.prepare(cfg,root,base,'test');shared.execute(root,backend([]))
    target=root/'segments/S1/last_frame.png'
    Image.new('RGB',(32,32)).save(target)
    with pytest.raises(ValueError):shared.execute(root,backend([]))
