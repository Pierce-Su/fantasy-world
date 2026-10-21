"""Run exactly one prepared clip, with exclusive started marker; no retries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from orbit_relay import Config,identity,camera_segments,command

def gate(root):
    root=Path(root).resolve();p=json.loads((root/'plan.json').read_text());cfg=Config(**p['config'])
    if identity(cfg)!=p['identity']:raise ValueError('source/input/checkpoint/commit changed; prepare a new root')
    index=p['segment_index']
    if json.loads((root/'camera.json').read_text())!=camera_segments(cfg)[index]:raise ValueError('camera changed')
    segment={'input':cfg.image,'end_image':None,'seed':cfg.seed+index,'camera':str(root/'camera.json'),'directory':str(root/'output')}
    if p['command']!=command(cfg,segment):raise ValueError('command changed')
    if (root/'output').exists() or (root/'started.json').exists():raise ValueError('partial/started output; never replay')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('GPU busy')
    if not os.environ.get('HF_HOME'):raise ValueError('HF_HOME must be explicit')
    return p

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:parser.error('--execute is required; planning never authorizes a model run')
    root=args.root.resolve();plan=gate(root)
    start=time.time()
    with (root/'started.json').open('x') as f:json.dump({'pid':os.getpid(),'time':start,'authorized_inferences':1},f)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
    with (root/'inference.log').open('xb') as log:
        proc=subprocess.Popen(plan['command'],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=Path(__file__).resolve().parents[1])
        (root/'inference.pid').write_text(str(proc.pid))
        samples=[]
        while proc.poll() is None:
            result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,capture_output=True)
            samples.append({'elapsed_s':time.time()-start,'nvml_csv':result.stdout,'exit_code':result.returncode})
            time.sleep(1)
        rc=proc.wait()
    (root/'vram-samples.json').write_text(json.dumps(samples,indent=2))
    (root/'runtime.json').write_text(json.dumps({'exit_code':rc,'elapsed_s':time.time()-start,'sampling':'~1s NVML, not allocator peak; host/container PID not independently mapped'},indent=2))
    (root/'inference.exit').write_text(str(rc)+'\n')
    print('DONE' if rc==0 else 'FAILED',rc);sys.exit(rc)
if __name__=='__main__':main()
