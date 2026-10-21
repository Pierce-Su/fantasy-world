"""Prepare a bounded clip without starting a relay or loading model weights."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orbit_relay import Config, camera_segments, command, RAY_POLICY, identity

def prepare(config, root, index=0):
    root = Path(root).resolve()
    cfg = Config(**config)
    cameras = camera_segments(cfg)
    if not 0 <= index < len(cameras):
        raise ValueError('segment index outside planned orbit')
    if root.exists():
        raise FileExistsError('new output root required; never overwrite evidence')
    camera = cameras[index]
    segment = {'input': cfg.image, 'end_image': None, 'seed': cfg.seed+index,
               'camera': str(root/'camera.json'), 'directory': str(root/'output')}
    plan = {'status': 'PREPARED_NOT_GENERATED', 'segment_index': index,
            'nominal_start_degrees': cfg.angle*index,
            'nominal_end_degrees': cfg.angle*(index+1),
            'actual_video_pose': 'UNVERIFIED', 'ray_policy': RAY_POLICY,
            'identity': identity(cfg), 'config': asdict(cfg),
            'command': command(cfg, segment),
            'seed_policy': 'base_seed_plus_segment_index_not_same_seed_ablation',
            'image_policy': 'index1 must receive true raw endpoint of index0; never decoded MP4'}
    root.mkdir(parents=True)
    for name,data in [('camera.json',camera),('plan.json',plan)]:
        (root/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    return plan

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--segment-index',type=int,choices=[0,1],default=0)
    a=p.parse_args()
    result=prepare(json.loads(a.config.read_text()),a.output_root,a.segment_index)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
