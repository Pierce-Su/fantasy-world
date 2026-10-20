"""SYNTHETIC CPU control-flow/media smoke test; never produces model results."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio
import numpy as np
from orbit_relay import Config, prepare, execute
from relay_media import write_segment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New empty output root')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    cfg = Config(image=str(Path(__file__).resolve().parents[1]/'examples/images/input_image.png'),
                 prompt='SYNTHETIC CPU CONTROL-FLOW TEST ONLY', width=32, height=32)
    prepare(cfg, root, backend='test')
    calls = []
    def synthetic_backend(segment):
        calls.append(segment['id'])
        frames = np.stack([np.full((32,32,3), (segment['index']*40+j)%256, np.uint8) for j in range(81)])
        write_segment(frames, segment['directory'], fps=16)
    manifest = execute(root, synthetic_backend)
    counts = {}
    for experiment, command in manifest['stitch_commands'].items():
        subprocess.run(command, check=True)
        reader = imageio.get_reader(str(root/experiment/'joined.mp4'))
        try:
            counts[experiment] = sum(1 for _ in reader)
        finally:
            reader.close()
    if calls != ['A:0','A:1','A:2','A:3','B:3'] or counts != {'A':321, 'B':321}:
        raise RuntimeError('unexpected control-flow/frame count')
    if 'torch' in sys.modules:
        raise RuntimeError('CPU smoke unexpectedly imported torch')
    print(json.dumps({'status': manifest['status'], 'gpu_effects': manifest['gpu_effects'],
                      'backend_calls': calls, 'joined_decoded_counts': counts,
                      'torch_imported': False, 'manifest': str(root/'manifest.json')}))


if __name__ == '__main__':
    main()
