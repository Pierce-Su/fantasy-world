"""SYNTHETIC C CPU smoke; no model, GPU, network, or scene quality evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio
import numpy as np
from bidirectional_relay import Config, prepare, execute
from relay_media import write_segment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New empty output root')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    cfg = Config(image=str(Path(__file__).resolve().parents[1]/'examples/images/input_image.png'),
                 prompt='SYNTHETIC C CPU TEST ONLY', width=32, height=32)
    prepare(cfg, root, backend='test')
    calls = []
    def synthetic_backend(segment):
        calls.append(segment['id'])
        # Unique monotonic per-clip intensity; neither endpoint is patched to a condition.
        frames = np.stack([np.full((32,32,3), segment['index']*40+j, np.uint8) for j in range(cfg.frames)])
        write_segment(frames, segment['directory'], cfg.fps)
    manifest = execute(root, synthetic_backend)
    execute(root, lambda s: (_ for _ in ()).throw(RuntimeError('resume unexpectedly called backend')))
    counts = {}
    for name, cmd in manifest['stitch_commands'].items():
        subprocess.run(cmd, check=True)
        reader = imageio.get_reader(str(root/'C'/f'{name}.mp4'))
        try:
            counts[name] = sum(1 for _ in reader)
        finally:
            reader.close()
    if calls != ['L1', 'L2', 'R1', 'R2'] or counts != {'CCW': 161, 'CW': 161, 'tour': 321}:
        raise RuntimeError('unexpected control-flow/frame count')
    if 'torch' in sys.modules:
        raise RuntimeError('CPU smoke unexpectedly imported torch')
    summary = {'status': manifest['status'], 'gpu_effects': manifest['gpu_effects'],
               'backend_calls': calls, 'decoded_counts': counts, 'torch_imported': False,
               'resume_backend_calls': 0, 'manifest': str(root/'manifest.json')}
    (root/'smoke_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
