"""Lossless relay endpoints plus independently decodable video artifacts (no GPU imports)."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_segment(frames, directory, fps):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) < 2:
        raise ValueError('expected at least two uint8 RGB frames')
    Image.fromarray(frames[0]).save(directory/'first_frame.png')
    Image.fromarray(frames[-1]).save(directory/'last_frame.png')
    imageio.mimwrite(str(directory/'video.mp4'), frames, fps=fps, macro_block_size=1, quality=8)
    receipt = {'frames': len(frames), 'height': frames.shape[1], 'width': frames.shape[2], 'fps': fps,
               'hashes': {name: sha256(directory/name) for name in ('video.mp4', 'first_frame.png', 'last_frame.png')}}
    (directory/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def validate_segment(directory, frames, height, width, fps):
    directory = Path(directory)
    try:
        receipt = json.loads((directory/'receipt.json').read_text())
        if any(receipt[k] != v for k, v in {'frames': frames, 'height': height, 'width': width, 'fps': fps}.items()):
            raise ValueError('artifact dimensions/count/fps mismatch')
        for name in ('video.mp4', 'first_frame.png', 'last_frame.png'):
            if sha256(directory/name) != receipt['hashes'][name]:
                raise ValueError('artifact hash mismatch')
        for name in ('first_frame.png', 'last_frame.png'):
            with Image.open(directory/name) as image:
                image.load()
                if image.size != (width, height) or image.mode != 'RGB':
                    raise ValueError('artifact endpoint format mismatch')
        reader = imageio.get_reader(str(directory/'video.mp4'))
        try:
            count = 0
            for frame in reader:
                if frame.shape != (height, width, 3):
                    raise ValueError('artifact video shape mismatch')
                count += 1
            if count != frames:
                raise ValueError('artifact video frame count mismatch')
            if abs(reader.get_meta_data()['fps'] - fps) > .01:
                raise ValueError('artifact video fps mismatch')
        finally:
            reader.close()
        return receipt
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        raise ValueError(f'invalid segment artifact in {directory}: {exc}') from exc


def ordered_frames(reader, reverse, scratch):
    """Reverse with a disk spool: O(one frame) RAM, O(clip pixels) temporary disk.

    No ffmpeg reverse filter (which buffers the full clip in memory). Decode once
    and seek fixed-size RGB records backwards. TemporaryFile cleans up on errors.
    """
    if not reverse:
        yield from reader
        return
    with tempfile.TemporaryFile(dir=scratch) as spool:
        shape, size, count = None, 0, 0
        for frame in reader:
            if shape is None:
                shape, size = frame.shape, frame.nbytes
            if frame.shape != shape or frame.dtype != np.uint8:
                raise ValueError('segment size/dtype mismatch')
            spool.write(frame.tobytes())
            count += 1
        for index in range(count-1, -1, -1):
            spool.seek(index*size)
            data = spool.read(size)
            if len(data) != size:
                raise ValueError('truncated reverse spool')
            yield np.frombuffer(data, dtype=np.uint8).reshape(shape)


def stitch(videos, output, fps, reverse=()):
    """Keep all frames of first clip; discard first frame of every later clip.

    Equal camera pose does NOT imply equal generated RGB. This convention removes
    one overlapping time sample, not a claim of pixel-identical boundaries.
    """
    if not videos:
        raise ValueError('no segments')
    reverse = set(reverse)
    if any(index < 0 or index >= len(videos) for index in reverse):
        raise ValueError('reverse indices must identify zero-based input clips')
    output = Path(output)
    if output.exists() or output.resolve() in [Path(v).resolve() for v in videos]:
        raise ValueError('refusing to overwrite output or segment')
    with imageio.get_writer(str(output), fps=fps, macro_block_size=1, quality=8) as writer:
        shape = None
        for index, video in enumerate(videos):
            reader = imageio.get_reader(str(video))
            try:
                if abs(reader.get_meta_data()['fps']-fps) > .01:
                    raise ValueError('segment fps mismatch')
                count = 0
                for frame_number, frame in enumerate(ordered_frames(reader, index in reverse, output.parent)):
                    count += 1
                    if shape is None:
                        shape = frame.shape
                    if frame.shape != shape:
                        raise ValueError('segment size mismatch')
                    if index == 0 or frame_number > 0:
                        writer.append_data(frame)
                if count < 2:
                    raise ValueError('segment needs at least two frames')
            finally:
                reader.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Join relay segments, dropping later first frames')
    parser.add_argument('--fps', type=int, default=16)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reverse', type=int, action='append', default=[],
                        help='Zero-based input clip to time-reverse; repeat for multiple clips (disk spooled)')
    parser.add_argument('videos', nargs='+')
    args = parser.parse_args()
    stitch(args.videos, args.output, args.fps, args.reverse)
