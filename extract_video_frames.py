#!/usr/bin/env python3
# Copyright Alibaba Inc. All Rights Reserved.
"""Extract frames and periodic keyframes from a generated video.

Writes:
  {output_dir}/frames/frame_{:05d}.png
  {output_dir}/keyframes/keyframe_{:05d}.png   (every N frames, 0-based)

Usage:
  python extract_video_frames.py path/to/video.mp4
  python extract_video_frames.py path/to/output_dir/video.mp4 --keyframe_interval 8
  python extract_video_frames.py path/to/video.mp4 --output_dir path/to/output_dir
"""

import argparse
from pathlib import Path

from utils import extract_video_frames_and_keyframes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames and keyframes from a video file.")
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video (e.g. output_dir/video.mp4).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for frames/ and keyframes/ subfolders "
             "(default: parent directory of video_path).",
    )
    parser.add_argument(
        "--keyframe_interval",
        type=int,
        default=16,
        help="Save every N-th frame (0-based) to keyframes/ (default: 16).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    result = extract_video_frames_and_keyframes(
        video_path=video_path,
        output_dir=args.output_dir,
        keyframe_interval=args.keyframe_interval,
    )
    print(f"Done: {result['total_frames']} frames, {len(result['keyframe_paths'])} keyframes")


if __name__ == "__main__":
    main()
