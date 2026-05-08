"""Generate 360-degree camera trajectories.

Two modes:
- "orbit": camera position traces a circle around a target point in front of
  it; orientation continuously yaws so the target stays centered.
- "spin":  camera stays at the origin and only yaws (rotates around the world
  Y axis) through a full 360-degree pan.

Conventions (matching utils.py / OpenCV):
- Each 4x4 matrix is camera-to-world (c2w).
- Camera local axes: +X right, +Y down, +Z forward (look direction).
- Columns of the rotation block are the camera axes in world coordinates:
  R = [X_cam | Y_cam | Z_cam], translation t is camera position.

Usage:
    python _make_360.py             # produces camera_data_360.json (orbit)
    python _make_360.py spin        # produces camera_data_360_spin.json (spin)
    python _make_360.py both        # produces both
"""

import json
import math
import sys
from pathlib import Path


def orbit_c2w(theta: float, radius: float = 1.0) -> list[list[float]]:
    """c2w for a horizontal orbit around T = (0, 0, radius); theta=0 -> identity."""
    c, s = math.cos(theta), math.sin(theta)
    return [
        [ c,         0.0, -s,        radius * s         ],
        [ 0.0,       1.0,  0.0,      0.0                ],
        [ s,         0.0,  c,        radius * (1.0 - c) ],
        [ 0.0,       0.0,  0.0,      1.0                ],
    ]


def spin_c2w(theta: float) -> list[list[float]]:
    """c2w for a pure yaw around world Y at the origin; theta=0 -> identity.

    Camera stays at (0,0,0); only its orientation rotates. The camera's forward
    axis sweeps from +Z through +X, -Z, -X and back to +Z over theta in [0, 2*pi].
    """
    c, s = math.cos(theta), math.sin(theta)
    return [
        [ c,   0.0,  s,   0.0 ],
        [ 0.0, 1.0,  0.0, 0.0 ],
        [-s,   0.0,  c,   0.0 ],
        [ 0.0, 0.0,  0.0, 1.0 ],
    ]


def quantize(mat: list[list[float]]) -> list[list[float]]:
    """Match the float32-ish look of the original file (clean zeros/ones, ~7 sig figs)."""
    out = []
    for row in mat:
        new_row = []
        for v in row:
            if abs(v) < 1e-7:
                new_row.append(0.0)
            elif abs(v - 1.0) < 1e-7:
                new_row.append(1.0)
            elif abs(v + 1.0) < 1e-7:
                new_row.append(-1.0)
            else:
                new_row.append(float(f"{v:.8g}"))
        out.append(new_row)
    return out


def build(mode: str, num_frames: int = 81) -> dict:
    if mode == "orbit":
        pose_fn = orbit_c2w
    elif mode == "spin":
        pose_fn = spin_c2w
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    interp = [quantize(pose_fn(2.0 * math.pi * i / (num_frames - 1)))
              for i in range(num_frames)]

    # Distinct keyframes at 0, 90, 180, 270 degrees (the 360 keyframe equals
    # the 0 keyframe, so we omit the duplicate).
    keyframe_angles = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
    cameras = [quantize(pose_fn(a)) for a in keyframe_angles]

    return {
        "focal_length": 500,
        "scale": 1,
        "cameras": cameras,
        "cameras_interp": interp,
    }


def write(mode: str) -> Path:
    data = build(mode)
    out_name = "camera_data_360.json" if mode == "orbit" else "camera_data_360_spin.json"
    out_path = Path(__file__).with_name(out_name)
    with out_path.open("w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
    print(f"Wrote {out_path} "
          f"(mode={mode}, {len(data['cameras_interp'])} interpolated frames, "
          f"{len(data['cameras'])} keyframes).")
    return out_path


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "orbit"
    if arg == "both":
        write("orbit")
        write("spin")
    else:
        write(arg)
