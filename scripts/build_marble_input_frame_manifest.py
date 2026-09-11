#!/usr/bin/env python3
"""Build an auditable pose/intrinsics manifest for a Marble patrol video.

The input trajectory stores camera position/look-at/up in World A.  This is
purposefully retained as the source-of-truth convention; no silently assumed
camera matrix convention is imposed on downstream registration.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--render-dir', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--hfov-deg', type=float, default=90.0)
    args = parser.parse_args()
    trajectory = json.loads(args.trajectory.read_text())
    focal = 0.5 * args.width / math.tan(math.radians(args.hfov_deg) / 2.0)
    frames = []
    for index, source in enumerate(trajectory['frames']):
        rgb = args.render_dir / f'frame_{index:04d}.png'
        if not rgb.exists():
            raise FileNotFoundError(rgb)
        frames.append({
            'marble_video_frame': index,
            'timestamp_s': source['timestamp'],
            'source_patrol_frame_id': source['frame_id'],
            'rgb_uri': str(rgb.resolve()),
            'depth_uri': None,
            'intrinsics_pinhole': [focal, 0.0, args.width / 2, 0.0, focal, args.height / 2, 0.0, 0.0, 1.0],
            'camera_pose_world_a': {
                'representation': 'position_look_at_up',
                'camera_position': source['camera_position'],
                'look_at': source['look_at'],
                'up': source['up'],
            },
            'occupancy_pixel': source['pixel'],
        })
    out = {
        'schema_version': '0.1',
        'world_a': trajectory['sequence_id'],
        'world_b': 'marble_bbbc5728',
        'purpose': 'input-frame anchors for B-to-A registration',
        'video_uri': str(args.video.resolve()),
        'video_fps': trajectory['fps'],
        'world_a_coordinate_frame': trajectory['coordinate_frame'],
        'camera_convention': 'world pose is explicitly position/look_at/up; derive a renderer-specific T_world_camera only at the consumer boundary',
        'capture_mode': 'rendered RGB from InteriorGS World-A 3DGS along an occupancy-feasible robot patrol',
        'depth_available': False,
        'depth_note': 'No per-frame depth was exported for this Marble submission. Camera centers suffice for the first Sim(3) fit; depth can be re-rendered later if needed.',
        'frames': frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')

if __name__ == '__main__': main()
