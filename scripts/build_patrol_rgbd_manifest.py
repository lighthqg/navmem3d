#!/usr/bin/env python3
"""Create the robot-observation RGB-D manifest for an InteriorGS patrol.

Depth is rendered from the evaluator scene only to simulate a real depth
sensor. The downstream mapping contract consumes this manifest, never the
scene's hidden occupancy/labels.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--patrol',type=Path,required=True);ap.add_argument('--rgb-dir',type=Path,required=True);ap.add_argument('--depth-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--width',type=int,default=640);ap.add_argument('--height',type=int,default=480);ap.add_argument('--hfov',type=float,default=90.0);a=ap.parse_args()
 p=json.loads(a.patrol.read_text());focal=.5*a.width/math.tan(math.radians(a.hfov)/2);frames=[]
 for f in p['frames']:
  stem=f['frame_id'];rgb=a.rgb_dir/f'{stem}.png';depth=a.depth_dir/f'{stem}.npy'
  if not rgb.exists():raise FileNotFoundError(rgb)
  if not depth.exists():raise FileNotFoundError(depth)
  frames.append({'frame_id':stem,'timestamp_s':f['timestamp'],'rgb_uri':str(rgb.resolve()),'depth_uri':str(depth.resolve()),'intrinsics':[focal,0,a.width/2,0,focal,a.height/2,0,0,1],'camera_position':f['camera_position'],'look_at':f['look_at'],'up':f['up'],'pose_source':'patrol_sensor_pose','observation_role':'robot_sensor_input'})
 out={'schema_version':'0.1','type':'robot_patrol_rgbd_manifest','world_role':'A_sensor_simulation','source_patrol':str(a.patrol.resolve()),'hidden_occupancy_consumed':False,'hidden_labels_consumed':False,'depth_model':'to_be_rendered_from_A_at_matching_camera_pose','frames':frames};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'frames':len(frames),'output':str(a.output)},ensure_ascii=False))
if __name__=='__main__':main()
