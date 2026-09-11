#!/usr/bin/env python3
"""Map World-A patrol camera poses into World-B using a B→A initializer."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def map_point(inv,p): return (inv @ np.r_[p,1.])[:3].tolist()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--patrol',type=Path,required=True);p.add_argument('--b-to-a',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 patrol=json.loads(args.patrol.read_text()); reg=json.loads(args.b_to_a.read_text())
 m=np.asarray(reg['transform_b_to_a'],float).reshape(4,4); inv=np.linalg.inv(m)
 frames=[]
 for f in patrol['frames']:
  eye=np.asarray(f['camera_position'],float); target=np.asarray(f['look_at'],float); up=np.asarray(f['up'],float)
  # Map two world points to preserve look direction; transform the up endpoint likewise.
  frames.append({'frame_id':f['frame_id'],'timestamp':f['timestamp'],'camera_position':map_point(inv,eye),'look_at':map_point(inv,target),'up':(np.asarray(map_point(inv,eye+up))-np.asarray(map_point(inv,eye))).tolist(),'source_a_camera_position':eye.tolist()})
 out={'schema_version':'0.1','world_role':'B_internal_twin','purpose':'coarse B renders at the known uploaded A patrol poses','source_patrol':str(args.patrol.resolve()),'source_b_to_a':str(args.b_to_a.resolve()),'initializer_status':reg['status'],'frames':frames}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
