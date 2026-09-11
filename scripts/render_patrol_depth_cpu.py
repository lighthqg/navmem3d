#!/usr/bin/env python3
"""CPU depth fallback for InteriorGS patrol using Gaussian centers as a z-buffer."""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import numpy as np
from PIL import Image

def load_points(path):
 from navmem3d.world.supersplat import load_supersplat_compressed_ply
 return load_supersplat_compressed_ply(path).means.astype(np.float32)

def basis(eye,target,up):
 f=np.asarray(target,np.float32)-np.asarray(eye,np.float32); f/=np.linalg.norm(f)
 r=np.cross(f,np.asarray(up,np.float32)); r/=np.linalg.norm(r)
 u=np.cross(r,f); return r,-u,f

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--asset',type=Path,required=True);ap.add_argument('--patrol',type=Path,required=True);ap.add_argument('--rgb-dir',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--width',type=int,default=640);ap.add_argument('--height',type=int,default=480);ap.add_argument('--hfov',type=float,default=90);ap.add_argument('--stride',type=int,default=1);a=ap.parse_args()
 pts=load_points(a.asset)[::max(1,a.stride)]; patrol=json.loads(a.patrol.read_text()); out=a.output_dir; out.mkdir(parents=True,exist_ok=True); (out/'depth').mkdir(exist_ok=True)
 fx=.5*a.width/math.tan(math.radians(a.hfov)/2); fy=fx; cx=a.width/2; cy=a.height/2; records=[]; t0=time.perf_counter()
 for k,fr in enumerate(patrol['frames']):
  eye=np.asarray(fr['camera_position'],np.float32); r,u,f=basis(eye,fr['look_at'],fr['up']); rel=pts-eye; z=rel@f; good=z>0.05; rel=rel[good]; z=z[good]; x=rel@r; y=rel@u; px=np.rint(fx*x/z+cx).astype(np.int32); py=np.rint(fy*y/z+cy).astype(np.int32); good=(px>=0)&(px<a.width)&(py>=0)&(py<a.height); px=px[good];py=py[good];z=z[good]
  depth=np.full((a.height,a.width),np.inf,np.float32); idx=py*a.width+px; np.minimum.at(depth.ravel(),idx,z); depth[~np.isfinite(depth)]=0
  np.save(out/'depth'/f"{fr['frame_id']}.npy",depth)
  records.append({'frame_id':fr['frame_id'],'timestamp_s':fr['timestamp'],'rgb_uri':str((a.rgb_dir/f"{fr['frame_id']}.png").resolve()),'depth_uri':str((out/'depth'/f"{fr['frame_id']}.npy").resolve()),'intrinsics':[fx,0,cx,0,fy,cy,0,0,1],'camera_position':fr['camera_position'],'look_at':fr['look_at'],'up':fr['up'],'pose_source':'patrol_sensor_pose','depth_source':'cpu_gaussian_center_zbuffer','valid_depth_fraction':float(np.count_nonzero(depth)/depth.size)})
  if k%50==0: print(k,records[-1]['valid_depth_fraction'],flush=True)
 manifest={'schema_version':'0.2','type':'robot_patrol_rgbd_manifest','world_role':'A_sensor_simulation','source_patrol':str(a.patrol.resolve()),'hidden_occupancy_consumed':False,'hidden_labels_consumed':False,'depth_model':'cpu_gaussian_center_zbuffer','gaussian_sample_stride':a.stride,'image_size':[a.width,a.height],'frames':records,'elapsed_s':time.perf_counter()-t0}
 (out/'robot_patrol_rgbd_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'frames':len(records),'output':str(out),'elapsed_s':manifest['elapsed_s']},ensure_ascii=False))
if __name__=='__main__':main()
