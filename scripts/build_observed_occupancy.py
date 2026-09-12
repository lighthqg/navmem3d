#!/usr/bin/env python3
"""Build a robot-observed three-state XY occupancy grid from RGB-D patrol data."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image

def bresenham(x0,y0,x1,y1):
 dx=abs(x1-x0); sx=1 if x0<x1 else -1; dy=-abs(y1-y0); sy=1 if y0<y1 else -1; err=dx+dy; x,y=x0,y0
 while True:
  yield x,y
  if x==x1 and y==y1: break
  e2=2*err
  if e2>=dy: err+=dy;x+=sx
  if e2<=dx: err+=dx;y+=sy

def sampled_pixels(depth, sample, scanline_rows):
 h,w=depth.shape
 if scanline_rows:
  rows=[min(h-1,max(0,int(v))) for v in scanline_rows.split(',') if v.strip()]
  columns=np.arange(0,w,max(1,sample),dtype=np.int32)
  return np.repeat(np.asarray(rows,dtype=np.int32),len(columns)),np.tile(columns,len(rows))
 ys,xs=np.where(depth>0)
 return ys[::max(1,sample)],xs[::max(1,sample)]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--scale',type=float,default=.05);ap.add_argument('--sample',type=int,default=6);ap.add_argument('--scanline-rows',default='',help='comma-separated depth rows for a 2D LiDAR-like scan');ap.add_argument('--mapping-mode',choices=('ray','elevation'),default='ray');ap.add_argument('--ground-height-m',type=float,default=.12,help='endpoints below this height are floor free-space evidence, not obstacles');ap.add_argument('--min-obstacle-hits',type=int,default=3,help='multi-view elevation evidence required for an occupied cell');ap.add_argument('--max-depth',type=float,default=12);ap.add_argument('--robot-radius',type=float,default=.2);a=ap.parse_args()
 m=json.loads(a.manifest.read_text()); fs=m['frames']; all_xy=[]
 for fr in fs:
  eye=np.asarray(fr['camera_position'],float); all_xy.append(eye[:2])
  d=np.load(fr['depth_uri']); ys,xs=sampled_pixels(d,a.sample,a.scanline_rows)
  fx,_,cx,_,fy,cy,_,_,_=fr['intrinsics']; r=np.asarray(fr['look_at'])-eye; r/=np.linalg.norm(r); right=np.cross(r,np.asarray(fr['up']));right/=np.linalg.norm(right); up=np.cross(right,r)
  z=d[ys,xs]; good=(z>0.05)&(z<a.max_depth); xx=(xs[good]-cx)/fx*z[good]; yy=(ys[good]-cy)/fy*z[good]; pts=eye[None,:]+xx[:,None]*right-yy[:,None]*up+z[good,None]*r; all_xy.append(pts[:,:2])
 arr=np.concatenate([x if np.asarray(x).ndim==2 else np.asarray(x)[None,:] for x in all_xy]); lo=arr.min(0)-.4; hi=arr.max(0)+.4; w=int(math.ceil((hi[0]-lo[0])/a.scale))+1; h=int(math.ceil((hi[1]-lo[1])/a.scale))+1; state=np.full((h,w),127,np.uint8)
 def pix(x,y): return int(round((x-lo[0])/a.scale)), int(round((y-lo[1])/a.scale))
 free_hits=np.zeros((h,w),np.uint16) if a.mapping_mode=='elevation' else None
 occupied_hits=np.zeros((h,w),np.uint16) if a.mapping_mode=='elevation' else None
 for fr in fs:
  eye=np.asarray(fr['camera_position'],float); ex,ey=pix(*eye[:2]);
  if 0<=ex<w and 0<=ey<h: state[ey,ex]=255
  d=np.load(fr['depth_uri']); fx,_,cx,_,fy,cy,_,_,_=fr['intrinsics']; f=np.asarray(fr['look_at'])-eye; f/=np.linalg.norm(f); right=np.cross(f,np.asarray(fr['up']));right/=np.linalg.norm(right); up=np.cross(right,f)
  ys,xs=sampled_pixels(d,a.sample,a.scanline_rows)
  for j in range(len(xs)):
   z=float(d[ys[j],xs[j]]); x=(xs[j]-cx)/fx*z; y=(ys[j]-cy)/fy*z; p=eye+x*right-y*up+z*f; hx,hy=pix(p[0],p[1]);
   if not (0<=hx<w and 0<=hy<h): continue
   if a.mapping_mode=='ray':
    ray=list(bresenham(ex,ey,hx,hy));
    for qx,qy in ray[:-1]:
     if 0<=qx<w and 0<=qy<h and state[qy,qx]!=0: state[qy,qx]=255
   # In elevation mode, only observed floor endpoints yield free evidence.
   # This prevents high camera rays from carving through table tops.
   if a.mapping_mode=='elevation':
    if p[2] > a.ground_height_m: occupied_hits[hy,hx]+=1
    else: free_hits[hy,hx]+=1
   elif p[2] > a.ground_height_m: state[hy,hx]=0
   elif state[hy,hx]!=0: state[hy,hx]=255
 if a.mapping_mode=='elevation':
  state[:]=127
  free_confirm=free_hits>0
  occupied_confirm=(occupied_hits>=a.min_obstacle_hits)&(occupied_hits>=free_hits)
  state[free_confirm]=255
  state[occupied_confirm]=0
 # Preserve the uninflated sensor map for geometric evaluation.  The output
 # used by navigation below is a separate costmap, not this raw layer.
 raw_sensor=state.copy()
 # clear isolated occupied speckles and inflate conservatively
 occ=np.argwhere(state==0); rad=max(0,int(math.ceil(a.robot_radius/a.scale))); inflated=state.copy()
 if rad:
  yy,xx=np.ogrid[-rad:rad+1,-rad:rad+1]; mask=xx*xx+yy*yy<=rad*rad
  for y,x in occ:
   y0=max(0,y-rad);y1=min(h,y+rad+1);x0=max(0,x-rad);x1=min(w,x+rad+1); sub=inflated[y0:y1,x0:x1]; mm=mask[(y0-(y-rad)):(y1-(y-rad)),(x0-(x-rad)):(x1-(x-rad))]; sub[mm]=0
 # The robot's swept footprint is direct free-space evidence.  Apply it
 # after obstacle inflation so a noisy depth endpoint cannot mark a place
 # that the robot demonstrably traversed as occupied.
 patrol_pixels=[pix(*np.asarray(fr['camera_position'],float)[:2]) for fr in fs]
 for start,end in zip(patrol_pixels,patrol_pixels[1:]):
  for cx,cy in bresenham(*start,*end):
   y0=max(0,cy-rad);y1=min(h,cy+rad+1);x0=max(0,cx-rad);x1=min(w,cx+rad+1)
   if rad:
    mm=mask[(y0-(cy-rad)):(y1-(cy-rad)),(x0-(cx-rad)):(x1-(cx-rad))]
    inflated[y0:y1,x0:x1][mm]=255
   elif 0<=cx<w and 0<=cy<h:
    inflated[cy,cx]=255
 out=a.output_dir;out.mkdir(parents=True,exist_ok=True);Image.fromarray(raw_sensor).save(out/'observed_occupancy_raw.png');np.save(out/'observed_occupancy_raw.npy',raw_sensor);Image.fromarray(inflated).save(out/'observed_occupancy.png');np.save(out/'observed_occupancy.npy',inflated)
 meta={'schema_version':'0.6','type':'robot_observed_occupancy','source_manifest':str(a.manifest.resolve()),'hidden_occupancy_consumed':False,'hidden_labels_consumed':False,'state_encoding':{'occupied':0,'unknown':127,'free':255},'raw_sensor_layer':'observed_occupancy_raw.png','navigation_costmap_layer':'observed_occupancy.png','scale_m':a.scale,'origin_xy':lo.tolist(),'width':w,'height':h,'robot_radius_m':a.robot_radius,'depth_sample_stride':a.sample,'scanline_rows':a.scanline_rows or None,'mapping_mode':a.mapping_mode,'ground_height_m':a.ground_height_m,'min_obstacle_hits':a.min_obstacle_hits if a.mapping_mode=='elevation' else None,'traversed_footprint_carved':True,'raw_sensor_free_cells':int(np.count_nonzero(raw_sensor==255)),'raw_sensor_occupied_cells':int(np.count_nonzero(raw_sensor==0)),'observed_free_cells':int(np.count_nonzero(inflated==255)),'observed_occupied_cells':int(np.count_nonzero(inflated==0)),'unknown_cells':int(np.count_nonzero(inflated==127))}
 (out/'occupancy_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');print(json.dumps(meta,ensure_ascii=False))
if __name__=='__main__':main()
