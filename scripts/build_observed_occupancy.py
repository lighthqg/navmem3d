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

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--scale',type=float,default=.05);ap.add_argument('--sample',type=int,default=6);ap.add_argument('--max-depth',type=float,default=12);ap.add_argument('--robot-radius',type=float,default=.2);a=ap.parse_args()
 m=json.loads(a.manifest.read_text()); fs=m['frames']; all_xy=[]
 for fr in fs:
  eye=np.asarray(fr['camera_position'],float); all_xy.append(eye[:2])
  d=np.load(fr['depth_uri']); ys,xs=np.where(d>0); sel=np.arange(len(xs))[::max(1,a.sample)]
  fx,_,cx,_,fy,cy,_,_,_=fr['intrinsics']; r=np.asarray(fr['look_at'])-eye; r/=np.linalg.norm(r); right=np.cross(r,np.asarray(fr['up']));right/=np.linalg.norm(right); up=np.cross(right,r)
  z=d[ys[sel],xs[sel]]; good=(z>0.05)&(z<a.max_depth); xx=(xs[sel][good]-cx)/fx*z[good]; yy=(ys[sel][good]-cy)/fy*z[good]; pts=eye[None,:]+xx[:,None]*right+yy[:,None]*up+z[good,None]*r; all_xy.append(pts[:,:2])
 arr=np.concatenate([x if np.asarray(x).ndim==2 else np.asarray(x)[None,:] for x in all_xy]); lo=arr.min(0)-.4; hi=arr.max(0)+.4; w=int(math.ceil((hi[0]-lo[0])/a.scale))+1; h=int(math.ceil((hi[1]-lo[1])/a.scale))+1; state=np.full((h,w),127,np.uint8)
 def pix(x,y): return int(round((x-lo[0])/a.scale)), int(round((y-lo[1])/a.scale))
 for fr in fs:
  eye=np.asarray(fr['camera_position'],float); ex,ey=pix(*eye[:2]);
  if 0<=ex<w and 0<=ey<h: state[ey,ex]=255
  d=np.load(fr['depth_uri']); fx,_,cx,_,fy,cy,_,_,_=fr['intrinsics']; f=np.asarray(fr['look_at'])-eye; f/=np.linalg.norm(f); right=np.cross(f,np.asarray(fr['up']));right/=np.linalg.norm(right); up=np.cross(right,f)
  ys,xs=np.where((d>0.05)&(d<a.max_depth)); sel=np.arange(len(xs))[::max(1,a.sample)]
  for j in sel:
   z=float(d[ys[j],xs[j]]); x=(xs[j]-cx)/fx*z; y=(ys[j]-cy)/fy*z; p=eye+x*right+y*up+z*f; hx,hy=pix(p[0],p[1]);
   if not (0<=hx<w and 0<=hy<h): continue
   ray=list(bresenham(ex,ey,hx,hy));
   for qx,qy in ray[:-1]:
    if 0<=qx<w and 0<=qy<h and state[qy,qx]!=0: state[qy,qx]=255
   if state[hy,hx]!=255: state[hy,hx]=0
 # clear isolated occupied speckles and inflate conservatively
 occ=np.argwhere(state==0); rad=max(0,int(math.ceil(a.robot_radius/a.scale))); inflated=state.copy()
 if rad:
  yy,xx=np.ogrid[-rad:rad+1,-rad:rad+1]; mask=xx*xx+yy*yy<=rad*rad
  for y,x in occ:
   y0=max(0,y-rad);y1=min(h,y+rad+1);x0=max(0,x-rad);x1=min(w,x+rad+1); sub=inflated[y0:y1,x0:x1]; mm=mask[(y0-(y-rad)):(y1-(y-rad)),(x0-(x-rad)):(x1-(x-rad))]; sub[mm]=0
 out=a.output_dir;out.mkdir(parents=True,exist_ok=True);Image.fromarray(inflated).save(out/'observed_occupancy.png');np.save(out/'observed_occupancy.npy',inflated)
 meta={'schema_version':'0.2','type':'robot_observed_occupancy','source_manifest':str(a.manifest.resolve()),'hidden_occupancy_consumed':False,'hidden_labels_consumed':False,'state_encoding':{'occupied':0,'unknown':127,'free':255},'scale_m':a.scale,'origin_xy':lo.tolist(),'width':w,'height':h,'robot_radius_m':a.robot_radius,'depth_sample_stride':a.sample,'observed_free_cells':int(np.count_nonzero(inflated==255)),'observed_occupied_cells':int(np.count_nonzero(inflated==0)),'unknown_cells':int(np.count_nonzero(inflated==127))}
 (out/'occupancy_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');print(json.dumps(meta,ensure_ascii=False))
if __name__=='__main__':main()
