#!/usr/bin/env python3
"""Render the spatial observation layer and topology layer together.

The renderer is data-driven: it never infers edges from pixel proximity. It
accepts a three-state occupancy image plus an undirected topology graph and
only overlays the graph records supplied by the caller.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--occupancy',type=Path,required=True);ap.add_argument('--topology',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--route',type=Path);ap.add_argument('--scale',type=int,default=4);ap.add_argument('--label-mode',choices=('none','route','all'),default='route');a=ap.parse_args()
 occ=np.array(Image.open(a.occupancy).convert('L')); topo=json.loads(a.topology.read_text()); route=[]
 if a.route and a.route.exists(): route=json.loads(a.route.read_text()).get('node_sequence',[])
 h,w=occ.shape; im=Image.new('RGB',(w,h),'#b5b5b5'); pix=np.zeros((h,w,3),np.uint8); pix[occ==127]=[155,155,155];pix[occ==255]=[242,242,242];pix[occ==0]=[20,20,20];im=Image.fromarray(pix)
 # obstacle outline from the spatial layer
 d=ImageDraw.Draw(im); edge=np.zeros_like(occ,bool); edge[1:]=edge[1:] | ((occ[1:] == 0) & (occ[:-1] != 0));edge[:-1]|=((occ[:-1]==0)&(occ[1:]!=0));edge[:,1:]|=((occ[:,1:]==0)&(occ[:,:-1]!=0));edge[:,:-1]|=((occ[:,:-1]==0)&(occ[:,1:]!=0));
 for y,x in zip(*np.where(edge)): d.point((int(x),int(y)),fill=(90,90,90))
 nodes={n['node_id']:n for n in topo['nodes']}; route=set(route)
 for e in topo.get('edges',[]):
  pts=e.get('polyline_pixel') or e.get('polyline_px')
  if not pts:
   u=nodes[e.get('from_node',e.get('source'))]['pixel'];v=nodes[e.get('to_node',e.get('target'))]['pixel'];pts=[u,v]
  d.line([(int(p[0]),int(p[1])) for p in pts],fill=(36,155,214) if not ({e.get('from_node',e.get('source')),e.get('to_node',e.get('target'))}&route) else (236,100,55),width=2)
 for i,n in enumerate(topo['nodes']):
  x,y=map(int,n['pixel']); selected=n['node_id'] in route; c=(242,145,35) if selected else (30,125,92);d.ellipse((x-3,y-3,x+3,y+3),fill=c,outline='white')
  if a.label_mode=='all' or (a.label_mode=='route' and selected): d.text((x+4,y-4),n['node_id'].replace('topo_','T'),fill=(20,20,20))
 im.resize((w*a.scale,h*a.scale),Image.Resampling.NEAREST).save(a.output)
 meta={'schema_version':'0.1','type':'layered_space_topology_visualization','spatial_layer':str(a.occupancy.resolve()),'topology_layer':str(a.topology.resolve()),'route':str(a.route.resolve()) if a.route else None,'obstacle_outline':'derived from occupied/free/unknown boundaries','graph_edges_source':'topology JSON only','label_mode':a.label_mode,'nodes':len(topo['nodes']),'edges':len(topo.get('edges',[])),'route_nodes':sorted(route)}
 a.output.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(meta,ensure_ascii=False))
if __name__=='__main__':main()
