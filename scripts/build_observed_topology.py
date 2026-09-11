#!/usr/bin/env python3
"""Compile an undirected navigation topology from robot-observed occupancy only."""
from __future__ import annotations
import argparse,json,math,heapq
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def astar(free,s,g):
 h,w=free.shape; q=[(0,0,s)]; cost={s:0}; prev={}; moves=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]
 while q:
  _,c,p=heapq.heappop(q)
  if c!=cost.get(p):continue
  if p==g:
   out=[p]
   while p in prev:p=prev[p];out.append(p)
   return out[::-1]
  for dx,dy in moves:
   n=(p[0]+dx,p[1]+dy)
   if not(0<=n[0]<w and 0<=n[1]<h) or not free[n[1],n[0]]:continue
   if dx and dy and (not free[p[1],n[0]] or not free[n[1],p[0]]):continue
   nc=c+(1.4142 if dx and dy else 1)
   if nc<cost.get(n,1e18):cost[n]=nc;prev[n]=p;heapq.heappush(q,(nc+math.hypot(g[0]-n[0],g[1]-n[1]),nc,n))
 return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--occupancy',type=Path,required=True);ap.add_argument('--metadata',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--spacing-px',type=int,default=12);a=ap.parse_args()
 import cv2
 img=np.array(Image.open(a.occupancy).convert('L')); meta=json.loads(a.metadata.read_text()); free=img==255
 # remove isolated speckle and close narrow gaps in observed free space
 k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)); clean=cv2.morphologyEx(free.astype(np.uint8),cv2.MORPH_OPEN,k); clean=cv2.morphologyEx(clean,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))).astype(bool)
 n,lab,stats,_=cv2.connectedComponentsWithStats(clean.astype(np.uint8),8); idx=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]); clean=lab==idx
 # Morphological skeleton of the observed free component.  Nodes are derived
 # from skeleton endpoints/junctions and regularly sampled skeleton pixels,
 # rather than from a free-space lattice (which creates false shortcuts).
 sk=np.zeros_like(clean,dtype=np.uint8); work=clean.astype(np.uint8)
 kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
 while work.any():
  er=cv2.erode(work,kernel); op=cv2.dilate(er,kernel); sk |= work & (~op); work=er
 ys,xs=np.where(sk>0); pixset={(int(x),int(y)) for x,y in zip(xs,ys)}
 def neigh(p):
  x,y=p; return [(xx,yy) for yy in range(y-1,y+2) for xx in range(x-1,x+2) if (xx,yy)!=(x,y) and (xx,yy) in pixset]
 special=[p for p in pixset if len(neigh(p))!=2]
 # cluster nearby special pixels into one junction/endpoint
 nodes=[]; taken=set()
 for p in special:
  if p in taken: continue
  stack=[p];taken.add(p);group=[]
  while stack:
   q=stack.pop();group.append(q)
   for z in neigh(q):
    if z in special and z not in taken:taken.add(z);stack.append(z)
  nodes.append(tuple(map(int,(sum(x for x,y in group)/len(group),sum(y for x,y in group)/len(group)))))
 # sample long skeleton runs so execution has intermediate reference views
 for p in sorted(pixset,key=lambda z:(z[1],z[0])):
  if all((p[0]-q[0])**2+(p[1]-q[1])**2 >= a.spacing_px*a.spacing_px for q in nodes): nodes.append(p)
 # Connect only along the morphological skeleton.  A* on the full free mask
 # would create shortcuts across rooms; the skeleton constraint preserves the
 # observed corridor structure while allowing sparse node sampling.
 manifest=json.loads(a.manifest.read_text()); frames=manifest['frames']; ox,oy=meta['origin_xy']; scale=meta['scale_m']
 def fp(fr): return ((fr['camera_position'][0]-ox)/scale,(fr['camera_position'][1]-oy)/scale)
 frame_nodes=[]
 for fr in frames:
  p=fp(fr); frame_nodes.append(min(range(len(nodes)),key=lambda i:(nodes[i][0]-p[0])**2+(nodes[i][1]-p[1])**2))
 transitions=[]
 for i,p in enumerate(nodes):
  near=sorted((math.hypot(p[0]-q[0],p[1]-q[1]),j) for j,q in enumerate(nodes) if j!=i)[:4]
  for dist,j in near:
   if dist<=a.spacing_px*2.8 and (i,j) not in transitions and (j,i) not in transitions: transitions.append((i,j))
 edges=[]
 for u,v in transitions:
  path=astar(sk>0,nodes[u],nodes[v])
  if not path: continue
  length_px=len(path)-1
  edges.append({'edge_id':f'edge_{len(edges):03d}','source':f'topo_{u:03d}','target':f'topo_{v:03d}','from_node':f'topo_{u:03d}','to_node':f'topo_{v:03d}','length_px':length_px,'length_m':float(length_px*scale),'cost':float(length_px*scale),'polyline_px':[[int(x),int(y)] for x,y in path]})
 # add nearest patrol reference frame
 ox,oy=meta['origin_xy'];scale=meta['scale_m']
 def world(px):return [ox+px[0]*scale,oy+px[1]*scale]
 outnodes=[]
 for i,p in enumerate(nodes):
  wp=world(p); fi=min(range(len(frames)),key=lambda k:(frames[k]['camera_position'][0]-wp[0])**2+(frames[k]['camera_position'][1]-wp[1])**2); fr=frames[fi]
  outnodes.append({'node_id':f'topo_{i:03d}','type':'observed_free_anchor','pixel':[p[0],p[1]],'position_xy':wp,'reference_frame_id':fr['frame_id'],'reference_rgb_uri':fr['rgb_uri']})
 # visualization
 W=1200;H=900; canvas=Image.new('RGB',(W,H),'#7f7f7f'); d=ImageDraw.Draw(canvas); sx=(W-40)/img.shape[1];sy=(H-40)/img.shape[0]
 def pp(p):return (20+p[0]*sx,20+p[1]*sy)
 for e in edges:d.line([pp(p) for p in e['polyline_px']],fill='#55d6ff',width=2)
 for i,nod in enumerate(outnodes):
  x,y=pp(nod['pixel']);d.ellipse((x-4,y-4,x+4,y+4),fill='#ff8c2a');d.text((x+5,y-5),str(i),fill='white')
 d.text((20,5),f'Observed occupancy topology | nodes={len(outnodes)} edges={len(edges)}',fill='white'); a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'schema_version':'0.2','type':'undirected_observed_occupancy_topology','source_occupancy':str(a.occupancy.resolve()),'hidden_occupancy_consumed':False,'unknown_is_blocked':True,'nodes':outnodes,'edges':edges,'topology_keyframes':[{'topology_node_id':n['node_id'],'source_frame_id':n['reference_frame_id'],'reference_rgb_uri':n['reference_rgb_uri']} for n in outnodes]},ensure_ascii=False,indent=2)+'\n');canvas.save(a.output.with_suffix('.png'));print(json.dumps({'nodes':len(nodes),'edges':len(edges),'output':str(a.output)},ensure_ascii=False))
if __name__=='__main__':main()
