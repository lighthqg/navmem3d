#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def rdp(points,eps):
 if len(points)<3:return points
 a=np.asarray(points[0],float);b=np.asarray(points[-1],float);v=b-a;den=float(v@v); best=(0,0)
 for i,p in enumerate(points[1:-1],1):
  q=np.asarray(p,float);t=max(0,min(1,float(((q-a)@v)/den))) if den else 0; d=float(np.linalg.norm(q-(a+t*v)))
  if d>best[0]:best=(d,i)
 d,i=best
 return [points[0],points[-1]] if d<=eps else rdp(points[:i+1],eps)[:-1]+rdp(points[i:],eps)
def dist(a,b):return math.hypot(float(a[0])-float(b[0]),float(a[1])-float(b[1]))
def intersection(a,b,c,d):
 x1,y1=a;x2,y2=b;x3,y3=c;x4,y4=d; den=(x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
 if abs(den)<1e-8:return None
 t=((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/den; u=-((x1-x2)*(y1-y3)-(y1-y2)*(x1-x3))/den
 if 1e-4<t<1-1e-4 and 1e-4<u<1-1e-4:return (x1+t*(x2-x1),y1+t*(y2-y1))
 return None
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scene',type=Path,required=True);ap.add_argument('--graph',type=Path,required=True);ap.add_argument('--patrol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--simplify-px',type=float,default=4);ap.add_argument('--max-edge-px',type=float,default=35);args=ap.parse_args()
 meta=json.loads((args.scene/'occupancy.json').read_text());patrol=json.loads(args.patrol.read_text());visual=json.loads(args.graph.read_text());route=[(int(f['pixel'][0]),int(f['pixel'][1])) for f in patrol['frames']]
 simp=rdp(route,args.simplify_px);simp.append(simp[0])
 pts=[simp[0]]
 for a,b in zip(simp,simp[1:]):
  n=max(1,int(math.ceil(dist(a,b)/args.max_edge_px)))
  pts += [(round(a[0]+(b[0]-a[0])*k/n),round(a[1]+(b[1]-a[1])*k/n)) for k in range(1,n+1)]
 if pts[-1]==pts[0]:pts.pop()
 clean=[]
 for p in pts:
  if not clean or dist(p,clean[-1])>=2:clean.append(p)
 pts=clean;nodes=[{'node_id':f'topo_{i:03d}','type':'route_turn_or_sample','pixel':[int(x),int(y)]} for i,(x,y) in enumerate(pts)]
 edges=[]
 for i in range(len(nodes)):
  j=(i+1)%len(nodes);a,b=nodes[i]['pixel'],nodes[j]['pixel'];edges.append({'edge_id':f'corridor_{i:03d}','from_node':nodes[i]['node_id'],'to_node':nodes[j]['node_id'],'undirected':True,'polyline_pixel':[a,b],'pixel_length':dist(a,b)})
 # Split crossing patrol segments. A crossing in this single-floor patrol is
 # evidence that the robot can re-enter the route there, so it becomes a
 # shared undirected junction rather than two merely overlapping drawings.
 cuts={i:[] for i in range(len(edges))}
 for i,e in enumerate(edges):
  for j,f in enumerate(edges):
   if j<=i or len({i,j})==1: continue
   p=intersection(e['polyline_pixel'][0],e['polyline_pixel'][1],f['polyline_pixel'][0],f['polyline_pixel'][1])
   if p is not None: cuts[i].append(p);cuts[j].append(p)
 if any(cuts.values()):
  new_nodes=[]; new_edges=[]; key_to_node={}
  for n in nodes:new_nodes.append(n);key_to_node[tuple(n['pixel'])]=n['node_id']
  def node_for(p):
   k=(round(p[0],3),round(p[1],3))
   if k not in key_to_node:
    nid=f'topo_{len(new_nodes):03d}';key_to_node[k]=nid;new_nodes.append({'node_id':nid,'type':'patrol_intersection','pixel':[int(round(p[0])),int(round(p[1]))]})
   return key_to_node[k]
  for eidx,e in enumerate(edges):
   a,b=e['polyline_pixel']; ps=[a,*cuts[eidx],b]; ps.sort(key=lambda p:dist(a,p))
   for u,v in zip(ps,ps[1:]):
    if dist(u,v)<1:continue
    un=node_for(u);vn=node_for(v);new_edges.append({'edge_id':f'corridor_{len(new_edges):03d}','from_node':un,'to_node':vn,'undirected':True,'polyline_pixel':[[int(round(u[0])),int(round(u[1]))],[int(round(v[0])),int(round(v[1]))]],'pixel_length':dist(u,v)})
  nodes,edges=new_nodes,new_edges
 # Canonicalize duplicate pixels introduced by crossing splits.  Without this
 # pass the original endpoint and a split endpoint can become two isolated
 # IDs even though they occupy the same location.
 canonical={}; remap={}; merged=[]
 for n in nodes:
  key=tuple(n['pixel'])
  if key in canonical: remap[n['node_id']]=canonical[key]
  else: canonical[key]=n['node_id']; remap[n['node_id']]=n['node_id']; merged.append(n)
 nodes=merged
 for e in edges:
  e['from_node']=remap[e['from_node']]; e['to_node']=remap[e['to_node']]
 edges=[e for e in edges if e['from_node']!=e['to_node']]
 scale=float(meta['scale']);upper=meta['upper'];lower=meta['lower'];world=lambda x,y:[-scale*float(x)+float(upper[0]),scale*float(y)+float(lower[1])]
 for n in nodes:n['position_xy']=world(*n['pixel'])
 for e in edges:e['polyline_xy']=[world(*p) for p in e['polyline_pixel']];e['length_m']=e['pixel_length']*scale;e['cost']=e['length_m']
 # Bind one actual patrol frame to every topology node.
 topology_keyframes=[]
 for n in nodes:
  ni=min(range(len(patrol['frames'])),key=lambda i:dist(patrol['frames'][i]['pixel'],n['pixel']));f=patrol['frames'][ni];nd=dist(f['pixel'],n['pixel']);uri=str((args.patrol.parent/'renders'/f'{f["frame_id"]}.png').resolve());n.update({'reference_frame_id':f['frame_id'],'reference_frame_index':int(ni),'reference_rgb_uri':uri});topology_keyframes.append({'topology_node_id':n['node_id'],'source_frame_id':f['frame_id'],'source_frame_index':int(ni),'reference_rgb_uri':uri,'distance_pixel':float(nd)})
 out={'schema_version':'0.4','type':'undirected_topology_graph','source_observed_patrol':str(args.patrol.resolve()),'runtime_contract':'topology compiled from traversed path; no unobserved occupancy region is included','nodes':nodes,'edges':edges,'topology_keyframes':topology_keyframes,'visual_anchors_legacy':visual['nodes']};args.output.parent.mkdir(parents=True,exist_ok=True);args.output.with_suffix('.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 size=1200;margin=80;xs=[n['position_xy'][0] for n in nodes];ys=[n['position_xy'][1] for n in nodes];xmin,xmax=min(xs),max(xs);ymin,ymax=min(ys),max(ys);scl=min((size-2*margin)/max(xmax-xmin,1e-6),(size-2*margin)/max(ymax-ymin,1e-6));proj=lambda p:(round(margin+(p[0]-xmin)*scl),round(size-margin-(p[1]-ymin)*scl));canvas=Image.new('RGB',(size,size),(248,250,252));d=ImageDraw.Draw(canvas);font=ImageFont.load_default();d.text((20,20),'Undirected Patrol Topology',fill=(20,30,40),font=font);d.text((20,38),'blue: connectivity  green: topology node + reference patrol frame',fill=(90,100,110),font=font)
 for e in edges:d.line([proj(p) for p in e['polyline_xy']],fill=(65,105,145),width=5)
 for n in nodes:q=proj(n['position_xy']);d.ellipse((q[0]-8,q[1]-8,q[0]+8,q[1]+8),fill=(35,125,90),outline='white',width=2);d.text((q[0]+10,q[1]-6),n['node_id'].replace('topo_','T'),fill=(20,30,40),font=font)
 canvas.save(args.output);print(json.dumps({'topology_nodes':len(nodes),'edges':len(edges),'topology_keyframes':len(topology_keyframes),'output':str(args.output)},ensure_ascii=False))
if __name__=='__main__':main()
