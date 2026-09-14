#!/usr/bin/env python3
"""Compile an undirected topology only from a recorded patrol and generated M occupancy."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

def rdp(points:np.ndarray,eps:float)->list[int]:
 def rec(ids):
  if len(ids)<3:return ids
  a,b=points[ids[0]],points[ids[-1]];v=b-a;den=float(v@v)
  d=np.array([np.linalg.norm(points[i]-(a+(np.clip(((points[i]-a)@v)/(den or 1),0,1))*v)) for i in ids[1:-1]])
  j=int(np.argmax(d)) if len(d) else 0
  # RDP keeps only segment endpoints once all intermediate samples fit the
  # tolerance.  Returning ``ids`` here would silently retain every frame.
  return [ids[0],ids[-1]] if not len(d) or d[j]<=eps else rec(ids[:j+2])[:-1]+rec(ids[j+1:])
 return rec(list(range(len(points))))
def line(x0,y0,x1,y1):
 dx,sx=abs(x1-x0),1 if x0<x1 else -1;dy,sy=-abs(y1-y0),1 if y0<y1 else -1;e=dx+dy
 while True:
  yield x0,y0
  if (x0,y0)==(x1,y1):return
  q=2*e
  if q>=dy:e+=dy;x0+=sx
  if q<=dx:e+=dx;y0+=sy
p=argparse.ArgumentParser();p.add_argument('--patrol',type=Path,required=True);p.add_argument('--occupancy',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--turn-epsilon-m',type=float,default=.35);p.add_argument('--max-edge-m',type=float,default=1.0);a=p.parse_args()
frames=json.loads(a.patrol.read_text())['frames'];xy=np.asarray([f['camera_position'][:2] for f in frames],float)
# Patrols are commonly closed loops.  RDP on a loop whose first/last point
# coincide has a zero baseline and marks every sample as a turn.  Split at the
# point farthest from the start, simplify the two open halves, then rejoin.
if np.linalg.norm(xy[0]-xy[-1]) < 1e-4 and len(xy) > 3:
 pivot=int(np.argmax(np.linalg.norm(xy-xy[0],axis=1)))
 keep=rdp(xy[:pivot+1],a.turn_epsilon_m)[:-1]+[i+pivot for i in rdp(xy[pivot:],a.turn_epsilon_m)]
else:
 keep=rdp(xy,a.turn_epsilon_m)
# sample each simplified segment, but retain its chronological reference frame.
samples=[]
for left,right in zip(keep,keep[1:]):
 n=max(1,int(math.ceil(np.linalg.norm(xy[right]-xy[left])/a.max_edge_m)))
 samples += [round(left+(right-left)*k/n) for k in range(n)]
samples.append(keep[-1]);samples=sorted(set(samples))
nodes=[{'node_id':f'topo_{i:03d}','type':'patrol_turn_or_sample','position_xy':xy[k].tolist(),'reference_frame_id':frames[k]['frame_id'],'reference_frame_index':int(k),'reference_rgb_uri':frames[k].get('rgb_uri')} for i,k in enumerate(samples)]
# An edge is the *recorded patrol polyline* between key nodes, not a chord
# between their positions.  RDP/sampling only reduces route decisions; it must
# never change the physical path the robot actually traversed.
node_frame_indices=[n['reference_frame_index'] for n in nodes]
def make_edge(edge_id,left,right,evidence,polyline):
 polyline=np.asarray(polyline,float)
 length=float(np.linalg.norm(np.diff(polyline,axis=0),axis=1).sum()) if len(polyline)>1 else 0.0
 return {'edge_id':edge_id,'from_node':nodes[left]['node_id'],'to_node':nodes[right]['node_id'],
         'undirected':True,'cost':length,'length_m':length,'evidence':evidence,
         'polyline_xy':polyline.tolist(),'source_frame_index_range':[int(node_frame_indices[left]),int(node_frame_indices[right])]}
edges=[]
for i in range(len(nodes)-1):
 start,end=node_frame_indices[i],node_frame_indices[i+1]
 edges.append(make_edge(f'edge_{i:03d}',i,i+1,'consecutive_patrol_segment',xy[start:end+1]))
if np.linalg.norm(xy[0]-xy[-1]) < 1e-4 and len(nodes) > 2:
 start,end=node_frame_indices[-1],node_frame_indices[0]
 # A closed patrol normally ends at its first pose.  Retain every final
 # recorded sample, then append the initial segment if node 0 is not frame 0.
 closure=np.concatenate([xy[start:],xy[1:end+1]],axis=0) if end else xy[start:]
 if len(closure) == 1:
  # The logged final pose can equal the first pose exactly. Preserve an explicit
  # zero-length return geometry instead of leaving a malformed edge.
  closure=np.vstack([closure, xy[end]])
 edges.append(make_edge(f'edge_{len(edges):03d}',len(nodes)-1,0,'closed_patrol_return',closure))
# Validate rather than infer: every recorded segment must lie in generated free space.
state=np.asarray(Image.open(a.occupancy).convert('L'));m=json.loads(a.metadata.read_text());o=np.asarray(m['origin_xy']);s=float(m['scale_m']);pix=np.floor((xy-o)/s).astype(int);bad=0;total=0
for u,v in zip(pix,pix[1:]):
 for x,y in line(*u,*v):
  total+=1
  if not(0<=x<state.shape[1] and 0<=y<state.shape[0]) or state[y,x]!=255:bad+=1
keyframes=[{'topology_node_id':n['node_id'],'source_frame_id':n['reference_frame_id'],'source_frame_index':n['reference_frame_index'],'reference_rgb_uri':n['reference_rgb_uri']} for n in nodes]
out={'schema_version':'1.1','type':'undirected_patrol_topology','source_patrol':str(a.patrol.resolve()),'source_occupancy':str(a.occupancy.resolve()),'hidden_occupancy_consumed':False,'edges_are_observed_patrol_only':True,'closed_patrol':bool(np.linalg.norm(xy[0]-xy[-1]) < 1e-4),'nodes':nodes,'edges':edges,'topology_keyframes':keyframes,'patrol_free_validation':{'checked_cells':total,'non_free_cells':bad,'passed':bad==0}}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'nodes':len(nodes),'edges':len(edges),'validation':out['patrol_free_validation']},ensure_ascii=False))
