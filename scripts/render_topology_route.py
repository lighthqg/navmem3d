#!/usr/bin/env python3
"""Render a three-state occupancy map, observed topology, and an optional route."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--occupancy',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--topology',type=Path,required=True);p.add_argument('--route',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
grid=np.asarray(Image.open(a.occupancy).convert('L'));m=json.loads(a.metadata.read_text());g=json.loads(a.topology.read_text());route=json.loads(a.route.read_text()) if a.route else None
canvas=np.empty((*grid.shape,3),np.uint8);canvas[grid==255]=(245,247,250);canvas[grid==0]=(20,23,28);canvas[grid==127]=(128,132,138)
im=Image.fromarray(canvas).resize((grid.shape[1]*2,grid.shape[0]*2),Image.Resampling.NEAREST);draw=ImageDraw.Draw(im);nodes={x['node_id']:x for x in g['nodes']};origin=np.asarray(m['origin_xy']);scale=float(m['scale_m'])
def point(node_id):return tuple((np.floor((np.asarray(nodes[node_id]['position_xy'])-origin)/scale)*2).astype(int))
def polyline(edge):
 pts=np.asarray(edge.get('polyline_xy') or [nodes[edge['from_node']]['position_xy'],nodes[edge['to_node']]['position_xy']])
 return [tuple((np.floor((xy-origin)/scale)*2).astype(int)) for xy in pts]
edges={edge['edge_id']:edge for edge in g['edges']}
for edge in edges.values():draw.line(polyline(edge),fill=(75,170,210),width=2)
for n in nodes:
 x,y=point(n);draw.ellipse((x-3,y-3,x+3,y+3),fill=(255,185,35),outline=(0,0,0))
if route:
 for edge_id in route['edge_sequence']:
  draw.line(polyline(edges[edge_id]),fill=(70,220,255),width=4)
 for label,node,color in [('start',route['start_topology_node_id'],(60,245,110)),('target',route['target_topology_node_id'],(255,70,110))]:
  x,y=point(node);draw.ellipse((x-7,y-7,x+7,y+7),fill=color,outline=(0,0,0));draw.text((x+8,y-8),label,fill='white',stroke_width=2,stroke_fill='black')
 draw.text((10,10),f"query: {route['query_term']} | route: {route['cost_m']:.2f} m | confirm in current RGB",fill='white',stroke_width=2,stroke_fill='black')
else:draw.text((10,10),f"observed topology | {len(nodes)} nodes / {len(g['edges'])} edges",fill='white',stroke_width=2,stroke_fill='black')
a.output.parent.mkdir(parents=True,exist_ok=True);im.save(a.output)
