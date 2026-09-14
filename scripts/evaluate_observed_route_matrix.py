#!/usr/bin/env python3
"""Evaluate every topology-start × semantic-candidate route in an M map.

This is an algorithm audit, not an evaluator-truth comparison.  It uses only
M's generated three-state occupancy, topology anchors, and B→M candidate index.
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from navmem3d.navigation.grid_runtime import GridTransform, OccupancyGrid, astar_path

p=argparse.ArgumentParser()
p.add_argument('--occupancy',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True)
p.add_argument('--topology',type=Path,required=True);p.add_argument('--entities',type=Path,required=True)
p.add_argument('--robot-radius-m',type=float,default=.0);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
meta=json.loads(a.metadata.read_text());top=json.loads(a.topology.read_text());entities=json.loads(a.entities.read_text())['entities']
transform=GridTransform((float(meta['scale_m']),0,float(meta['origin_xy'][0]),0,float(meta['scale_m']),float(meta['origin_xy'][1]),0,0,1))
grid=OccupancyGrid.from_png(a.occupancy,transform);radius=math.ceil(a.robot_radius_m/float(meta['scale_m']));blocked=grid.inflated_blocked(radius)
nodes={n['node_id']:n for n in top['nodes']}
def pixel(node):
 col,row=transform.to_pixel(*nodes[node]['position_xy']);return round(col),round(row)
starts=[node_id for node_id in nodes if pixel(node_id) not in blocked]
records=[]
path_cache={}
for entity in entities:
 for rank,candidate in enumerate(entity.get('candidate_topology_nodes',[]),1):
  target=candidate.get('topology_node_id')
  if target not in nodes or pixel(target) in blocked: continue
  for start in starts:
   cache_key=(start,target)
   if cache_key not in path_cache:
    path_cache[cache_key]=astar_path(grid,pixel(start),pixel(target),blocked=blocked)
   path=path_cache[cache_key]
   cost=None if path is None else sum(math.hypot(x1-x0,y1-y0) for (x0,y0),(x1,y1) in zip(path,path[1:]))*float(meta['scale_m'])
   records.append({'start_node_id':start,'entity_id':entity['entity_id'],'category_id':entity.get('category_id'),'candidate_rank':rank,'target_node_id':target,'reachable':path is not None,'cost_m':cost})
by_entity=defaultdict(list)
for record in records: by_entity[record['entity_id']].append(record)
entity_summary=[]
for entity in entities:
 rows=by_entity[entity['entity_id']]; reachable=[r for r in rows if r['reachable']]
 entity_summary.append({'entity_id':entity['entity_id'],'category_id':entity.get('category_id'),'tested_pairs':len(rows),'reachable_pairs':len(reachable),'reachability':len(reachable)/max(1,len(rows)),'mean_cost_m':sum(r['cost_m'] for r in reachable)/max(1,len(reachable))})
summary={'schema_version':'1.0','protocol':'M_occupancy_route_matrix_source_only','hidden_truth_used':False,'unknown_blocked':True,'robot_radius_m':a.robot_radius_m,'topology_start_count':len(starts),'entity_count':len(entities),'candidate_route_pairs':len(records),'unique_start_target_pairs':len(path_cache),'reachable_pairs':sum(r['reachable'] for r in records),'reachability':sum(r['reachable'] for r in records)/max(1,len(records)),'entity_summary':entity_summary,'routes':records}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:summary[k] for k in ['topology_start_count','entity_count','candidate_route_pairs','unique_start_target_pairs','reachable_pairs','reachability']},ensure_ascii=False))
