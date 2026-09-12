#!/usr/bin/env python3
"""Bind Marble entities to patrol topology through verified B-view → A-RGB retrieval.

This intentionally avoids treating a coarse B→A Sim(3) as metric ground truth.
Each binding is a visual route candidate and requires current-view confirmation.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--entities',type=Path,required=True);p.add_argument('--view-matches',type=Path,required=True);p.add_argument('--topology',type=Path,required=True);p.add_argument('--patrol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
entities=json.loads(a.entities.read_text());matches=json.loads(a.view_matches.read_text());topo=json.loads(a.topology.read_text())
match_by_view={int(x['b_index']):x for x in matches['matches']};nodes=topo['nodes']
patrol={x['frame_id']:x for x in json.loads(a.patrol.read_text())['frames']}
def node_for(frame_id):
    # Retrieval can match any of the 300 patrol frames; bind it to the nearest
    # sparse topology keyframe rather than requiring an exact frame-ID match.
    frame=patrol.get(frame_id)
    if not frame:return None
    x,y=frame['camera_position'][:2]
    return min(nodes,key=lambda n:(n['position_xy'][0]-x)**2+(n['position_xy'][1]-y)**2)
out=[];unbound=0
for e in entities['entities']:
 candidates=[]
 for obs in e.get('observations',[]):
  vi=int(obs['view_index']);m=match_by_view.get(vi)
  if not m:continue
  n=node_for(m['a_frame_id'])
  if n:candidates.append((float(m['similarity']),obs,m,n))
 if not candidates:
  out.append({**e,'topology_node_id':None,'navigation_eligible':False,'binding_type':'no_visual_patrol_match'});unbound+=1;continue
 score,obs,m,n=max(candidates,key=lambda x:x[0])
 multi_view=len({int(x['view_index']) for x in e.get('observations',[])})
 confidence=score*min(1.0,0.55+0.15*multi_view)
 out.append({**e,'source_frame_id':m['a_frame_id'],'topology_node_id':n['node_id'],'topology_reference_frame_id':n['reference_frame_id'],'binding_type':'marble_view_to_patrol_rgb_retrieval','binding_confidence':confidence,'visual_match_similarity':score,'marble_view_index':int(obs['view_index']),'multi_view_observation_count':multi_view,'navigation_eligible':True,'execution_requirement':'confirm target in current robot RGB before declaring arrival'})
doc={'schema_version':'1.0','protocol':'marble_entity_to_patrol_topology_via_visual_retrieval','source_entities':str(a.entities.resolve()),'source_view_matches':str(a.view_matches.resolve()),'source_topology':str(a.topology.resolve()),'source_patrol':str(a.patrol.resolve()),'metric_b_to_a_registration_used':False,'hidden_occupancy_consumed':False,'entities':out,'statistics':{'total':len(out),'bound':len(out)-unbound,'unbound':unbound}}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n');print(json.dumps(doc['statistics'],ensure_ascii=False))
