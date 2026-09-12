#!/usr/bin/env python3
"""Validate the source-only offline closed-loop artifacts without evaluator truth."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--m-dir',type=Path,required=True);p.add_argument('--b-dir',type=Path,required=True);p.add_argument('--patrol',type=Path,required=True);p.add_argument('--route',type=Path);a=p.parse_args()
errors=[];checks={}
def load(path):
 if not path.exists():errors.append(f'missing:{path}');return {}
 return json.loads(path.read_text())
m=load(a.m_dir/'occupancy_metadata.json');top=load(a.m_dir/'patrol_topology.json');route=load(a.route or a.m_dir/'route_to_sofa.json');ent=load(a.b_dir/'entities_to_m.json')
occ_path=a.m_dir/'observed_occupancy.png'
if occ_path.exists():
 grid=np.asarray(Image.open(occ_path).convert('L'));checks['occupancy_states_valid']=bool(np.isin(grid,[0,127,255]).all());checks['occupancy_counts']={str(v):int((grid==v).sum()) for v in [0,127,255]}
else:errors.append('missing:observed_occupancy.png')
for key in ['hidden_occupancy_consumed','simulator_navmesh_consumed','simulator_collider_consumed','semantic_labels_consumed']:
 checks[key]=m.get(key) is False
 if not checks[key]:errors.append(f'forbidden_input_flag:{key}')
nodes={x.get('node_id') for x in top.get('nodes',[])};edges=top.get('edges',[])
checks['topology_nonempty']=bool(nodes and edges)
checks['topology_edges_reference_known_nodes']=all(x.get('from_node') in nodes and x.get('to_node') in nodes for x in edges)
checks['topology_edges_are_patrol_evidence']=all(x.get('evidence') in {'consecutive_patrol_segment','closed_patrol_return'} and x.get('undirected') is True for x in edges)
if not checks['topology_edges_are_patrol_evidence']:errors.append('topology_contains_non_patrol_edge')
checks['topology_patrol_validation']=bool(top.get('patrol_free_validation',{}).get('passed'))
if not checks['topology_patrol_validation']:errors.append('patrol_validation_failed')
checks['semantic_entities_nonempty']=len(ent.get('entities',[]))>0
checks['semantic_metric_registration_disabled']=ent.get('metric_b_to_a_registration_used') is False
checks['all_entity_bindings_valid']=all(x.get('topology_node_id') in nodes and len(x.get('candidate_topology_nodes',[]))>0 for x in ent.get('entities',[]))
checks['route_nodes_valid']=all(x in nodes for x in route.get('node_sequence',[]))
checks['route_target_valid']=route.get('target_topology_node_id') in nodes
checks['route_requires_current_rgb_confirmation']='confirm' in str(route.get('arrival_policy','')).lower() and ('current rgb' in str(route.get('arrival_policy','')).lower() or 'current-rgb' in str(route.get('arrival_policy','')).lower())
if not checks['route_requires_current_rgb_confirmation']:errors.append('route_missing_current_rgb_confirmation_policy')
checks['route_candidate_routes_nonempty']=len(route.get('target_candidate_routes',[]))>0
if not checks['route_candidate_routes_nonempty']:errors.append('route_missing_visual_candidate_routes')
# Recheck every patrol pose lands in free M cells, independent of builder metadata.
if occ_path.exists() and m and a.patrol.exists():
 frames=load(a.patrol).get('frames',[]);origin=np.asarray(m['origin_xy']);scale=float(m['scale_m']);poses=np.asarray([x['camera_position'][:2] for x in frames]);pix=np.floor((poses-origin)/scale).astype(int);inside=(pix[:,0]>=0)&(pix[:,0]<grid.shape[1])&(pix[:,1]>=0)&(pix[:,1]<grid.shape[0]);checks['patrol_pose_free_rate']=float((grid[pix[inside,1],pix[inside,0]]==255).mean()) if inside.any() else 0.0
 if checks['patrol_pose_free_rate']<1.0:errors.append('patrol_pose_not_all_free')
report={'schema_version':'1.0','protocol':'source_only_offline_closed_loop_self_check','passed':not errors,'errors':errors,'checks':checks,'truth_inputs_read':False}
out=a.m_dir/'closed_loop_self_check.json';out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'passed':report['passed'],'errors':errors},ensure_ascii=False))
raise SystemExit(0 if report['passed'] else 1)
