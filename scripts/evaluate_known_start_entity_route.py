#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from navmem3d.navigation.visual_route import plan_visual_route
from navmem3d.navigation.route_localization import localize_to_visual_route
p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--graph',type=Path,required=True);p.add_argument('--render-dir',type=Path,required=True);p.add_argument('--start-node',required=True);p.add_argument('--term',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=json.loads(a.binding.read_text());g=json.loads(a.graph.read_text());term=a.term.lower();cands=[x for x in b['entities'] if x.get('navigation_eligible') and term in ' '.join(x.get('labels',[])+[x.get('category_id','')]).lower()]
if not cands: raise SystemExit('no navigation-eligible entity matches query')
nonlocal_targets=[x for x in cands if x['selected_route_node_id'] != a.start_node]
target=max(nonlocal_targets or cands,key=lambda x:(x['local_visual_inliers'],x['binding_confidence']));route=plan_visual_route(a.graph,start_node_id=a.start_node,target_node_id=target['selected_route_node_id']);nodes={x['node_id']:x for x in g['nodes']};replay=[]
for expected in route['node_sequence']:
 n=nodes[expected];idx=n['source_frame_index'];loc=localize_to_visual_route(a.graph,a.render_dir/f'frame_{idx:04d}.png');replay.append({'expected_node_id':expected,'frame_index':idx,'predicted_node_id':loc.get('best',{}).get('node_id'),'status':loc['status'],'inliers':loc.get('best',{}).get('homography_inliers',0),'correct':loc.get('best',{}).get('node_id')==expected})
localized=[x for x in replay if x['status']=='localized'];out={'schema_version':'0.1','protocol':'known_start_visual_repeat_replay_no_occupancy_or_truth_input','query_term':a.term,'start_node_id':a.start_node,'target_entity':target,'route':route,'replay':replay,'node_localization_rate':len(localized)/len(replay),'exact_node_rate':sum(x['correct'] for x in localized)/max(1,len(localized)),'arrival_confirmation_policy':'at target node, run current-RGB object detection/query confirmation; route binding alone is insufficient'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'target':target['entity_id'],'nodes':len(replay),'localized_rate':out['node_localization_rate'],'exact_node_rate':out['exact_node_rate']},ensure_ascii=False))
