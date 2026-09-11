#!/usr/bin/env python3
"""Evaluate route localization on held-out patrol frames without occupancy/labels."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from navmem3d.navigation.route_localization import localize_to_visual_route

p=argparse.ArgumentParser();p.add_argument('--graph',type=Path,required=True);p.add_argument('--render-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples-per-edge',type=int,default=2);a=p.parse_args()
g=json.loads(a.graph.read_text());nodes=g['nodes'];results=[]
for edge in g['edges']:
    ids=edge['frame_indices'][1:-1]
    if not ids: continue
    picks=sorted(set(ids[round(i*(len(ids)-1)/max(1,a.samples_per_edge-1))] for i in range(a.samples_per_edge)))
    expected_start=int(nodes[[x['node_id'] for x in nodes].index(edge['from_node'])]['source_frame_index']);expected_end=int(nodes[[x['node_id'] for x in nodes].index(edge['to_node'])]['source_frame_index'])
    expected_node = edge['from_node'] if True else edge['to_node']
    for idx in picks:
        result=localize_to_visual_route(a.graph,a.render_dir/f'frame_{idx:04d}.png')
        predicted=result.get('best',{}).get('node_id')
        # Correct if localizer chooses either endpoint of the edge; this tests route association, not sub-frame metric pose.
        correct=predicted in {edge['from_node'],edge['to_node']}
        results.append({'query_frame_index':idx,'edge_id':edge['edge_id'],'expected_edge_endpoints':[edge['from_node'],edge['to_node']],'predicted_node_id':predicted,'status':result['status'],'correct_edge_association':correct,'best':result.get('best')})
valid=[x for x in results if x['status']=='localized']; correct=[x for x in valid if x['correct_edge_association']]
out={'schema_version':'0.1','protocol':'held_out_patrol_frame_to_visual_route_node; occupancy_and_world_pose_not_read','query_count':len(results),'localized_count':len(valid),'localized_rate':len(valid)/max(1,len(results)),'correct_edge_association_count':len(correct),'correct_edge_association_rate':len(correct)/max(1,len(valid)),'results':results}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:out[k] for k in out if k.endswith('count') or k.endswith('rate')}))
