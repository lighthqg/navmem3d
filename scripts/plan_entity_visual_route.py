#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from navmem3d.navigation.visual_route import plan_visual_route
p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--graph',type=Path,required=True);p.add_argument('--start-node',required=True);p.add_argument('--term',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.loads(a.binding.read_text());term=a.term.lower();matches=[x for x in d['entities'] if term in ' '.join(x.get('labels',[])+[x.get('category_id','')]).lower()]
if not matches: raise SystemExit('no bound entity matches term')
# Prefer binding evidence, then greater cross-view support is already represented by entity index upstream.
target=max(matches,key=lambda x:x['binding_confidence']);route=plan_visual_route(a.graph,start_node_id=a.start_node,target_node_id=target['selected_route_node_id'])
out={'schema_version':'0.1','query_term':a.term,'start_node_id':a.start_node,'target_entity':target,'route':route,'arrival_policy':'use current RGB to confirm target at selected route node; do not use simulator truth'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'target':target['entity_id'],'node':target['selected_route_node_id'],'steps':len(route['edge_sequence'])},ensure_ascii=False))
