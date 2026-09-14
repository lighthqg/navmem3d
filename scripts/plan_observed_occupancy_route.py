#!/usr/bin/env python3
"""Plan a collision-free route on the robot-generated M occupancy grid.

The grid is the only geometry input: free is traversable; occupied and unknown
are blocked.  Topology nodes provide visual/keyframe anchors, while A* supplies
shortcuts that are demonstrably free in M rather than replaying patrol history.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from navmem3d.navigation.grid_runtime import GridTransform, OccupancyGrid, astar_path


def raster_line(a, b):
    x0,y0=a; x1,y1=b; dx,sx=abs(x1-x0),1 if x0<x1 else -1; dy,sy=-abs(y1-y0),1 if y0<y1 else -1; err=dx+dy
    while True:
        yield x0,y0
        if (x0,y0)==(x1,y1): return
        twice=2*err
        if twice>=dy: err+=dy; x0+=sx
        if twice<=dx: err+=dx; y0+=sy


def simplify(path, blocked):
    """Line-of-sight simplify without crossing blocked grid cells."""
    if len(path) < 3: return path
    result=[path[0]]; i=0
    while i < len(path)-1:
        chosen=i+1
        for j in range(i+2,len(path)):
            line=list(raster_line(path[i],path[j]))
            if any(cell in blocked for cell in line): break
            chosen=j
        result.append(path[chosen]); i=chosen
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--occupancy',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--topology',type=Path,required=True);p.add_argument('--entities',type=Path,required=True)
    p.add_argument('--term',required=True);p.add_argument('--entity-id',help='optional exact semantic instance id');p.add_argument('--start-node',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--robot-radius-m',type=float,default=.0,help='inflate occupied/unknown cells by this radius')
    a=p.parse_args()
    meta=json.loads(a.metadata.read_text()); topo=json.loads(a.topology.read_text()); index=json.loads(a.entities.read_text())
    transform=GridTransform((float(meta['scale_m']),0,float(meta['origin_xy'][0]),0,float(meta['scale_m']),float(meta['origin_xy'][1]),0,0,1))
    grid=OccupancyGrid.from_png(a.occupancy,transform); nodes={n['node_id']:n for n in topo['nodes']}
    if a.start_node not in nodes: raise ValueError(f'unknown start node: {a.start_node}')
    radius=math.ceil(a.robot_radius_m/float(meta['scale_m'])); blocked=grid.inflated_blocked(radius)
    def node_pixel(node_id):
        x,y=nodes[node_id]['position_xy']; col,row=transform.to_pixel(x,y); return round(col),round(row)
    start=node_pixel(a.start_node)
    if start in blocked: raise ValueError('start node is not free after inflation')
    term=a.term.lower(); matches=[e for e in index['entities'] if term in str(e.get('category_id','')).lower() or any(term in str(v).lower() for v in e.get('labels',[]))]
    if a.entity_id: matches=[e for e in matches if e.get('entity_id') == a.entity_id]
    if not matches: raise ValueError(f'no entities match {a.term}')
    entity_routes=[]
    for entity in matches:
        candidates=entity.get('candidate_topology_nodes') or [{'topology_node_id':entity.get('topology_node_id'),'similarity':None,'source_frame_id':entity.get('source_frame_id')}]
        routes=[];seen=set()
        for rank,candidate in enumerate(candidates,1):
            target=candidate.get('topology_node_id')
            if target not in nodes or target in seen: continue
            seen.add(target); goal=node_pixel(target)
            raw=astar_path(grid,start,goal,blocked=blocked)
            if raw is None: continue
            compact=simplify(raw,blocked)
            grid_length=sum(math.hypot(x1-x0,y1-y0) for (x0,y0),(x1,y1) in zip(raw,raw[1:]))*float(meta['scale_m'])
            routes.append({'candidate_rank':rank,'topology_node_id':target,'source_frame_id':candidate.get('source_frame_id'),'visual_similarity':candidate.get('similarity'),'cost_m':grid_length,'grid_cell_count':len(raw),'path_xy':[list(transform.to_world(col,row)) for col,row in compact]})
        if routes: entity_routes.append({'target_entity':entity,'candidate_routes':routes,'primary_route':routes[0]})
    if not entity_routes: raise ValueError('no target candidate is reachable through M free space')
    entity_routes.sort(key=lambda item:(not bool(item['target_entity'].get('navigation_eligible')),item['primary_route']['candidate_rank'],item['primary_route']['cost_m']))
    selected=entity_routes[0]; primary=selected['primary_route']
    out={'schema_version':'1.0','protocol':'marble_visual_candidates_to_M_occupancy_astar_route','query_term':a.term,'start_topology_node_id':a.start_node,'target_entity':selected['target_entity'],'target_topology_node_id':primary['topology_node_id'],'cost_m':primary['cost_m'],'grid_path_xy':primary['path_xy'],'grid_cell_count':primary['grid_cell_count'],'target_candidate_routes':selected['candidate_routes'],'entity_candidate_routes':entity_routes,'handoff':{'semantic_world':'Marble World B','planning_world':'robot-generated M occupancy grid','execution_world':'robot/current RGB + local controller','hidden_truth_used':False,'unknown_is_blocked':True,'robot_radius_m':a.robot_radius_m},'arrival_policy':'primary route is a visual hypothesis; confirm target in current RGB before declaring success'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'target_entity':selected['target_entity']['entity_id'],'target_node':primary['topology_node_id'],'cost_m':primary['cost_m'],'grid_cells':primary['grid_cell_count'],'waypoints':len(primary['path_xy'])},ensure_ascii=False))
if __name__=='__main__': main()
