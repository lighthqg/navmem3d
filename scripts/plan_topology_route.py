#!/usr/bin/env python3
"""Plan an approximate shortest route on the undirected topology graph.

The input entity index is Marble/B semantic output.  This script is the
semantic handoff: entity -> observed patrol frame -> topology node -> route.
"""
from __future__ import annotations
import argparse, heapq, json, math
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--topology',type=Path,required=True);ap.add_argument('--entities',type=Path,required=True);ap.add_argument('--term',required=True);ap.add_argument('--start-node',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--allow-target-at-start',action='store_true');a=ap.parse_args()
    topo=json.loads(a.topology.read_text()); ent=json.loads(a.entities.read_text())
    nodes={n['node_id']:n for n in topo['nodes']}; keyframes=topo['topology_keyframes']
    if a.start_node not in nodes: raise ValueError(f'unknown start topology node: {a.start_node}')
    matches=[e for e in ent['entities'] if any(a.term.lower() in str(label).lower() for label in e.get('labels',[])) or a.term.lower() in str(e.get('category_id','')).lower()]
    if not matches: raise ValueError(f'no entity matches term: {a.term}')
    # Keep every semantic candidate.  A category query such as “餐桌” may
    # legitimately refer to several instances; the planner must rank routes,
    # rather than silently selecting the strongest visual match.
    legacy={x['node_id']:x for x in topo.get('visual_anchors_legacy',[])}
    def handoff(e):
        frame_id=e.get('source_frame_id') or e.get('selected_source_frame_id')
        if e.get('topology_node_id') in nodes:
            return next(x for x in keyframes if x['topology_node_id']==e['topology_node_id'])
        if not frame_id:
            old=legacy.get(e.get('selected_route_node_id'),{}); frame_id=old.get('source_frame_id')
        return next((x for x in keyframes if x['source_frame_id']==frame_id),None)
    def shortest(goal):
        adj={n:[] for n in nodes}
        for edge in topo['edges']:
            u,v=edge['from_node'],edge['to_node']; c=float(edge.get('cost',edge.get('length_m',1))); adj[u].append((v,c,edge['edge_id']));adj[v].append((u,c,edge['edge_id']))
        q=[(0,a.start_node)]; ds={a.start_node:0.0}; prev={}
        while q:
            d,u=heapq.heappop(q)
            if d!=ds[u]:continue
            if u==goal:break
            for v,c,eid in adj[u]:
                nd=d+c
                if nd<ds.get(v,math.inf):ds[v]=nd;prev[v]=(u,eid);heapq.heappush(q,(nd,v))
        if goal not in ds:return None
        seq=[]; es=[];u=goal
        while u!=a.start_node:p,eid=prev[u];seq.append(u);es.append(eid);u=p
        seq.append(a.start_node);return list(reversed(seq)),list(reversed(es)),ds[goal]
    ranked=[]
    for target in matches:
        k=handoff(target)
        if k is None:continue
        route=shortest(k['topology_node_id'])
        if route is None:continue
        seq,edges,cost=route; ranked.append({'target_entity':target,'target_topology_node_id':k['topology_node_id'],'node_sequence':seq,'edge_sequence':edges,'cost_m':cost})
    if len(ranked)>1 and not a.allow_target_at_start:
        nonzero=[x for x in ranked if x['cost_m']>1e-6]
        if nonzero: ranked=nonzero
    if not ranked: raise ValueError('matching entities have no topology handoff or route')
    ranked.sort(key=lambda x:(not bool(x['target_entity'].get('navigation_eligible')),x['cost_m'],-x['target_entity'].get('local_visual_inliers',0)))
    best=ranked[0];target=best['target_entity'];goal=best['target_topology_node_id'];seq=best['node_sequence'];edges=best['edge_sequence']
    frame_by_node={n['node_id']:n.get('reference_frame_id') for n in topo['nodes']}
    out={'schema_version':'0.2','protocol':'marble_entity_to_undirected_topology_route','query_term':a.term,'start_topology_node_id':a.start_node,'target_entity':target,'target_topology_node_id':goal,'node_sequence':seq,'edge_sequence':edges,'reference_frame_sequence':[frame_by_node.get(n) for n in seq],'cost_m':best['cost_m'],'candidate_routes':ranked,'handoff':{'semantic_world':'Marble World B','planning_world':'observed topology graph','execution_world':'robot/current RGB + controller'},'arrival_policy':'route arrival is provisional; confirm target with current RGB'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'target_entity':target['entity_id'],'target_node':goal,'nodes':len(seq),'cost_m':best['cost_m'],'candidate_routes':len(ranked)},ensure_ascii=False))
if __name__=='__main__':main()
