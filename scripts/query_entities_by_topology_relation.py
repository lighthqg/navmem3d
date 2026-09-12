#!/usr/bin/env python3
"""First-version keyword + topology-relation entity query.

Relation is evaluated only on observed patrol topology, not on a hidden simulator map.
"""
from __future__ import annotations
import argparse,heapq,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--entities',type=Path,required=True);p.add_argument('--topology',type=Path,required=True);p.add_argument('--term',required=True);p.add_argument('--reference-term',required=True);p.add_argument('--relation',choices=['near'],default='near');p.add_argument('--near-m',type=float,default=4.0);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
e=json.loads(a.entities.read_text());g=json.loads(a.topology.read_text());nodes={x['node_id'] for x in g['nodes']};adj={x:[] for x in nodes}
for x in g['edges']:
 adj[x['from_node']].append((x['to_node'],float(x['cost'])));adj[x['to_node']].append((x['from_node'],float(x['cost'])))
def match(x,q):return q.lower() in x.get('category_id','').lower() or any(q.lower() in str(y).lower() for y in x.get('labels',[]))
def shortest(s):
 q=[(0.,s)];d={s:0.}
 while q:
  c,u=heapq.heappop(q)
  if c!=d[u]:continue
  for v,w in adj[u]:
   z=c+w
   if z<d.get(v,math.inf):d[v]=z;heapq.heappush(q,(z,v))
 return d
targets=[x for x in e['entities'] if x.get('navigation_eligible') and match(x,a.term) and x.get('topology_node_id') in nodes]
refs=[x for x in e['entities'] if x.get('navigation_eligible') and match(x,a.reference_term) and x.get('topology_node_id') in nodes]
if not targets:raise ValueError(f'no target matches: {a.term}')
if not refs:raise ValueError(f'no reference matches: {a.reference_term}')
kept=[]
for x in targets:
 d=shortest(x['topology_node_id'])
 best=min(((d.get(r['topology_node_id'],math.inf),r) for r in refs),key=lambda z:z[0])
 if best[0]<=a.near_m:kept.append({**x,'relation_evidence':{'relation':a.relation,'reference_entity_id':best[1]['entity_id'],'topology_distance_m':best[0]}})
out={'schema_version':'1.0','protocol':'keyword_then_observed_topology_relation','query':{'term':a.term,'reference_term':a.reference_term,'relation':a.relation,'near_m':a.near_m},'hidden_occupancy_consumed':False,'entities':kept,'statistics':{'target_candidates':len(targets),'reference_candidates':len(refs),'relation_matches':len(kept)}}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out['statistics'],ensure_ascii=False))
