#!/usr/bin/env python3
"""Keyword + observed-topology relation query without hidden scene geometry."""
from __future__ import annotations
import argparse, heapq, json, math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--entities', type=Path, required=True); ap.add_argument('--topology', type=Path, required=True)
ap.add_argument('--term', required=True); ap.add_argument('--reference-term', required=True)
ap.add_argument('--relation', choices=['near'], default='near'); ap.add_argument('--near-m', type=float, default=4.0)
ap.add_argument('--output', type=Path, required=True)
a = ap.parse_args()
entities = json.loads(a.entities.read_text()); graph = json.loads(a.topology.read_text())
nodes = {node['node_id'] for node in graph['nodes']}; adjacency = {node: [] for node in nodes}
for edge in graph['edges']:
    adjacency[edge['from_node']].append((edge['to_node'], float(edge['cost'])))
    adjacency[edge['to_node']].append((edge['from_node'], float(edge['cost'])))

def match(entity, term):
    term = term.lower()
    return term in entity.get('category_id', '').lower() or any(term in str(label).lower() for label in entity.get('labels', []))

def candidates(entity):
    raw = entity.get('candidate_topology_nodes') or [{'topology_node_id': entity.get('topology_node_id'), 'similarity': None}]
    return [candidate for candidate in raw if candidate.get('topology_node_id') in nodes]

def dijkstra(source):
    queue, distances = [(0.0, source)], {source: 0.0}
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]: continue
        for neighbour, weight in adjacency[node]:
            next_distance = distance + weight
            if next_distance < distances.get(neighbour, math.inf):
                distances[neighbour] = next_distance; heapq.heappush(queue, (next_distance, neighbour))
    return distances

targets = [e for e in entities['entities'] if e.get('navigation_eligible') and match(e, a.term) and candidates(e)]
references = [e for e in entities['entities'] if e.get('navigation_eligible') and match(e, a.reference_term) and candidates(e)]
if not targets: raise ValueError(f'no target matches: {a.term}')
if not references: raise ValueError(f'no reference matches: {a.reference_term}')
kept = []
for target in targets:
    evidence = []
    for tc in candidates(target):
        distances = dijkstra(tc['topology_node_id'])
        for reference in references:
            for rc in candidates(reference):
                distance = distances.get(rc['topology_node_id'], math.inf)
                evidence.append({'target_topology_node_id': tc['topology_node_id'], 'target_similarity': tc.get('similarity'),
                                 'reference_entity_id': reference['entity_id'], 'reference_topology_node_id': rc['topology_node_id'],
                                 'reference_similarity': rc.get('similarity'), 'topology_distance_m': distance})
    evidence.sort(key=lambda item: (item['topology_distance_m'], -(item['target_similarity'] or -1), -(item['reference_similarity'] or -1)))
    # Do not relabel target's primary pose: relation is evidence across its visual hypotheses.
    if evidence and evidence[0]['topology_distance_m'] <= a.near_m:
        kept.append({**target, 'relation_evidence': {'relation': a.relation, 'best_candidate_pair': evidence[0],
                                                      'candidate_pairs_within_threshold': [item for item in evidence if item['topology_distance_m'] <= a.near_m]}})
out = {'schema_version': '1.1', 'protocol': 'keyword_then_observed_topology_relation',
       'query': {'term': a.term, 'reference_term': a.reference_term, 'relation': a.relation, 'near_m': a.near_m},
       'hidden_occupancy_consumed': False, 'metric_b_to_a_transform_used': False, 'entities': kept,
       'statistics': {'target_candidates': len(targets), 'reference_candidates': len(references), 'relation_matches': len(kept)}}
a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(out['statistics'], ensure_ascii=False))
