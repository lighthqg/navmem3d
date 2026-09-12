#!/usr/bin/env python3
"""Plan routes from a Marble semantic entity to a robot-observed topology.

A B-to-M visual match is a *candidate distribution*, never a metric pose.  The
first candidate is the nominal route; alternatives remain available for
relocalisation or current-RGB confirmation instead of being silently replaced
by a shorter but visually weaker route.
"""
from __future__ import annotations
import argparse, heapq, json, math
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--topology', type=Path, required=True)
    ap.add_argument('--entities', type=Path, required=True)
    ap.add_argument('--term', required=True)
    ap.add_argument('--start-node', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--allow-target-at-start', action='store_true')
    args = ap.parse_args()
    topo = json.loads(args.topology.read_text())
    index = json.loads(args.entities.read_text())
    nodes = {n['node_id']: n for n in topo['nodes']}
    if args.start_node not in nodes:
        raise ValueError(f'unknown start topology node: {args.start_node}')
    adj = {node_id: [] for node_id in nodes}
    for edge in topo['edges']:
        u, v = edge['from_node'], edge['to_node']
        cost = float(edge.get('cost', edge.get('length_m', 1.0)))
        adj[u].append((v, cost, edge['edge_id']))
        adj[v].append((u, cost, edge['edge_id']))

    def shortest(goal: str):
        queue = [(0.0, args.start_node)]
        distances, previous = {args.start_node: 0.0}, {}
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            if node == goal:
                break
            for neighbour, weight, edge_id in adj[node]:
                candidate = distance + weight
                if candidate < distances.get(neighbour, math.inf):
                    distances[neighbour] = candidate
                    previous[neighbour] = (node, edge_id)
                    heapq.heappush(queue, (candidate, neighbour))
        if goal not in distances:
            return None
        route_nodes, route_edges, node = [], [], goal
        while node != args.start_node:
            parent, edge_id = previous[node]
            route_nodes.append(node); route_edges.append(edge_id); node = parent
        route_nodes.append(args.start_node)
        return list(reversed(route_nodes)), list(reversed(route_edges)), distances[goal]

    term = args.term.lower()
    matches = [entity for entity in index['entities'] if (
        term in str(entity.get('category_id', '')).lower() or
        any(term in str(label).lower() for label in entity.get('labels', []))
    )]
    if not matches:
        raise ValueError(f'no entity matches term: {args.term}')

    entity_routes = []
    for entity in matches:
        candidates = list(entity.get('candidate_topology_nodes') or [])
        if not candidates and entity.get('topology_node_id'):
            candidates = [{'topology_node_id': entity['topology_node_id'], 'similarity': None,
                           'source_frame_id': entity.get('source_frame_id')}]
        seen = set()
        routes = []
        for rank, candidate in enumerate(candidates, start=1):
            target = candidate.get('topology_node_id')
            if target not in nodes or target in seen:
                continue
            seen.add(target)
            planned = shortest(target)
            if planned is None:
                continue
            route_nodes, route_edges, cost = planned
            routes.append({
                'candidate_rank': rank,
                'topology_node_id': target,
                'source_frame_id': candidate.get('source_frame_id'),
                'visual_similarity': candidate.get('similarity'),
                'node_sequence': route_nodes,
                'edge_sequence': route_edges,
                'cost_m': cost,
            })
        if routes:
            # Preserve retrieval confidence first.  Distance only breaks equal-confidence ties.
            routes.sort(key=lambda route: (route['candidate_rank'], route['cost_m']))
            entity_routes.append({'target_entity': entity, 'candidate_routes': routes,
                                  'primary_route': routes[0]})
    if len(entity_routes) > 1 and not args.allow_target_at_start:
        nonzero = [item for item in entity_routes if item['primary_route']['cost_m'] > 1e-6]
        if nonzero:
            entity_routes = nonzero
    if not entity_routes:
        raise ValueError('matching entities have no usable topology candidates')
    # Different semantic instances are still ranked by their primary visual candidate, then route cost.
    entity_routes.sort(key=lambda item: (
        not bool(item['target_entity'].get('navigation_eligible')),
        item['primary_route']['candidate_rank'], item['primary_route']['cost_m'],
        -item['target_entity'].get('local_visual_inliers', 0),
    ))
    selected = entity_routes[0]
    primary = selected['primary_route']
    reference_frames = {node['node_id']: node.get('reference_frame_id') for node in topo['nodes']}
    out = {
        'schema_version': '0.3',
        'protocol': 'marble_entity_visual_candidates_to_undirected_topology_route',
        'query_term': args.term,
        'start_topology_node_id': args.start_node,
        'target_entity': selected['target_entity'],
        'target_topology_node_id': primary['topology_node_id'],
        'node_sequence': primary['node_sequence'],
        'edge_sequence': primary['edge_sequence'],
        'reference_frame_sequence': [reference_frames[node] for node in primary['node_sequence']],
        'cost_m': primary['cost_m'],
        'target_candidate_routes': selected['candidate_routes'],
        'entity_candidate_routes': entity_routes,
        'handoff': {
            'semantic_world': 'Marble World B',
            'planning_world': 'robot-observed undirected topology graph M',
            'execution_world': 'robot/current RGB + controller',
            'metric_b_to_a_transform_used': False,
        },
        'arrival_policy': 'primary route is a visual hypothesis; test alternatives only after current-RGB confirmation or relocalisation failure',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'target_entity': selected['target_entity']['entity_id'],
                      'target_node': primary['topology_node_id'], 'nodes': len(primary['node_sequence']),
                      'cost_m': primary['cost_m'], 'target_candidates': len(selected['candidate_routes']),
                      'entity_candidates': len(entity_routes)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
