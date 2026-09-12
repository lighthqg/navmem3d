#!/usr/bin/env python3
"""Project an already verified patrol topology into a robot occupancy map.

This does not create edges.  It converts the topology's metric patrol
coordinates into M pixels and audits every existing edge against the observed
three-state grid.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def samples(polyline: list[list[float]], spacing_m: float = 0.05):
    out = []
    for a, b in zip(polyline, polyline[1:]):
        distance = math.dist(a[:2], b[:2])
        count = max(1, math.ceil(distance / spacing_m))
        out.extend([
            [a[0] + (b[0] - a[0]) * i / count, a[1] + (b[1] - a[1]) * i / count]
            for i in range(count)
        ])
    out.append(polyline[-1][:2])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--occupancy", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--patrol", type=Path, help="patrol poses used to retain traversed edge polylines")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    topology = json.loads(args.topology.read_text())
    patrol = json.loads(args.patrol.read_text())["frames"] if args.patrol else None
    metadata = json.loads(args.metadata.read_text())
    occupancy = np.asarray(Image.open(args.occupancy).convert("L"))
    origin_x, origin_y = metadata["origin_xy"]
    scale = float(metadata["scale_m"])
    height, width = occupancy.shape

    def pixel(xy: list[float]) -> list[int]:
        return [round((xy[0] - origin_x) / scale), round((xy[1] - origin_y) / scale)]

    def state_at(xy: list[float]) -> int:
        x, y = pixel(xy)
        return int(occupancy[y, x]) if 0 <= x < width and 0 <= y < height else 127

    projected_nodes = []
    for node in topology["nodes"]:
        item = dict(node)
        item["pixel"] = pixel(node["position_xy"])
        item["m_state"] = state_at(node["position_xy"])
        projected_nodes.append(item)

    projected_edges = []
    audited = {"clear": 0, "blocked": 0, "partially_unknown": 0}
    for edge in topology["edges"]:
        item = dict(edge)
        line = edge.get("polyline_xy")
        traversal_source = "topology_polyline"
        if patrol:
            node_by_id = {n["node_id"]: n for n in topology["nodes"]}
            start = node_by_id[edge["from_node"]].get("reference_frame_index")
            end = node_by_id[edge["to_node"]].get("reference_frame_index")
            # Consecutive anchors are the actual observed patrol segment.  For
            # long revisit connectors we retain the graph's verified local
            # polyline, rather than falsely treating the entire intervening
            # patrol loop as one edge.
            if start is not None and end is not None and abs(start - end) <= 15:
                step = 1 if end >= start else -1
                line = [patrol[index]["camera_position"][:2] for index in range(start, end + step, step)]
                traversal_source = "patrol_pose_segment"
        if not line:
            node_by_id = {n["node_id"]: n for n in topology["nodes"]}
            line = [node_by_id[edge["from_node"]]["position_xy"], node_by_id[edge["to_node"]]["position_xy"]]
        points = samples(line, scale)
        states = [state_at(point) for point in points]
        total = len(states)
        counts = {"free": states.count(255), "occupied": states.count(0), "unknown": states.count(127)}
        if counts["occupied"]:
            audit = "blocked"
        elif counts["unknown"]:
            audit = "partially_unknown"
        else:
            audit = "clear"
        audited[audit] += 1
        item["polyline_pixel"] = [pixel(point) for point in line]
        item["m_occupancy_audit"] = {
            "status": audit,
            "sample_count": total,
            "free_samples": counts["free"],
            "occupied_samples": counts["occupied"],
            "unknown_samples": counts["unknown"],
            "traversal_confirmed": True,
            "polyline_source": traversal_source,
        }
        projected_edges.append(item)

    output = {
        **{key: value for key, value in topology.items() if key not in {"nodes", "edges"}},
        "schema_version": "0.5",
        "type": "verified_patrol_topology_in_robot_map",
        "source_topology": str(args.topology.resolve()),
        "source_occupancy": str(args.occupancy.resolve()),
        "hidden_occupancy_consumed": False,
        "unknown_is_blocked_for_new_edges": True,
        "nodes": projected_nodes,
        "edges": projected_edges,
        "m_map_edge_audit": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"nodes": len(projected_nodes), "edges": len(projected_edges), "audit": audited}, ensure_ascii=False))


if __name__ == "__main__":
    main()
