#!/usr/bin/env python3
"""Draw a node-only plan view of the observed visual route.

This is an evaluation/debug visualization.  It reads camera positions from
the patrol record, but does not export occupancy, labels, or obstacles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, required=True)
    ap.add_argument("--patrol", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--metadata-output", type=Path)
    ap.add_argument("--route", type=Path, help="optional planned route JSON to highlight")
    ap.add_argument("--size", type=int, default=1200)
    args = ap.parse_args()

    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    patrol = json.loads(args.patrol.read_text(encoding="utf-8"))
    frames = patrol["frames"]
    by_index = {int(f["frame_id"].split("_")[-1]): f for f in frames}
    nodes = graph["nodes"]
    route_nodes = set()
    route_sequence = []
    if args.route:
        route_doc = json.loads(args.route.read_text(encoding="utf-8"))
        route_sequence = route_doc.get("route", route_doc).get("node_sequence", [])
        route_nodes = set(route_sequence)
    positions = {}
    for node in nodes:
        index = int(node["source_frame_index"])
        frame = by_index[index]
        x, y, *_ = frame["camera_position"]
        positions[node["node_id"]] = (float(x), float(y))
    if not positions:
        raise ValueError("graph contains no nodes")

    margin = 90
    width = height = args.size
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    span_x = max(max(xs) - min(xs), 1e-6)
    span_y = max(max(ys) - min(ys), 1e-6)
    scale = min((width - 2 * margin) / span_x, (height - 2 * margin) / span_y)

    def project(p: tuple[float, float]) -> tuple[int, int]:
        # Image y grows down; preserve the usual plan-view convention.
        return (round(margin + (p[0] - min(xs)) * scale),
                round(height - margin - (p[1] - min(ys)) * scale))

    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    # Axes are intentionally unlabeled as world coordinates are not a runtime
    # contract; the small legend makes the provenance explicit.
    draw.text((24, 20), "Visual Route Graph · node connectivity", fill=(25, 35, 45), font=font)
    draw.text((24, 38), "plan view from patrol camera positions; obstacles omitted", fill=(95, 105, 115), font=font)

    # Draw observed graph edges first so nodes remain legible.
    for edge in graph.get("edges", []):
        a = project(positions[edge["from_node"]])
        b = project(positions[edge["to_node"]])
        on_route = edge["from_node"] in route_nodes and edge["to_node"] in route_nodes
        draw.line((a, b), fill=(224, 112, 45) if on_route else (72, 111, 145), width=7 if on_route else 4)
        # A small arrow indicates the observed forward traversal.
        ax, ay = a; bx, by = b
        t = 0.62
        px, py = ax + t * (bx - ax), ay + t * (by - ay)
        draw.polygon([(round(px), round(py)),
                      (round(px - 10 - (by - ay) * .03), round(py - 4 + (bx - ax) * .03)),
                      (round(px - 10 + (by - ay) * .03), round(py + 4 - (bx - ax) * .03))], fill=(72, 111, 145))

    for number, node in enumerate(nodes):
        point = project(positions[node["node_id"]])
        radius = 11
        if node["node_id"] in route_nodes:
            color = (238, 127, 36)
        else:
            color = (217, 91, 67) if number in (0, len(nodes) - 1) else (35, 126, 94)
        draw.ellipse((point[0] - radius, point[1] - radius,
                      point[0] + radius, point[1] + radius), fill=color, outline=(255, 255, 255), width=3)
        label = node["node_id"].replace("node_", "")
        draw.text((point[0] + 14, point[1] - 7), label, fill=(25, 35, 45), font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    metadata = {
        "schema_version": "0.1",
        "type": "visual_route_node_plan",
        "source_graph": str(args.graph.resolve()),
        "source_patrol": str(args.patrol.resolve()),
        "node_count": len(nodes),
        "edge_count": len(graph.get("edges", [])),
        "coordinates": "patrol camera XY, visualization only",
        "occupancy_rendered": False,
        "obstacles_rendered": False,
        "highlighted_route_node_sequence": route_sequence,
        "nodes": [{"node_id": n["node_id"], "x": positions[n["node_id"]][0], "y": positions[n["node_id"]][1]} for n in nodes],
    }
    metadata_path = args.metadata_output or args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "nodes": len(nodes), "edges": len(graph.get("edges", []))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
