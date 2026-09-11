from __future__ import annotations
from pathlib import Path
import json


def build_visual_route_graph(patrol_path: str | Path, render_dir: str | Path, output_path: str | Path, *, keyframe_stride: int = 15) -> dict[str, object]:
    """Create a route graph from an observed patrol without exporting world coordinates.

    Nodes retain only visual references, sequence order and relative odometry
    between sampled frames. The caller may provide a patrol generated in a
    simulator, but no occupancy, labels, absolute poses or world coordinates
    are emitted into the system-visible graph.
    """
    if keyframe_stride < 1:
        raise ValueError("keyframe_stride must be positive")
    patrol = json.loads(Path(patrol_path).read_text(encoding="utf-8"))
    frames = patrol.get("frames", [])
    if len(frames) < 2:
        raise ValueError("patrol needs at least two frames")
    root = Path(render_dir)
    # Select one high-texture reference inside each temporal section. Fixed
    # sampling can select a blank wall at a turn, where visual repeat fails.
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("visual route graph construction requires OpenCV") from exc
    nominal = list(range(0, len(frames), keyframe_stride))
    if nominal[-1] != len(frames) - 1:
        nominal.append(len(frames) - 1)
    half_window = max(1, keyframe_stride // 3)
    indices = []
    quality_by_index = {}
    for point in nominal:
        choices = range(max(0, point - half_window), min(len(frames), point + half_window + 1))
        best_index = None; best_score = -1
        for index in choices:
            image = root / f"{frames[index]['frame_id']}.png"
            if not image.exists():
                raise FileNotFoundError(image)
            gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            keypoints = cv2.SIFT_create(nfeatures=600).detect(gray, None)
            score = len(keypoints)
            if score > best_score:
                best_index, best_score = index, score
        if best_index is not None and (not indices or best_index > indices[-1]):
            indices.append(best_index); quality_by_index[best_index] = best_score
    if indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)
        image = root / f"{frames[-1]['frame_id']}.png"
        quality_by_index[len(frames) - 1] = len(cv2.SIFT_create(nfeatures=600).detect(cv2.imread(str(image), cv2.IMREAD_GRAYSCALE), None))
    nodes = []
    for number, index in enumerate(indices):
        frame = frames[index]
        image = root / f"{frame['frame_id']}.png"
        nodes.append({
            "node_id": f"node_{number:03d}",
            "source_frame_id": frame["frame_id"],
            "source_frame_index": index,
            "timestamp_s": frame["timestamp"],
            "reference_rgb_uri": str(image.resolve()),
            "visual_feature_count": quality_by_index.get(index, 0),
        })
    edges = []
    for number, (left, right) in enumerate(zip(nodes, nodes[1:])):
        first, last = left["source_frame_index"], right["source_frame_index"]
        edges.append({
            "edge_id": f"edge_{number:03d}", "from_node": left["node_id"], "to_node": right["node_id"],
            "frame_indices": list(range(first, last + 1)), "frame_count": last - first + 1,
            "duration_s": right["timestamp_s"] - left["timestamp_s"],
            "traversal": "observed_forward", "reverse_available": True,
        })
    result = {
        "schema_version": "0.1", "world_role": "B_route_support", "system_visible": True,
        "source": {"type": "single_patrol_rgb_sequence", "frame_count": len(frames), "absolute_pose_exported": False, "occupancy_exported": False},
        "keyframe_policy": "temporal coverage with local maximum SIFT feature count", "nodes": nodes, "edges": edges,
    }
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def plan_visual_route(graph_path: str | Path, *, start_node_id: str, target_node_id: str, output_path: str | Path | None = None) -> dict[str, object]:
    graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    nodes = {x["node_id"]: x for x in graph["nodes"]}
    if start_node_id not in nodes or target_node_id not in nodes:
        raise ValueError("start or target node is not in route graph")
    # First version graph is a connected patrol chain. Shortest route is found
    # by BFS so later loop-closure edges can be added without changing API.
    adj = {n: [] for n in nodes}
    for e in graph["edges"]:
        adj[e["from_node"]].append((e["to_node"], e["edge_id"], "forward"))
        if e.get("reverse_available"):
            adj[e["to_node"]].append((e["from_node"], e["edge_id"], "reverse"))
    queue=[start_node_id]; prev={start_node_id:None}
    while queue:
        current=queue.pop(0)
        if current==target_node_id: break
        for nxt,eid,direction in adj[current]:
            if nxt not in prev: prev[nxt]=(current,eid,direction); queue.append(nxt)
    if target_node_id not in prev: raise RuntimeError("target is unreachable in visual route graph")
    steps=[]; current=target_node_id
    while prev[current] is not None:
        parent,eid,direction=prev[current]; steps.append({"edge_id":eid,"from_node":parent,"to_node":current,"direction":direction});current=parent
    steps.reverse()
    result={"schema_version":"0.1","start_node_id":start_node_id,"target_node_id":target_node_id,"node_sequence":[start_node_id]+[x["to_node"] for x in steps],"edge_sequence":steps,"status":"ready_for_visual_repeat"}
    if output_path:
        p=Path(output_path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    return result
