from __future__ import annotations
from pathlib import Path
import json


def localize_to_visual_route(graph_path: str | Path, image_path: str | Path, *, ratio_test: float = 0.72) -> dict[str, object]:
    """Localize an RGB observation to a route node using local visual evidence.

    The result is a route hypothesis rather than a metric world pose. SIFT is
    deliberately used after global retrieval was found too scene-generic in
    the Marble restaurant. An empty/weak match is an explicit failure state.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("route localization requires OpenCV") from exc
    graph = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    query = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if query is None:
        raise FileNotFoundError(image_path)
    sift = cv2.SIFT_create(nfeatures=1800)
    kp_q, desc_q = sift.detectAndCompute(query, None)
    if desc_q is None:
        return {"status": "no_query_features", "candidates": []}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    candidates = []
    for node in graph["nodes"]:
        reference = cv2.imread(node["reference_rgb_uri"], cv2.IMREAD_GRAYSCALE)
        kp_r, desc_r = sift.detectAndCompute(reference, None)
        if desc_r is None:
            continue
        pairs = matcher.knnMatch(desc_q, desc_r, k=2)
        good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < ratio_test * pair[1].distance]
        # A homography inlier count rejects repeated texture matches.
        inliers = 0
        if len(good) >= 4:
            import numpy as np
            p_q = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            p_r = np.float32([kp_r[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(p_q, p_r, cv2.RANSAC, 5.0)
            inliers = int(mask.sum()) if mask is not None else 0
        candidates.append({"node_id": node["node_id"], "source_frame_id": node["source_frame_id"], "ratio_matches": len(good), "homography_inliers": inliers})
    candidates.sort(key=lambda x: (x["homography_inliers"], x["ratio_matches"]), reverse=True)
    best = candidates[0] if candidates else None
    return {"status": "localized" if best and best["homography_inliers"] >= 6 else "insufficient_visual_evidence", "best": best, "candidates": candidates[:5]}
