from __future__ import annotations

from pathlib import Path
import json


def estimate_pnp_registration(
    virtual_views_path: str | Path,
    patrol_path: str | Path,
    patrol_render_dir: str | Path,
    gaussian_asset: str | Path,
    output_path: str | Path,
    *,
    stride: int = 5,
    ratio_test: float = 0.72,
    min_inliers: int = 12,
    max_transform_translation_m: float = 20.0,
) -> dict[str, object]:
    """Validate B→A correspondence using B Gaussian 3D points and A RGB PnP.

    For each B virtual image, SIFT matches against sampled A patrol RGB. A
    matched B pixel supplies its dominant Gaussian center; the known A camera
    pose is used only to score PnP inliers and pose consistency. This avoids
    A labels and is intentionally conservative: no accepted pair means no
    visual geometric registration is claimed.
    """
    try:
        import cv2
        import numpy as np
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("PnP registration requires OpenCV, NumPy and plyfile") from exc
    if stride <= 0:
        raise ValueError("stride must be positive")
    views_doc = json.loads(Path(virtual_views_path).read_text(encoding="utf-8"))
    patrol_doc = json.loads(Path(patrol_path).read_text(encoding="utf-8"))
    view_root = Path(virtual_views_path).parent
    render_root = Path(patrol_render_dir)
    sampled_frames = [frame for frame in patrol_doc.get("frames", [])[::stride] if (render_root / f"{frame['frame_id']}.png").exists()]
    if not sampled_frames:
        raise ValueError("no sampled patrol RGB frames exist")
    vertex = PlyData.read(str(gaussian_asset)).elements[0]
    positions = np.stack([vertex[name] for name in ("x", "y", "z")], axis=1).astype(np.float32)
    detector = cv2.SIFT_create(nfeatures=1800)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    a_cache = []
    for frame in sampled_frames:
        image = cv2.imread(str(render_root / f"{frame['frame_id']}.png"), cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = detector.detectAndCompute(image, None)
        if descriptors is not None and len(keypoints) >= 8:
            a_cache.append((frame, keypoints, descriptors))
    if not a_cache:
        raise RuntimeError("no usable A patrol descriptors")
    accepted, attempted = [], []
    for view in views_doc.get("views", []):
        image = cv2.imread(str(view_root / view["rgb_uri"]), cv2.IMREAD_GRAYSCALE)
        keypoints_b, descriptors_b = detector.detectAndCompute(image, None)
        ids_uri = view.get("top_contributor_ids_uri")
        if descriptors_b is None or not ids_uri:
            continue
        ids = np.load(view_root / ids_uri, allow_pickle=False)[..., 0]
        best = None
        for frame, keypoints_a, descriptors_a in a_cache:
            raw = matcher.knnMatch(descriptors_b, descriptors_a, k=2)
            good = [pair[0] for pair in raw if len(pair) == 2 and pair[0].distance < ratio_test * pair[1].distance]
            if len(good) < min_inliers:
                continue
            points_3d, points_2d = [], []
            for match in good:
                x, y = keypoints_b[match.queryIdx].pt
                xi, yi = int(round(x)), int(round(y))
                if not (0 <= xi < ids.shape[1] and 0 <= yi < ids.shape[0]):
                    continue
                gaussian_id = int(ids[yi, xi])
                if gaussian_id == int(np.iinfo(np.uint32).max) or gaussian_id >= len(positions):
                    continue
                points_3d.append(positions[gaussian_id])
                points_2d.append(keypoints_a[match.trainIdx].pt)
            if len(points_3d) < min_inliers:
                continue
            focal = 0.5 * image.shape[1]  # 90 degree horizontal FOV
            camera = np.array([[focal, 0, image.shape[1] / 2], [0, focal, image.shape[0] / 2], [0, 0, 1]], dtype=np.float64)
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.asarray(points_3d, dtype=np.float32), np.asarray(points_2d, dtype=np.float32),
                camera, None, iterationsCount=200, reprojectionError=5.0, confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
            inlier_count = 0 if inliers is None else int(len(inliers))
            candidate = {"b_view_id": view["view_id"], "a_frame_id": frame["frame_id"], "raw_ratio_matches": len(good), "pnp_correspondences": len(points_3d), "pnp_success": bool(success), "pnp_inliers": inlier_count}
            if success:
                # OpenCV PnP returns B-world → camera coordinates. The A patrol
                # pose yields A-world → the same OpenCV camera convention.
                eye = np.asarray(frame["camera_position"], dtype=float)
                look_at = np.asarray(frame["look_at"], dtype=float)
                up = np.asarray(frame.get("up", [0.0, 0.0, 1.0]), dtype=float)
                forward = look_at - eye
                forward /= np.linalg.norm(forward)
                right = np.cross(forward, up)
                right /= np.linalg.norm(right)
                camera_up = np.cross(right, forward)
                a_world_to_camera = np.eye(4)
                a_world_to_camera[:3, :3] = np.stack([right, -camera_up, forward], axis=0)
                a_world_to_camera[:3, 3] = -a_world_to_camera[:3, :3] @ eye
                b_world_to_camera = np.eye(4)
                b_world_to_camera[:3, :3] = cv2.Rodrigues(rvec)[0]
                b_world_to_camera[:3, 3] = tvec.reshape(3)
                transform = np.linalg.inv(a_world_to_camera) @ b_world_to_camera
                translation_norm = float(np.linalg.norm(transform[:3, 3]))
                candidate["transform_b_to_a_rigid"] = transform.reshape(-1).tolist()
                candidate["transform_translation_norm_m"] = translation_norm
                candidate["transform_plausible"] = bool(
                    np.isfinite(transform).all() and translation_norm <= max_transform_translation_m
                )
            attempted.append(candidate)
            if success and candidate.get("transform_plausible", False) and inlier_count >= min_inliers and (best is None or inlier_count > best["pnp_inliers"]):
                best = candidate
        if best is not None:
            accepted.append(best)
    # A repeated texture can match multiple B views to one A frame. Keep the
    # strongest result per A pose when judging independent evidence.
    best_by_a = {}
    for item in accepted:
        previous = best_by_a.get(item["a_frame_id"])
        if previous is None or item["pnp_inliers"] > previous["pnp_inliers"]:
            best_by_a[item["a_frame_id"]] = item
    independent = list(best_by_a.values())
    if len(independent) >= 3:
        status = "independent_geometric_pairs_available_for_global_registration"
    elif independent:
        status = "geometric_evidence_insufficient_distinct_a_poses"
    else:
        status = "no_geometric_matches_meet_threshold"
    result = {
        "schema_version": "0.2",
        "method": "sift_ratio_match_plus_gaussian_center_pnp_ransac",
        "status": status,
        "source_virtual_views": str(Path(virtual_views_path).resolve()),
        "source_patrol": str(Path(patrol_path).resolve()),
        "source_gaussian_asset": str(Path(gaussian_asset).resolve()),
        "stride": stride,
        "ratio_test": ratio_test,
        "min_inliers": min_inliers,
        "max_transform_translation_m": max_transform_translation_m,
        "accepted_pair_count": len(accepted),
        "accepted_pairs": accepted,
        "independent_a_pose_count": len(independent),
        "independent_a_pose_pairs": independent,
        "attempted_pair_count": len(attempted),
        "top_attempts": sorted(attempted, key=lambda x: (x["pnp_inliers"], x["pnp_correspondences"]), reverse=True)[:30],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
