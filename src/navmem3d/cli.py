from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Sequence

from navmem3d.capture.keyframes import select_pose_coverage, select_uniform
from navmem3d.capture.inspect import inspect_sequence
from navmem3d.schemas import CaptureSequence, SchemaError
from navmem3d.world.package import inspect_world
from navmem3d.world.gaussians import inspect_gaussian_asset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="navmem3d")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-world", help="validate and summarize a world manifest")
    inspect_parser.add_argument("manifest", type=Path)

    inspect_gaussians_parser = subparsers.add_parser(
        "inspect-gaussians", help="validate and summarize a Gaussian asset"
    )
    inspect_gaussians_parser.add_argument("asset", type=Path)
    inspect_gaussians_parser.add_argument("--max-samples", type=int, default=200_000)

    interiorgs_parser = subparsers.add_parser(
        "register-interiorgs", help="validate an InteriorGS scene and create a world-A manifest"
    )
    interiorgs_parser.add_argument("--scene", type=Path, required=True)
    interiorgs_parser.add_argument("--output", type=Path, required=True)

    patrol_parser = subparsers.add_parser(
        "build-interiorgs-patrol",
        help="build a deterministic occupancy-constrained robot patrol in world A",
    )
    patrol_parser.add_argument("--scene", type=Path, required=True)
    patrol_parser.add_argument("--output", type=Path, required=True)
    patrol_parser.add_argument("--frames", type=int, default=300)
    patrol_parser.add_argument("--waypoints", type=int, default=8)
    patrol_parser.add_argument("--fps", type=float, default=10.0)
    patrol_parser.add_argument("--robot-radius", type=float, default=0.20)
    patrol_parser.add_argument("--sensor-height", type=float, default=1.25)

    inspect_sequence_parser = subparsers.add_parser(
        "inspect-sequence", help="validate a capture sequence and its file references"
    )
    inspect_sequence_parser.add_argument("sequence", type=Path)

    select_parser = subparsers.add_parser("select-frames", help="select frames from a capture sequence")
    select_parser.add_argument("sequence", type=Path)
    select_parser.add_argument("--count", type=int, required=True)
    select_parser.add_argument(
        "--strategy", choices=("uniform", "pose-coverage"), default="uniform"
    )
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.add_argument(
        "--copy-rgb-to", type=Path, help="copy selected RGB images into an upload-ready directory"
    )

    habitat_parser = subparsers.add_parser(
        "capture-habitat", help="render an RGB-D patrol with an optional Habitat-Sim install"
    )
    habitat_parser.add_argument("--scene", type=Path, required=True)
    habitat_parser.add_argument("--output", type=Path, required=True)
    habitat_parser.add_argument("--sequence-id", required=True)
    habitat_parser.add_argument("--frames", type=int, default=120)
    habitat_parser.add_argument("--width", type=int, default=640)
    habitat_parser.add_argument("--height", type=int, default=480)
    habitat_parser.add_argument("--hfov", type=float, default=90.0)
    habitat_parser.add_argument("--sensor-height", type=float, default=1.25)
    habitat_parser.add_argument("--seed", type=int, default=7)

    glb_parser = subparsers.add_parser(
        "capture-glb", help="render a deterministic RGB-D room patrol from a GLB asset"
    )
    glb_parser.add_argument("--scene", type=Path, required=True)
    glb_parser.add_argument("--output", type=Path, required=True)
    glb_parser.add_argument("--sequence-id", required=True)
    glb_parser.add_argument("--frames", type=int, default=120)
    glb_parser.add_argument("--width", type=int, default=640)
    glb_parser.add_argument("--height", type=int, default=480)
    glb_parser.add_argument("--hfov", type=float, default=90.0)
    glb_parser.add_argument("--sensor-height", type=float, default=1.25)
    glb_parser.add_argument("--path-radius-ratio", type=float, default=0.28)

    video_parser = subparsers.add_parser(
        "make-video", help="encode capture RGB frames under a duration and size limit"
    )
    video_parser.add_argument("--sequence", type=Path, required=True)
    video_parser.add_argument("--output", type=Path, required=True)
    video_parser.add_argument("--fps", type=float, default=10.0)
    video_parser.add_argument("--max-duration", type=float, default=30.0)
    video_parser.add_argument("--max-size-mb", type=float, default=100.0)

    render_parser = subparsers.add_parser(
        "render-splat-views", help="render virtual RGB views and Gaussian center-ID maps"
    )
    render_parser.add_argument("--asset", type=Path, required=True)
    render_parser.add_argument("--output", type=Path, required=True)
    render_parser.add_argument("--views", type=int, default=8)
    render_parser.add_argument("--width", type=int, default=640)
    render_parser.add_argument("--height", type=int, default=480)
    render_parser.add_argument("--hfov", type=float, default=90.0)
    render_parser.add_argument("--orbit-radius-ratio", type=float, default=0.18)
    render_parser.add_argument("--point-radius-px", type=int, default=1)

    compile_parser = subparsers.add_parser(
        "compile-masks", help="lift instance masks into Gaussian entity memberships"
    )
    compile_parser.add_argument("--annotations", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument("--min-view-votes", type=int, default=1)

    query_parser = subparsers.add_parser(
        "query-index", help="query compiled entities by exact labels, attributes and relation"
    )
    query_parser.add_argument("--index", type=Path, required=True)
    query_parser.add_argument("--term", action="append", required=True)
    query_parser.add_argument("--attribute", action="append", default=[])
    query_parser.add_argument("--predicate")
    query_parser.add_argument("--reference-term", action="append", default=[])
    query_parser.add_argument(
        "--compact", action="store_true", help="print entity IDs and labels without full geometry"
    )

    preflight_parser = subparsers.add_parser(
        "preflight-input", help="check a multi-view set before spending a world-builder call"
    )
    preflight_parser.add_argument("--selection", type=Path, required=True)
    preflight_parser.add_argument("--output", type=Path)

    sam_parser = subparsers.add_parser(
        "generate-sam2-masks", help="generate class-agnostic instance masks with SAM 2.1"
    )
    sam_parser.add_argument("--image", type=Path, required=True)
    sam_parser.add_argument("--output", type=Path, required=True)
    sam_parser.add_argument("--checkpoint", type=Path, required=True)
    sam_parser.add_argument("--points-per-side", type=int, default=24)
    sam_parser.add_argument("--max-masks", type=int, default=128)
    sam_parser.add_argument("--device", default="cuda")

    lift_set_parser = subparsers.add_parser(
        "lift-mask-set", help="lift a SAM mask manifest into unlabeled Gaussian proposals"
    )
    lift_set_parser.add_argument("--masks", type=Path, required=True)
    lift_set_parser.add_argument("--gaussian-ids", type=Path, required=True)
    lift_set_parser.add_argument("--gaussian-weights", type=Path)
    lift_set_parser.add_argument("--output", type=Path, required=True)
    lift_set_parser.add_argument("--min-gaussians", type=int, default=32)
    lift_set_parser.add_argument("--min-weight", type=float, default=0.0)
    lift_set_parser.add_argument("--min-pixel-hits", type=int, default=1)

    gsplat_parser = subparsers.add_parser(
        "render-gsplat-views", help="render true anisotropic Gaussian views with CUDA gsplat"
    )
    gsplat_parser.add_argument("--asset", type=Path, required=True)
    gsplat_parser.add_argument("--output", type=Path, required=True)
    gsplat_parser.add_argument("--views", type=int, default=4)
    gsplat_parser.add_argument("--width", type=int, default=640)
    gsplat_parser.add_argument("--height", type=int, default=480)
    gsplat_parser.add_argument("--hfov", type=float, default=90.0)
    gsplat_parser.add_argument("--device", default="cuda")
    gsplat_parser.add_argument("--top-contributors", type=int, default=0)
    gsplat_parser.add_argument(
        "--export-depth",
        action="store_true",
        help="write dense expected-depth maps from the Gaussian rasterizer",
    )
    gsplat_parser.add_argument(
        "--trajectory", type=Path, help="render exact camera poses from a patrol JSON"
    )

    fusion_parser = subparsers.add_parser(
        "fuse-proposals", help="fuse cross-view proposals by shared Gaussian membership"
    )
    fusion_parser.add_argument("--proposal-index", type=Path, action="append", required=True)
    fusion_parser.add_argument("--output", type=Path, required=True)
    fusion_parser.add_argument("--minimum-containment", type=float, default=0.20)
    fusion_parser.add_argument("--minimum-jaccard", type=float, default=0.08)

    geometry_parser = subparsers.add_parser(
        "derive-geometry", help="derive entity bounds and relations from Gaussian memberships"
    )
    geometry_parser.add_argument("--entities", type=Path, required=True)
    geometry_parser.add_argument("--asset", type=Path, required=True)
    geometry_parser.add_argument("--output", type=Path, required=True)
    geometry_parser.add_argument("--near-threshold-ratio", type=float, default=0.06)

    filter_parser = subparsers.add_parser(
        "filter-entities", help="separate searchable object candidates from broad regions"
    )
    filter_parser.add_argument("--geometry-index", type=Path, required=True)
    filter_parser.add_argument("--output", type=Path, required=True)
    filter_parser.add_argument("--structural-axis-ratio", type=float, default=0.45)
    filter_parser.add_argument("--single-view-max-axis-ratio", type=float, default=0.30)

    crop_parser = subparsers.add_parser(
        "export-entity-crops", help="export one mask-aware representative crop per entity"
    )
    crop_parser.add_argument("--entity-index", type=Path, required=True)
    crop_parser.add_argument("--output", type=Path, required=True)
    crop_parser.add_argument("--padding-ratio", type=float, default=0.15)

    label_parser = subparsers.add_parser(
        "label-entity-crops", help="label representative crops with a fixed CLIP vocabulary"
    )
    label_parser.add_argument("--entity-index", type=Path, required=True)
    label_parser.add_argument("--crop-manifest", type=Path, required=True)
    label_parser.add_argument("--vocabulary", type=Path, required=True)
    label_parser.add_argument("--output", type=Path, required=True)
    label_parser.add_argument("--model-name", default="ViT-B-32")
    label_parser.add_argument("--pretrained", default="openai")
    label_parser.add_argument("--device", default="cuda")
    label_parser.add_argument("--batch-size", type=int, default=32)

    guided_parser = subparsers.add_parser(
        "generate-guided-masks", help="detect fixed-vocabulary objects and refine boxes with SAM2"
    )
    guided_parser.add_argument("--image", type=Path, action="append", required=True)
    guided_parser.add_argument("--output", type=Path, required=True)
    guided_parser.add_argument("--vocabulary", type=Path, required=True)
    guided_parser.add_argument("--sam-checkpoint", type=Path, required=True)
    guided_parser.add_argument("--detector-model", default="IDEA-Research/grounding-dino-tiny")
    guided_parser.add_argument("--box-threshold", type=float, default=0.25)
    guided_parser.add_argument("--text-threshold", type=float, default=0.20)
    guided_parser.add_argument("--device", default="cuda")

    dedup_parser = subparsers.add_parser(
        "deduplicate-entities", help="suppress same-category entities with overlapping 3D membership"
    )
    dedup_parser.add_argument("--entity-index", type=Path, required=True)
    dedup_parser.add_argument("--output", type=Path, required=True)
    dedup_parser.add_argument("--minimum-containment", type=float, default=0.75)
    dedup_parser.add_argument("--minimum-jaccard", type=float, default=0.45)

    goals_parser = subparsers.add_parser(
        "build-observation-goals", help="bind entities to source virtual-camera goal poses"
    )
    goals_parser.add_argument("--entity-index", type=Path, required=True)
    goals_parser.add_argument("--virtual-views", type=Path, required=True)
    goals_parser.add_argument("--output", type=Path, required=True)

    query_goals_parser = subparsers.add_parser(
        "query-navigation-goals", help="resolve an entity query into observation-pose candidates"
    )
    query_goals_parser.add_argument("--entity-index", type=Path, required=True)
    query_goals_parser.add_argument("--goal-index", type=Path, required=True)
    query_goals_parser.add_argument("--term", action="append", required=True)
    query_goals_parser.add_argument("--attribute", action="append", default=[])
    query_goals_parser.add_argument("--predicate")
    query_goals_parser.add_argument("--reference-term", action="append", default=[])
    query_goals_parser.add_argument("--max-goals-per-entity", type=int, default=3)

    route_parser = subparsers.add_parser("build-visual-route", help="build a system-visible keyframe route graph from a patrol")
    route_parser.add_argument("--patrol", type=Path, required=True)
    route_parser.add_argument("--renders", type=Path, required=True)
    route_parser.add_argument("--output", type=Path, required=True)
    route_parser.add_argument("--keyframe-stride", type=int, default=15)
    route_plan_parser = subparsers.add_parser("plan-visual-route", help="plan between known nodes in a visual route graph")
    route_plan_parser.add_argument("--graph", type=Path, required=True)
    route_plan_parser.add_argument("--start-node", required=True)
    route_plan_parser.add_argument("--target-node", required=True)
    route_plan_parser.add_argument("--output", type=Path, required=True)

    register_parser = subparsers.add_parser("register-goals-b-to-a", help="apply explicit B→A registration to navigation goals")
    register_parser.add_argument("--goal-index", type=Path, required=True)
    register_parser.add_argument("--output", type=Path, required=True)
    register_parser.add_argument("--transform", type=float, nargs=16)
    register_parser.add_argument("--registration-status")

    estimate_parser = subparsers.add_parser("estimate-b-to-a", help="estimate coarse Marble B→InteriorGS A registration")
    estimate_parser.add_argument("--b-splat", type=Path, required=True)
    estimate_parser.add_argument("--a-scene", type=Path, required=True)
    estimate_parser.add_argument("--output", type=Path, required=True)

    visual_register_parser = subparsers.add_parser(
        "estimate-b-to-a-visual", help="estimate provisional B→A Sim(3) from B views and A patrol RGB"
    )
    visual_register_parser.add_argument("--virtual-views", type=Path, required=True)
    visual_register_parser.add_argument("--patrol", type=Path, required=True)
    visual_register_parser.add_argument("--patrol-renders", type=Path, required=True)
    visual_register_parser.add_argument("--output", type=Path, required=True)
    visual_register_parser.add_argument("--stride", type=int, default=5)
    visual_register_parser.add_argument("--minimum-similarity", type=float, default=0.60)
    visual_register_parser.add_argument("--device", default="cuda")

    pnp_register_parser = subparsers.add_parser(
        "validate-b-to-a-pnp", help="seek local B-Gaussian to A-image PnP/RANSAC evidence"
    )
    pnp_register_parser.add_argument("--virtual-views", type=Path, required=True)
    pnp_register_parser.add_argument("--patrol", type=Path, required=True)
    pnp_register_parser.add_argument("--patrol-renders", type=Path, required=True)
    pnp_register_parser.add_argument("--asset", type=Path, required=True)
    pnp_register_parser.add_argument("--output", type=Path, required=True)
    pnp_register_parser.add_argument("--stride", type=int, default=5)
    pnp_register_parser.add_argument("--ratio-test", type=float, default=0.72)
    pnp_register_parser.add_argument("--min-inliers", type=int, default=12)
    pnp_register_parser.add_argument("--max-transform-translation-m", type=float, default=20.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect-world":
            print(json.dumps(inspect_world(args.manifest), ensure_ascii=False, indent=2))
            return 0
        if args.command == "inspect-gaussians":
            summary = inspect_gaussian_asset(args.asset, max_samples=args.max_samples)
            print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "register-interiorgs":
            from navmem3d.world.interiorgs import register_interiorgs_scene

            manifest, summary = register_interiorgs_scene(args.scene, args.output)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "world_id": manifest.world_id,
                        "scene": summary,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "build-interiorgs-patrol":
            from navmem3d.navigation.patrol import build_interiorgs_patrol

            result = build_interiorgs_patrol(
                args.scene,
                args.output,
                frame_count=args.frames,
                waypoint_count=args.waypoints,
                fps=args.fps,
                robot_radius_m=args.robot_radius,
                sensor_height_m=args.sensor_height,
            )
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "frames": len(result["frames"]),
                        "duration_s": result["duration_s"],
                        "route_nodes": result["route_nodes"],
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "inspect-sequence":
            print(json.dumps(inspect_sequence(args.sequence), ensure_ascii=False, indent=2))
            return 0
        if args.command == "select-frames":
            sequence = CaptureSequence.load(args.sequence)
            if args.strategy == "uniform":
                selected = select_uniform(sequence.frames, args.count)
            else:
                selected = select_pose_coverage(sequence.frames, args.count)
            result = {
                "sequence_id": sequence.sequence_id,
                "strategy": args.strategy,
                "requested_count": args.count,
                "selected_frame_ids": [frame.frame_id for frame in selected],
                "frames": [frame.to_dict() for frame in selected],
            }
            if args.copy_rgb_to:
                args.copy_rgb_to.mkdir(parents=True, exist_ok=True)
                copied = []
                for order, frame in enumerate(selected):
                    source = (args.sequence.parent / frame.rgb_uri).resolve()
                    if not source.exists():
                        raise FileNotFoundError(source)
                    destination = args.copy_rgb_to / f"{order:02d}_{frame.frame_id}{source.suffix.lower()}"
                    shutil.copy2(source, destination)
                    copied.append(str(destination))
                result["copied_rgb_uris"] = copied
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps({"output": str(args.output), "selected": len(selected)}))
            return 0
        if args.command == "capture-habitat":
            from navmem3d.capture.habitat import HabitatCaptureConfig, capture_sequence

            sequence = capture_sequence(
                HabitatCaptureConfig(
                    scene=args.scene,
                    output_dir=args.output,
                    sequence_id=args.sequence_id,
                    frame_count=args.frames,
                    width=args.width,
                    height=args.height,
                    horizontal_fov_deg=args.hfov,
                    sensor_height_m=args.sensor_height,
                    seed=args.seed,
                )
            )
            print(json.dumps({"output": str(args.output), "frames": len(sequence.frames)}))
            return 0
        if args.command == "capture-glb":
            from navmem3d.capture.glb import GlbCaptureConfig, capture_sequence

            sequence = capture_sequence(
                GlbCaptureConfig(
                    scene=args.scene,
                    output_dir=args.output,
                    sequence_id=args.sequence_id,
                    frame_count=args.frames,
                    width=args.width,
                    height=args.height,
                    horizontal_fov_deg=args.hfov,
                    sensor_height_m=args.sensor_height,
                    path_radius_ratio=args.path_radius_ratio,
                )
            )
            print(json.dumps({"output": str(args.output), "frames": len(sequence.frames)}))
            return 0
        if args.command == "make-video":
            from navmem3d.capture.video import make_video

            result = make_video(
                args.sequence,
                args.output,
                fps=args.fps,
                max_duration_s=args.max_duration,
                max_size_mb=args.max_size_mb,
            )
            print(json.dumps(result, ensure_ascii=False))
            return 0
        if args.command == "render-splat-views":
            from navmem3d.world.render import VirtualViewConfig, render_virtual_views

            result = render_virtual_views(
                VirtualViewConfig(
                    asset=args.asset,
                    output_dir=args.output,
                    view_count=args.views,
                    width=args.width,
                    height=args.height,
                    horizontal_fov_deg=args.hfov,
                    orbit_radius_ratio=args.orbit_radius_ratio,
                    point_radius_px=args.point_radius_px,
                )
            )
            print(json.dumps({"output": str(args.output), "views": len(result["views"])}))
            return 0
        if args.command == "compile-masks":
            from navmem3d.semantic.compiler import compile_mask_annotations

            result = compile_mask_annotations(
                args.annotations, args.output, min_view_votes=args.min_view_votes
            )
            print(json.dumps({"output": str(args.output), "entities": len(result["entities"])}))
            return 0
        if args.command == "query-index":
            from navmem3d.semantic.query import query_entity_index

            result = query_entity_index(
                args.index,
                terms=args.term,
                attributes=args.attribute,
                predicate=args.predicate,
                reference_terms=args.reference_term,
            )
            if args.compact:
                result = {
                    **{key: value for key, value in result.items() if key != "matches"},
                    "matches": [
                        {
                            "entity_id": entity["entity_id"],
                            "labels": entity.get("labels", []),
                            "view_count": entity.get("view_count"),
                            "gaussian_count": entity.get("gaussian_count"),
                        }
                        for entity in result["matches"]
                    ],
                }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "preflight-input":
            from navmem3d.world.preflight import inspect_world_builder_input

            result = inspect_world_builder_input(args.selection)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "generate-sam2-masks":
            from navmem3d.semantic.sam2_adapter import generate_automatic_masks

            result = generate_automatic_masks(
                args.image,
                args.output,
                checkpoint=args.checkpoint,
                points_per_side=args.points_per_side,
                max_masks=args.max_masks,
                device=args.device,
            )
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "masks": result["retained_mask_count"],
                        "elapsed_s": result["elapsed_s"],
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "lift-mask-set":
            from navmem3d.semantic.compiler import lift_mask_set

            result = lift_mask_set(
                args.masks,
                args.gaussian_ids,
                args.output,
                min_gaussians=args.min_gaussians,
                gaussian_weight_path=args.gaussian_weights,
                min_weight=args.min_weight,
                min_pixel_hits=args.min_pixel_hits,
            )
            print(
                json.dumps(
                    {"output": str(args.output), "proposals": result["proposal_count"]},
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "render-gsplat-views":
            from navmem3d.world.gsplat_render import render_gsplat_views

            result = render_gsplat_views(
                args.asset,
                args.output,
                view_count=args.views,
                width=args.width,
                height=args.height,
                horizontal_fov_deg=args.hfov,
                device=args.device,
                top_contributors=args.top_contributors,
                trajectory_path=args.trajectory,
                export_depth=args.export_depth,
            )
            print(
                json.dumps(
                    {"output": str(args.output), "views": len(result["views"])},
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "fuse-proposals":
            from navmem3d.semantic.fusion import fuse_proposal_indices

            result = fuse_proposal_indices(
                args.proposal_index,
                args.output,
                minimum_containment=args.minimum_containment,
                minimum_jaccard=args.minimum_jaccard,
            )
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "entities": result["entity_count"],
                        "multi_view_entities": result["multi_view_entity_count"],
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "derive-geometry":
            from navmem3d.semantic.geometry import derive_entity_geometry

            result = derive_entity_geometry(
                args.entities,
                args.asset,
                args.output,
                near_threshold_ratio=args.near_threshold_ratio,
            )
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "entities": len(result["entities"]),
                        "near_threshold": result["near_threshold"],
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "filter-entities":
            from navmem3d.semantic.filtering import build_searchable_entity_index

            result = build_searchable_entity_index(
                args.geometry_index,
                args.output,
                structural_axis_ratio=args.structural_axis_ratio,
                single_view_max_axis_ratio=args.single_view_max_axis_ratio,
            )
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "searchable_entities": result["searchable_entity_count"],
                        "structural_regions": result["structural_region_count"],
                        "ambiguous_regions": result["ambiguous_region_count"],
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "export-entity-crops":
            from navmem3d.semantic.crops import export_representative_crops

            result = export_representative_crops(
                args.entity_index,
                args.output,
                padding_ratio=args.padding_ratio,
            )
            print(
                json.dumps(
                    {"output": str(args.output), "crops": result["crop_count"]},
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "label-entity-crops":
            from navmem3d.semantic.clip_labeler import label_entity_crops

            result = label_entity_crops(
                args.entity_index,
                args.crop_manifest,
                args.vocabulary,
                args.output,
                model_name=args.model_name,
                pretrained=args.pretrained,
                device=args.device,
                batch_size=args.batch_size,
            )
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "labeled_entities": len(result["entities"]),
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "generate-guided-masks":
            from navmem3d.semantic.guided_masks import generate_guided_masks

            result = generate_guided_masks(
                args.image,
                args.output,
                vocabulary_path=args.vocabulary,
                sam_checkpoint=args.sam_checkpoint,
                detector_model=args.detector_model,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device=args.device,
            )
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "views": len(result["views"]),
                        "masks": result["total_mask_count"],
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "deduplicate-entities":
            from navmem3d.semantic.deduplication import deduplicate_entity_index

            result = deduplicate_entity_index(
                args.entity_index,
                args.output,
                minimum_containment=args.minimum_containment,
                minimum_jaccard=args.minimum_jaccard,
            )
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "entities": result["searchable_entity_count"],
                        "suppressed": result["deduplication"]["suppressed_count"],
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "build-observation-goals":
            from navmem3d.navigation.goal_pose import build_observation_goal_index

            result = build_observation_goal_index(
                args.entity_index, args.virtual_views, args.output
            )
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "entities": len(result["entities"]),
                        "goal_candidates": sum(
                            len(entity["goal_candidates"]) for entity in result["entities"]
                        ),
                        "reachability_filter_applied": result["reachability_filter_applied"],
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "query-navigation-goals":
            from navmem3d.navigation.query_goal import query_navigation_goals

            result = query_navigation_goals(
                args.entity_index,
                args.goal_index,
                terms=args.term,
                attributes=args.attribute,
                predicate=args.predicate,
                reference_terms=args.reference_term,
                max_goals_per_entity=args.max_goals_per_entity,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "build-visual-route":
            from navmem3d.navigation.visual_route import build_visual_route_graph
            result = build_visual_route_graph(args.patrol, args.renders, args.output, keyframe_stride=args.keyframe_stride)
            print(json.dumps({"output": str(args.output), "nodes": len(result["nodes"]), "edges": len(result["edges"])}, ensure_ascii=False))
            return 0
        if args.command == "plan-visual-route":
            from navmem3d.navigation.visual_route import plan_visual_route
            result = plan_visual_route(args.graph, start_node_id=args.start_node, target_node_id=args.target_node, output_path=args.output)
            print(json.dumps({"output": str(args.output), "steps": len(result["edge_sequence"]), "status": result["status"]}, ensure_ascii=False))
            return 0
        if args.command == "register-goals-b-to-a":
            from navmem3d.navigation.registration import register_world_b_to_a
            result = register_world_b_to_a(args.goal_index, transform_b_to_a=args.transform, output_path=args.output, registration_status=args.registration_status)
            print(json.dumps({"output": str(args.output), "entities": len(result["entities"]), "registration": result["registration"]["status"]}, ensure_ascii=False))
            return 0
        if args.command == "estimate-b-to-a":
            from navmem3d.navigation.registration import estimate_bounds_registration
            result = estimate_bounds_registration(args.b_splat, args.a_scene, output_path=args.output)
            print(json.dumps({"output": str(args.output), "scale": result["scale"], "status": result["status"]}, ensure_ascii=False))
            return 0
        if args.command == "estimate-b-to-a-visual":
            from navmem3d.navigation.visual_registration import estimate_visual_camera_registration
            result = estimate_visual_camera_registration(
                args.virtual_views, args.patrol, args.patrol_renders, args.output,
                stride=args.stride, minimum_similarity=args.minimum_similarity, device=args.device,
            )
            print(json.dumps({"output": str(args.output), "matches": result["match_count"], "median_residual_m": result["median_camera_residual_m"], "status": result["status"]}, ensure_ascii=False))
            return 0
        if args.command == "validate-b-to-a-pnp":
            from navmem3d.navigation.pnp_registration import estimate_pnp_registration
            result = estimate_pnp_registration(
                args.virtual_views, args.patrol, args.patrol_renders, args.asset, args.output,
                stride=args.stride, ratio_test=args.ratio_test, min_inliers=args.min_inliers,
                max_transform_translation_m=args.max_transform_translation_m,
            )
            print(json.dumps({"output": str(args.output), "accepted_pairs": result["accepted_pair_count"], "status": result["status"]}, ensure_ascii=False))
            return 0
    except (OSError, json.JSONDecodeError, SchemaError, ValueError, RuntimeError) as exc:
        parser = build_parser()
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
