from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from navmem3d.capture.habitat import pinhole_intrinsics
from navmem3d.schemas import CaptureSequence, FrameRecord


@dataclass(frozen=True)
class GlbCaptureConfig:
    scene: Path
    output_dir: Path
    sequence_id: str
    frame_count: int = 120
    width: int = 640
    height: int = 480
    horizontal_fov_deg: float = 90.0
    sensor_height_m: float = 1.25
    path_radius_ratio: float = 0.28


def capture_sequence(config: GlbCaptureConfig) -> CaptureSequence:
    """Render a deterministic inward-looking room patrol with Mesa-compatible pyrender."""
    try:
        import numpy as np  # type: ignore
        import pyrender  # type: ignore
        import trimesh  # type: ignore
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("capture-glb requires numpy, trimesh, pyrender and Pillow") from exc

    if not config.scene.exists():
        raise FileNotFoundError(config.scene)
    if config.frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    if not 0 < config.path_radius_ratio < 0.5:
        raise ValueError("path_radius_ratio must be between 0 and 0.5")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = config.output_dir / "rgb"
    depth_dir = config.output_dir / "depth"
    rgb_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)

    mesh_scene = trimesh.load(config.scene, force="scene")
    if mesh_scene.is_empty:
        raise RuntimeError("GLB scene contains no geometry")
    bounds = mesh_scene.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    height_z = min(float(bounds[0, 2] + config.sensor_height_m), float(bounds[1, 2] - 0.2))
    radius_x = float(bounds[1, 0] - bounds[0, 0]) * config.path_radius_ratio
    radius_y = float(bounds[1, 1] - bounds[0, 1]) * config.path_radius_ratio

    scene = pyrender.Scene.from_trimesh_scene(
        mesh_scene, bg_color=[20, 20, 20, 255], ambient_light=[0.35, 0.35, 0.35]
    )
    vertical_fov = 2.0 * math.atan(
        math.tan(math.radians(config.horizontal_fov_deg) / 2.0)
        * config.height
        / config.width
    )
    camera = pyrender.PerspectiveCamera(yfov=vertical_fov, aspectRatio=config.width / config.height)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.5)
    renderer = pyrender.OffscreenRenderer(config.width, config.height)
    intrinsics = pinhole_intrinsics(config.width, config.height, config.horizontal_fov_deg)
    frames: list[FrameRecord] = []

    try:
        for index in range(config.frame_count):
            angle = 2.0 * math.pi * index / config.frame_count
            eye = np.array(
                [center[0] + radius_x * math.cos(angle), center[1] + radius_y * math.sin(angle), height_z],
                dtype=float,
            )
            target = np.array([center[0], center[1], height_z - 0.1], dtype=float)
            pose = look_at_opengl(eye, target)
            camera_node = scene.add(camera, pose=pose)
            light_node = scene.add(light, pose=pose)
            color, depth = renderer.render(scene)
            scene.remove_node(camera_node)
            scene.remove_node(light_node)

            rgb_name = f"rgb/{index:06d}.png"
            depth_name = f"depth/{index:06d}.npy"
            Image.fromarray(color[..., :3].astype("uint8")).save(config.output_dir / rgb_name)
            np.save(config.output_dir / depth_name, depth.astype("float32"))
            frames.append(
                FrameRecord(
                    frame_id=f"f{index:06d}",
                    timestamp_s=index / 10.0,
                    rgb_uri=rgb_name,
                    depth_uri=depth_name,
                    intrinsics=intrinsics,
                    t_world_camera=tuple(float(value) for value in pose.reshape(-1)),
                )
            )
    finally:
        renderer.delete()

    sequence = CaptureSequence(
        sequence_id=config.sequence_id,
        coordinate_frame="gltf_world_z_up__camera_opengl",
        camera_model="pinhole",
        frames=tuple(frames),
    )
    sequence.save(config.output_dir / "capture_sequence.json")
    return sequence


def look_at_opengl(eye: object, target: object) -> object:
    import numpy as np  # type: ignore

    eye_array = np.asarray(eye, dtype=float)
    target_array = np.asarray(target, dtype=float)
    forward = target_array - eye_array
    length = np.linalg.norm(forward)
    if length == 0:
        raise ValueError("eye and target must differ")
    forward /= length
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-8:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye_array
    return pose

