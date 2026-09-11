from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

from navmem3d.schemas import CaptureSequence, FrameRecord


@dataclass(frozen=True)
class HabitatCaptureConfig:
    scene: Path
    output_dir: Path
    sequence_id: str
    frame_count: int = 120
    width: int = 640
    height: int = 480
    horizontal_fov_deg: float = 90.0
    sensor_height_m: float = 1.25
    seed: int = 7


def quaternion_xyzw_to_matrix(
    x: float, y: float, z: float, w: float, position: Iterable[float]
) -> tuple[float, ...]:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("quaternion must be non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    px, py, pz = (float(value) for value in position)
    return (
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), px,
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), py,
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), pz,
        0.0, 0.0, 0.0, 1.0,
    )


def pinhole_intrinsics(width: int, height: int, horizontal_fov_deg: float) -> tuple[float, ...]:
    if width < 1 or height < 1 or not 0 < horizontal_fov_deg < 180:
        raise ValueError("invalid image size or field of view")
    fx = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    fy = fx
    return (fx, 0.0, width / 2.0, 0.0, fy, height / 2.0, 0.0, 0.0, 1.0)


def capture_sequence(config: HabitatCaptureConfig) -> CaptureSequence:
    """Render a deterministic RGB-D patrol using an optional Habitat-Sim install."""
    try:
        import habitat_sim  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image
        from habitat_sim.utils.common import quat_from_angle_axis  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "capture-habitat requires habitat-sim and numpy in a Python 3.9/3.10 environment"
        ) from exc

    if not config.scene.exists():
        raise FileNotFoundError(config.scene)
    if config.frame_count < 2:
        raise ValueError("frame_count must be at least 2")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = config.output_dir / "rgb"
    depth_dir = config.output_dir / "depth"
    rgb_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)

    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_id = str(config.scene.resolve())
    simulator_config.random_seed = config.seed

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [config.height, config.width]
    rgb.position = [0.0, config.sensor_height_m, 0.0]
    rgb.hfov = config.horizontal_fov_deg

    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [config.height, config.width]
    depth.position = [0.0, config.sensor_height_m, 0.0]
    depth.hfov = config.horizontal_fov_deg

    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = [rgb, depth]
    intrinsics = pinhole_intrinsics(config.width, config.height, config.horizontal_fov_deg)
    frames: list[FrameRecord] = []

    with habitat_sim.Simulator(habitat_sim.Configuration(simulator_config, [agent_config])) as sim:
        sim.seed(config.seed)
        points = _sample_patrol_points(sim, config.frame_count)
        agent = sim.initialize_agent(0)
        for index, point in enumerate(points):
            following = points[min(index + 1, len(points) - 1)]
            delta_x = float(following[0] - point[0])
            delta_z = float(following[2] - point[2])
            yaw = math.atan2(delta_x, -delta_z) if delta_x or delta_z else 0.0
            state = habitat_sim.AgentState()
            state.position = point
            state.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
            agent.set_state(state, infer_sensor_states=True)
            observations = sim.get_sensor_observations()

            rgb_name = f"rgb/{index:06d}.png"
            depth_name = f"depth/{index:06d}.npy"
            Image.fromarray(observations["rgb"][..., :3].astype("uint8")).save(
                config.output_dir / rgb_name
            )
            np.save(config.output_dir / depth_name, observations["depth"].astype("float32"))

            sensor_state = agent.get_state().sensor_states["rgb"]
            qx, qy, qz, qw = _quaternion_xyzw(sensor_state.rotation)
            frames.append(
                FrameRecord(
                    frame_id=f"f{index:06d}",
                    timestamp_s=index / 10.0,
                    rgb_uri=rgb_name,
                    depth_uri=depth_name,
                    intrinsics=intrinsics,
                    t_world_camera=quaternion_xyzw_to_matrix(
                        qx, qy, qz, qw, sensor_state.position
                    ),
                )
            )

    sequence = CaptureSequence(
        sequence_id=config.sequence_id,
        coordinate_frame="habitat_world_y_up_right_handed",
        camera_model="pinhole",
        frames=tuple(frames),
    )
    sequence.save(config.output_dir / "capture_sequence.json")
    return sequence


def _sample_patrol_points(sim: Any, frame_count: int) -> list[Any]:
    """Join random navigable shortest paths, then resample by arc length."""
    import numpy as np  # type: ignore
    import habitat_sim  # type: ignore

    polyline: list[Any] = []
    attempts = 0
    while len(polyline) < 8 and attempts < 30:
        attempts += 1
        start = sim.pathfinder.get_random_navigable_point()
        end = sim.pathfinder.get_random_navigable_point()
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = end
        if sim.pathfinder.find_path(path) and len(path.points) >= 2:
            if polyline:
                polyline.extend(path.points[1:])
            else:
                polyline.extend(path.points)
    if len(polyline) < 2:
        raise RuntimeError("could not sample a navigable patrol in the scene")

    segments = [float(np.linalg.norm(polyline[i + 1] - polyline[i])) for i in range(len(polyline) - 1)]
    total = sum(segments)
    targets = np.linspace(0.0, total, frame_count)
    output: list[Any] = []
    segment_index = 0
    traversed = 0.0
    for target in targets:
        while segment_index < len(segments) - 1 and traversed + segments[segment_index] < target:
            traversed += segments[segment_index]
            segment_index += 1
        length = segments[segment_index]
        ratio = 0.0 if length == 0 else (target - traversed) / length
        output.append(polyline[segment_index] * (1 - ratio) + polyline[segment_index + 1] * ratio)
    return output


def _quaternion_xyzw(rotation: Any) -> tuple[float, float, float, float]:
    if all(hasattr(rotation, name) for name in ("x", "y", "z", "w")):
        return float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)
    if hasattr(rotation, "vector") and hasattr(rotation, "scalar"):
        vector = rotation.vector
        return float(vector[0]), float(vector[1]), float(vector[2]), float(rotation.scalar)
    raise TypeError(f"unsupported quaternion type: {type(rotation)!r}")

