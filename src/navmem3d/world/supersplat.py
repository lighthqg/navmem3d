from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompressedSplats:
    means: object
    log_scales: object
    quaternions: object
    colors: object
    opacities: object
    sh_rest: object | None


def _unpack_111011(values):
    import numpy as np

    values = values.astype(np.uint32, copy=False)
    return np.stack(
        [
            ((values >> 21) & 0x7FF) / 2047.0,
            ((values >> 11) & 0x3FF) / 1023.0,
            (values & 0x7FF) / 2047.0,
        ],
        axis=1,
    ).astype(np.float32)


def _unpack_rotation(values):
    import numpy as np

    values = values.astype(np.uint32, copy=False)
    largest = (values >> 30).astype(np.int64)
    packed = np.stack(
        [(values >> 20) & 0x3FF, (values >> 10) & 0x3FF, values & 0x3FF],
        axis=1,
    )
    components = (packed.astype(np.float32) / 1023.0 - 0.5) / (
        2.0**-0.5
    )
    result = np.zeros((len(values), 4), dtype=np.float32)
    component_indices = np.array(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], dtype=np.int64
    )
    rows = np.arange(len(values))[:, None]
    result[rows, component_indices[largest]] = components
    omitted = np.sqrt(np.maximum(0.0, 1.0 - np.sum(components * components, axis=1)))
    result[np.arange(len(values)), largest] = omitted
    return result


def load_supersplat_compressed_ply(path: str | Path) -> CompressedSplats:
    """Decode the SuperSplat compressed PLY layout used by InteriorGS."""
    import numpy as np

    path = Path(path)
    with path.open("rb") as stream:
        header = bytearray()
        while not header.endswith(b"end_header\n"):
            byte = stream.read(1)
            if not byte:
                raise ValueError("truncated PLY header")
            header.extend(byte)
        lines = header.decode("ascii").splitlines()
        if "format binary_little_endian 1.0" not in lines:
            raise ValueError("expected a binary little-endian PLY")
        counts: dict[str, int] = {}
        sh_properties = 0
        current_element = None
        for line in lines:
            fields = line.split()
            if len(fields) == 3 and fields[0] == "element":
                current_element = fields[1]
                counts[current_element] = int(fields[2])
            elif current_element == "sh" and fields[:2] == ["property", "uchar"]:
                sh_properties += 1
        chunk_count = counts.get("chunk")
        vertex_count = counts.get("vertex")
        if not chunk_count or not vertex_count:
            raise ValueError("not a SuperSplat compressed PLY")

        chunk_bounds = np.fromfile(
            stream, dtype="<f4", count=chunk_count * 18
        ).reshape(chunk_count, 18)
        packed = np.fromfile(
            stream, dtype="<u4", count=vertex_count * 4
        ).reshape(vertex_count, 4)
        sh_rest = None
        if sh_properties:
            sh_bytes = np.fromfile(
                stream, dtype="u1", count=vertex_count * sh_properties
            ).reshape(vertex_count, sh_properties)
            sh_rest = (sh_bytes.astype(np.float32) / 256.0 - 0.5) * 8.0

    chunk_size = (vertex_count + chunk_count - 1) // chunk_count
    chunk_ids = np.minimum(np.arange(vertex_count) // chunk_size, chunk_count - 1)
    bounds = chunk_bounds[chunk_ids]

    position_unit = _unpack_111011(packed[:, 0])
    means = bounds[:, 0:3] + position_unit * (bounds[:, 3:6] - bounds[:, 0:3])
    scale_unit = _unpack_111011(packed[:, 2])
    log_scales = bounds[:, 6:9] + scale_unit * (bounds[:, 9:12] - bounds[:, 6:9])

    color = packed[:, 3]
    color_unit = np.stack(
        [(color >> 24) & 255, (color >> 16) & 255, (color >> 8) & 255], axis=1
    ).astype(np.float32) / 255.0
    colors = bounds[:, 12:15] + color_unit * (bounds[:, 15:18] - bounds[:, 12:15])
    opacities = (color & 255).astype(np.float32) / 255.0

    return CompressedSplats(
        means=means.astype(np.float32, copy=False),
        log_scales=log_scales.astype(np.float32, copy=False),
        quaternions=_unpack_rotation(packed[:, 1]),
        colors=np.clip(colors, 0.0, 1.0).astype(np.float32, copy=False),
        opacities=opacities,
        sh_rest=sh_rest,
    )
