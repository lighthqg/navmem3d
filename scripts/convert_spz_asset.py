#!/usr/bin/env python3
"""Convert an SPZ world into standard 3DGS PLY and the compact .splat baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import spz


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()

    options = spz.UnpackOptions()
    options.to_coord = spz.CoordinateSystem.RUB
    cloud = spz.load_spz(str(args.input), options)
    stem = args.output_stem
    stem.parent.mkdir(parents=True, exist_ok=True)

    pack = spz.PackOptions()
    pack.from_coord = spz.CoordinateSystem.RUB
    ply_path = stem.with_suffix(".ply")
    if not spz.save_splat_to_ply(cloud, pack, str(ply_path)):
        raise RuntimeError("SPZ-to-PLY conversion failed")

    # The existing .splat renderer follows the usual RDF convention
    # (X right, Y down, Z forward), so convert explicitly from SPZ's RUB.
    cloud.convert_coordinates(spz.CoordinateSystem.RUB, spz.CoordinateSystem.RDF)
    count = cloud.num_points
    dtype = np.dtype([
        ("xyz", "<f4", (3,)), ("scale", "<f4", (3,)),
        ("rgba", "u1", (4,)), ("quat", "u1", (4,)),
    ])
    packed = np.empty(count, dtype=dtype)
    packed["xyz"] = cloud.positions.reshape(-1, 3)
    packed["scale"] = np.exp(cloud.scales.reshape(-1, 3))
    packed["rgba"][:, :3] = np.clip(cloud.colors.reshape(-1, 3) * 255, 0, 255).astype(np.uint8)
    alpha = 1.0 / (1.0 + np.exp(-cloud.alphas))
    packed["rgba"][:, 3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    quats = cloud.rotations.reshape(-1, 4)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True).clip(1e-8)
    packed["quat"] = np.clip(quats * 128 + 128, 0, 255).astype(np.uint8)
    splat_path = stem.with_suffix(".splat")
    packed.tofile(splat_path)

    positions = packed["xyz"]
    result = {
        "source": str(args.input), "ply_coordinate_system": "RDF",
        "splat_coordinate_system": "RDF",
        "gaussian_count": int(count), "sh_degree": int(cloud.sh_degree),
        "antialiased": bool(cloud.antialiased),
        "bounds": {"min": positions.min(0).tolist(), "max": positions.max(0).tolist()},
        "ply": str(ply_path), "splat": str(splat_path),
    }
    stem.with_suffix(".conversion.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
