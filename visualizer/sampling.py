from __future__ import annotations

import numpy as np


def render_scene_colors(points: np.ndarray, source_colors: np.ndarray | None) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if source_colors is not None and len(source_colors) == len(points):
        return (np.clip(source_colors.astype(np.float32, copy=False), 0.0, 1.0) * 255.0).astype(
            np.uint8
        )
    return np.full((len(points), 3), 220, dtype=np.uint8)


class CloudPicker:
    def __init__(self, points: np.ndarray | None):
        self.points = points

    def nearest_point_on_ray(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> np.ndarray:
        if self.points is None or len(self.points) == 0:
            return ray_origin + ray_direction * 2.0

        direction = ray_direction / max(float(np.linalg.norm(ray_direction)), 1e-12)
        sample = self.points
        if len(sample) > 100_000:
            stride = int(np.ceil(len(sample) / 100_000))
            sample = sample[::stride]
        rel = sample - ray_origin
        depth = rel @ direction
        in_front = depth > 0
        if not np.any(in_front):
            return ray_origin + direction * 2.0

        rel = rel[in_front]
        sample = sample[in_front]
        depth = depth[in_front]
        closest = ray_origin + depth[:, None] * direction
        distances = np.linalg.norm(sample - closest, axis=1)
        return sample[int(np.argmin(distances))]
