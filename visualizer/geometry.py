from __future__ import annotations

from typing import Any

import numpy as np


def finite_vector(value: Any, *, length: int, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"Invalid {field_name}: expected {length} finite numbers")
    return array


def normalize_quaternion(value: Any) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quaternion_to_matrix(wxyz: Any) -> np.ndarray:
    w, x, y, z = normalize_quaternion(wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(matrix: Any) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    trace = float(np.trace(m))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            quat = np.array(
                [
                    (m[2, 1] - m[1, 2]) / s,
                    0.25 * s,
                    (m[0, 1] + m[1, 0]) / s,
                    (m[0, 2] + m[2, 0]) / s,
                ],
                dtype=np.float64,
            )
        elif axis == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            quat = np.array(
                [
                    (m[0, 2] - m[2, 0]) / s,
                    (m[0, 1] + m[1, 0]) / s,
                    0.25 * s,
                    (m[1, 2] + m[2, 1]) / s,
                ],
                dtype=np.float64,
            )
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            quat = np.array(
                [
                    (m[1, 0] - m[0, 1]) / s,
                    (m[0, 2] + m[2, 0]) / s,
                    (m[1, 2] + m[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float64,
            )
    return normalize_quaternion(quat)


def annotation_to_transform(
    center_xyz: Any,
    size_xyz: Any,
    orientation_wxyz: Any,
) -> np.ndarray:
    center = finite_vector(center_xyz, length=3, field_name="center_xyz")
    size = finite_vector(size_xyz, length=3, field_name="size_xyz")
    rotation = quaternion_to_matrix(orientation_wxyz)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation @ np.diag(size)
    transform[:3, 3] = center
    return transform


def transform_to_box(
    region_from_local: Any,
    local_min_xyz: Any,
    local_max_xyz: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform = np.asarray(region_from_local, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Invalid region_from_local: expected finite 4x4 matrix")
    if not np.allclose(transform[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-5):
        raise ValueError("Invalid region_from_local: expected affine transform")

    local_min = finite_vector(local_min_xyz, length=3, field_name="local_bounds_min_xyz")
    local_max = finite_vector(local_max_xyz, length=3, field_name="local_bounds_max_xyz")
    local_span = local_max - local_min
    if np.any(local_span <= 0):
        raise ValueError("Invalid local bounds: max must be greater than min on every axis")

    local_mid = (local_min + local_max) * 0.5
    center_h = transform @ np.array([local_mid[0], local_mid[1], local_mid[2], 1.0])
    center = center_h[:3]

    columns = transform[:3, :3]
    axis_lengths = np.linalg.norm(columns, axis=0)
    if np.any(axis_lengths <= 1e-12):
        raise ValueError("Invalid region transform: local axes must be non-zero")

    size = axis_lengths * local_span
    rotation = columns / axis_lengths
    if np.linalg.det(rotation) < 0:
        raise ValueError("Invalid region transform: reflections are not supported")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError("Invalid region transform: box axes must be orthogonal")
    return center, np.abs(size), matrix_to_quaternion(rotation)
