from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from .domain import (
    ANNOTATOR_NOTES_FILE,
    MIN_ANNOTATION_EDGE,
    REGION_BOXES_FILE,
    Annotation,
    LoadedScene,
    SceneManifest,
)
from .geometry import annotation_to_transform, finite_vector, normalize_quaternion, transform_to_box

LOCAL_BOUNDS_MIN = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
LOCAL_BOUNDS_MAX = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def load_manifest(scene_dir: str | Path) -> SceneManifest:
    scene_path = Path(scene_dir).expanduser().resolve()
    notes_path = scene_path / ANNOTATOR_NOTES_FILE
    notes = notes_path.read_text(encoding="utf-8") if notes_path.is_file() else ""

    return SceneManifest(
        scene_dir=scene_path,
        annotations_asset=REGION_BOXES_FILE,
        annotator_notes=notes,
    )


def load_point_cloud(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    source = Path(path)
    pcd = o3d.io.read_point_cloud(str(source))
    if pcd.is_empty():
        raise ValueError(f"Scene cloud is empty: {source}")
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
    if colors is not None and len(colors) != len(points):
        colors = None
    return points, colors


def load_annotations(path: str | Path) -> list[Annotation]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        ids = payload["region_key"].astype(str)
        names = payload["display_name"].astype(str)
        transforms = payload["region_from_local"]
        mins = payload["local_bounds_min_xyz"]
        maxes = payload["local_bounds_max_xyz"]

    if not (
        ids.shape == (len(ids),)
        and names.shape == (len(ids),)
        and
        transforms.shape == (len(ids), 4, 4)
        and mins.shape == (len(ids), 3)
        and maxes.shape == (len(ids), 3)
    ):
        raise ValueError(f"Invalid annotation array shapes in {source}")

    annotations: list[Annotation] = []
    for idx in range(len(ids)):
        center, size, orientation = transform_to_box(transforms[idx], mins[idx], maxes[idx])
        annotations.append(
            Annotation(
                id=str(ids[idx]),
                name=str(names[idx]),
                center_xyz=finite_vector(center, length=3, field_name="center_xyz"),
                size_xyz=np.maximum(
                    finite_vector(size, length=3, field_name="size_xyz"),
                    MIN_ANNOTATION_EDGE,
                ),
                orientation_wxyz=normalize_quaternion(orientation),
            )
        )
    return annotations


def save_annotations(path: str | Path, annotations: list[Annotation]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = len(annotations)
    transforms = (
        np.stack(
            [
                annotation_to_transform(item.center_xyz, item.size_xyz, item.orientation_wxyz)
                for item in annotations
            ]
        ).astype(np.float32)
        if count
        else np.empty((0, 4, 4), dtype=np.float32)
    )
    bounds_min = (
        np.repeat(LOCAL_BOUNDS_MIN[None, :], count, axis=0)
        if count
        else np.empty((0, 3), dtype=np.float32)
    )
    bounds_max = (
        np.repeat(LOCAL_BOUNDS_MAX[None, :], count, axis=0)
        if count
        else np.empty((0, 3), dtype=np.float32)
    )
    with open(target, "wb") as handle:
        np.savez_compressed(
            handle,
            region_key=np.array([item.id for item in annotations], dtype=str),
            display_name=np.array([item.name for item in annotations], dtype=str),
            region_from_local=transforms,
            local_bounds_min_xyz=bounds_min.astype(np.float32),
            local_bounds_max_xyz=bounds_max.astype(np.float32),
        )


def load_scene(scene_dir: str | Path) -> LoadedScene:
    manifest = load_manifest(scene_dir)
    if not manifest.cloud_path.is_file():
        raise FileNotFoundError(f"Scene cloud not found: {manifest.cloud_path}")
    if not manifest.annotations_path.is_file():
        raise FileNotFoundError(f"Scene annotations not found: {manifest.annotations_path}")
    points, colors = load_point_cloud(manifest.cloud_path)
    annotations = load_annotations(manifest.annotations_path)
    return LoadedScene(manifest=manifest, points=points, colors=colors, annotations=annotations)
