from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np

from .domain import (
    MIN_ANNOTATION_EDGE,
    Annotation,
    EditorChange,
    EditorState,
    ViewerPrefs,
)
from .geometry import finite_vector, normalize_quaternion
from .io import load_scene, save_annotations


class SceneEditor:
    def __init__(self) -> None:
        self.state = EditorState()

    @property
    def scene(self) -> LoadedScene:
        if self.state.scene is None:
            raise RuntimeError("No scene is loaded")
        return self.state.scene

    @property
    def annotations(self) -> list[Annotation]:
        return self.scene.annotations

    def load_scene(self, path: str | Path) -> EditorChange:
        scene = load_scene(path)
        self.state.scene = scene
        self.state.prefs = ViewerPrefs(annotations_visible=True)
        self.state.selected_annotation_id = scene.annotations[0].id if scene.annotations else None
        self.state.dirty_annotations = False
        return EditorChange("scene_loaded", str(scene.manifest.scene_dir))

    def set_annotations_visible(self, value: bool) -> EditorChange:
        self.state.prefs.annotations_visible = bool(value)
        return EditorChange("annotations_visibility_changed")

    def selected_annotation(self) -> Annotation | None:
        selected_id = self.state.selected_annotation_id
        if selected_id is None:
            return None
        return next((item for item in self.annotations if item.id == selected_id), None)

    def select_annotation(self, annotation_id: str | None) -> EditorChange:
        if annotation_id is not None and not any(item.id == annotation_id for item in self.annotations):
            annotation_id = None
        self.state.selected_annotation_id = annotation_id
        return EditorChange("selection_changed", annotation_id or "")

    def create_annotation(self, center: np.ndarray, name: str = "region") -> EditorChange:
        annotation = Annotation(
            id=uuid.uuid4().hex[:10],
            name=name,
            center_xyz=finite_vector(center, length=3, field_name="center"),
            size_xyz=np.array([0.6, 0.6, 0.6], dtype=np.float64),
        )
        self.annotations.append(annotation)
        self.state.selected_annotation_id = annotation.id
        self.state.dirty_annotations = True
        return EditorChange("annotations_changed", annotation.id)

    def delete_selected(self) -> EditorChange:
        selected_id = self.state.selected_annotation_id
        if selected_id is None:
            return EditorChange("noop", "No selected annotation")
        self.scene.annotations = [item for item in self.annotations if item.id != selected_id]
        self.state.selected_annotation_id = self.annotations[0].id if self.annotations else None
        self.state.dirty_annotations = True
        return EditorChange("annotations_changed", selected_id)

    def delete_all(self) -> EditorChange:
        self.scene.annotations = []
        self.state.selected_annotation_id = None
        self.state.dirty_annotations = True
        return EditorChange("annotations_changed", "all")

    def rename_selected(self, name: str) -> EditorChange:
        annotation = self.selected_annotation()
        if annotation is None:
            return EditorChange("noop", "No selected annotation")
        annotation.name = str(name)
        self.state.dirty_annotations = True
        return EditorChange("annotations_changed", annotation.id)

    def transform_selected(self, center: np.ndarray, orientation: np.ndarray) -> EditorChange:
        annotation = self.selected_annotation()
        if annotation is None:
            return EditorChange("noop", "No selected annotation")
        annotation.center_xyz = finite_vector(center, length=3, field_name="center")
        annotation.orientation_wxyz = normalize_quaternion(orientation)
        self.state.dirty_annotations = True
        return EditorChange("annotations_changed", annotation.id)

    def resize_selected(self, size: np.ndarray) -> EditorChange:
        annotation = self.selected_annotation()
        if annotation is None:
            return EditorChange("noop", "No selected annotation")
        annotation.size_xyz = np.maximum(
            finite_vector(size, length=3, field_name="size"),
            MIN_ANNOTATION_EDGE,
        )
        self.state.dirty_annotations = True
        return EditorChange("annotations_changed", annotation.id)

    def save_annotations(self) -> EditorChange:
        save_annotations(self.scene.manifest.annotations_path, self.annotations)
        self.state.dirty_annotations = False
        return EditorChange("annotations_saved", str(self.scene.manifest.annotations_path))
