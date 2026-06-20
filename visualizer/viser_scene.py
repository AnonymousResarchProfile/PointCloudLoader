from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import viser

from .domain import Annotation
from .editor import SceneEditor
from .geometry import normalize_quaternion
from .sampling import CloudPicker, render_scene_colors

log = logging.getLogger(__name__)

ANNOTATION_FILL = (255, 105, 180)
ANNOTATION_WIRE = (255, 20, 147)
ANNOTATION_SELECTED = (255, 182, 193)
POINT_SIZE = 0.023


@dataclass
class RenderedAnnotation:
    solid: Any
    outline: Any
    control: Any
    label: Any

    def remove(self) -> None:
        for handle in (self.solid, self.outline, self.control, self.label):
            try:
                handle.remove()
            except Exception as exc:  # noqa: BLE001
                log.debug("Ignoring stale Viser handle removal: %s", exc)


class ViserSceneAdapter:
    def __init__(self, server: viser.ViserServer, editor: SceneEditor) -> None:
        self.server = server
        self.editor = editor
        self.on_editor_changed = None
        self._cloud_handle: Any | None = None
        self._rendered: dict[str, RenderedAnnotation] = {}
        self.display_points: np.ndarray | None = None
        self.display_colors: np.ndarray | None = None
        self._placement_callback_active = False
        self._selection_callback_active = False

    @property
    def picker(self) -> CloudPicker:
        return CloudPicker(self.display_points)

    def refresh_all(self) -> None:
        self.refresh_cloud()
        self.refresh_annotations()

    def refresh_cloud(self) -> None:
        state = self.editor.state
        scene = state.scene
        self._remove_cloud()
        if scene is None or len(scene.points) == 0:
            self.display_points = np.empty((0, 3), dtype=np.float32)
            self.display_colors = np.empty((0, 3), dtype=np.uint8)
            return

        self.display_points = scene.points
        self.display_colors = render_scene_colors(scene.points, scene.colors)
        self._cloud_handle = self.server.scene.add_point_cloud(
            name="/scene/cloud",
            points=self.display_points,
            colors=self.display_colors,
            point_size=POINT_SIZE,
            point_shape="sparkle",
            point_shading="gradient",
            precision="float32",
        )

    def refresh_annotations(self) -> None:
        scene = self.editor.state.scene
        live_ids = set()
        if scene is not None:
            for annotation in scene.annotations:
                live_ids.add(annotation.id)
                if annotation.id in self._rendered:
                    self._update_annotation(annotation)
                else:
                    self._draw_annotation(annotation)

        for stale_id in sorted(set(self._rendered) - live_ids):
            rendered = self._rendered.pop(stale_id)
            rendered.remove()
        self._apply_visibility_and_selection()

    def arm_point_placement(self, client: viser.ClientHandle | None = None) -> None:
        if self._placement_callback_active:
            return
        self._placement_callback_active = True

        @self.server.scene.on_pointer_event("click")
        def _place(event: viser.ScenePointerEvent) -> None:
            self.server.scene.remove_pointer_callback()
            self._placement_callback_active = False
            if event.ray_origin is None or event.ray_direction is None:
                return
            origin = np.asarray(event.ray_origin, dtype=np.float64)
            direction = np.asarray(event.ray_direction, dtype=np.float64)
            center = self.picker.nearest_point_on_ray(origin, direction)
            self.editor.create_annotation(center)
            self.refresh_annotations()
            self._notify_editor_changed()

        self._enable_pointer(client)

    def arm_annotation_selection(self, client: viser.ClientHandle | None = None) -> None:
        if self._selection_callback_active:
            return
        self._selection_callback_active = True

        @self.server.scene.on_pointer_event("click")
        def _select(event: viser.ScenePointerEvent) -> None:
            self.server.scene.remove_pointer_callback()
            self._selection_callback_active = False
            if event.ray_origin is None or event.ray_direction is None:
                return
            origin = np.asarray(event.ray_origin, dtype=np.float64)
            direction = np.asarray(event.ray_direction, dtype=np.float64)
            direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
            best_id = self._nearest_annotation_id(origin, direction)
            self.editor.select_annotation(best_id)
            self._apply_visibility_and_selection()
            self._notify_editor_changed()

        self._enable_pointer(client)

    def _nearest_annotation_id(self, origin: np.ndarray, direction: np.ndarray) -> str | None:
        scene = self.editor.state.scene
        if scene is None:
            return None
        best_id: str | None = None
        best_distance = float("inf")
        for annotation in scene.annotations:
            rel = annotation.center_xyz - origin
            depth = float(rel @ direction)
            if depth < 0:
                continue
            closest = origin + depth * direction
            distance = float(np.linalg.norm(annotation.center_xyz - closest))
            tolerance = max(0.25, float(np.max(annotation.size_xyz)) * 0.75)
            if distance <= tolerance and distance < best_distance:
                best_id = annotation.id
                best_distance = distance
        return best_id

    def _draw_annotation(self, annotation: Annotation) -> None:
        rendered = self._rendered.pop(annotation.id, None)
        if rendered is not None:
            rendered.remove()

        solid = self.server.scene.add_box(
            name=f"/annotations/{annotation.id}/volume",
            color=ANNOTATION_FILL,
            dimensions=annotation.size_xyz,
            opacity=0.14,
            material="toon3",
            wxyz=annotation.orientation_wxyz,
            position=annotation.center_xyz,
            visible=self.editor.state.prefs.annotations_visible,
        )
        outline = self.server.scene.add_box(
            name=f"/annotations/{annotation.id}/outline",
            color=ANNOTATION_WIRE,
            dimensions=annotation.size_xyz,
            wireframe=True,
            wxyz=annotation.orientation_wxyz,
            position=annotation.center_xyz,
            visible=self.editor.state.prefs.annotations_visible,
        )
        control = self.server.scene.add_transform_controls(
            name=f"/annotation_controls/{annotation.id}",
            scale=0.55,
            line_width=3.0,
            wxyz=annotation.orientation_wxyz,
            position=annotation.center_xyz,
            visible=False,
        )
        label = self.server.scene.add_label(
            name=f"/annotation_labels/{annotation.id}",
            text=annotation.name,
            position=self._label_position(annotation),
            visible=self.editor.state.prefs.annotations_visible,
            anchor="center-center",
        )
        self._rendered[annotation.id] = RenderedAnnotation(solid, outline, control, label)

        @control.on_update
        def _(_) -> None:
            self.editor.select_annotation(annotation.id)
            self.editor.transform_selected(
                np.asarray(control.position, dtype=np.float64),
                normalize_quaternion(control.wxyz),
            )
            current = self.editor.selected_annotation()
            if current is not None:
                self._update_annotation(current, from_control=True)
            self._notify_editor_changed()

    def _update_annotation(self, annotation: Annotation, *, from_control: bool = False) -> None:
        rendered = self._rendered.get(annotation.id)
        if rendered is None:
            self._draw_annotation(annotation)
            return

        rendered.solid.dimensions = annotation.size_xyz
        rendered.solid.position = annotation.center_xyz
        rendered.solid.wxyz = annotation.orientation_wxyz
        rendered.outline.dimensions = annotation.size_xyz
        rendered.outline.position = annotation.center_xyz
        rendered.outline.wxyz = annotation.orientation_wxyz
        rendered.label.text = annotation.name
        rendered.label.position = self._label_position(annotation)
        if not from_control:
            rendered.control.position = annotation.center_xyz
            rendered.control.wxyz = annotation.orientation_wxyz
        self._apply_visibility_and_selection()

    def _apply_visibility_and_selection(self) -> None:
        visible = self.editor.state.prefs.annotations_visible
        selected_id = self.editor.state.selected_annotation_id
        for annotation_id, rendered in self._rendered.items():
            selected = annotation_id == selected_id
            rendered.solid.visible = visible
            rendered.outline.visible = visible
            rendered.label.visible = visible
            rendered.control.visible = visible and selected
            rendered.outline.color = ANNOTATION_SELECTED if selected else ANNOTATION_WIRE

    def _remove_cloud(self) -> None:
        if self._cloud_handle is None:
            return
        try:
            self._cloud_handle.remove()
        except Exception as exc:  # noqa: BLE001
            log.debug("Ignoring stale point cloud handle removal: %s", exc)
        self._cloud_handle = None

    @staticmethod
    def _label_position(annotation: Annotation) -> np.ndarray:
        return annotation.center_xyz

    @staticmethod
    def _enable_pointer(client: viser.ClientHandle | None) -> None:
        if client is None:
            return
        try:
            client.scene_pointer.enable = True
        except Exception:
            pass

    def _notify_editor_changed(self) -> None:
        if self.on_editor_changed is not None:
            self.on_editor_changed()
