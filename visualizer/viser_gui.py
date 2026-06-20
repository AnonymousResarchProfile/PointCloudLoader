from __future__ import annotations

import logging
from pathlib import Path

import viser

from .domain import REGION_BOXES_FILE, SCENE_CLOUD_FILE
from .editor import SceneEditor
from .viser_scene import ViserSceneAdapter

log = logging.getLogger(__name__)


def discover_scene_paths(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if (root / SCENE_CLOUD_FILE).is_file() and (root / REGION_BOXES_FILE).is_file():
        return [root]
    scene_dirs = {
        cloud_path.parent
        for cloud_path in root.rglob(SCENE_CLOUD_FILE)
        if cloud_path.is_file() and (cloud_path.parent / REGION_BOXES_FILE).is_file()
    }
    return sorted(scene_dirs, key=lambda path: path.relative_to(root).as_posix())


def scene_label(scene_path: Path, root: Path) -> str:
    try:
        label = scene_path.relative_to(root).as_posix()
    except ValueError:
        label = scene_path.name
    return label if label != "." else scene_path.name


class ViserGuiAdapter:
    def __init__(
        self,
        server: viser.ViserServer,
        editor: SceneEditor,
        renderer: ViserSceneAdapter,
        *,
        scene_root: Path,
        scene_paths: list[Path],
    ) -> None:
        self.server = server
        self.editor = editor
        self.renderer = renderer
        self.scene_root = scene_root
        self.scene_paths = scene_paths
        self.path_by_label: dict[str, Path] = {}
        self.syncing = False
        self.renderer.on_editor_changed = self.refresh_annotations
        self._build()
        self.refresh_all()

    def _build(self) -> None:
        self._reset_scene_options(self.scene_paths)
        current_scene = self.editor.scene.manifest.scene_dir
        self.scene_select = self.server.gui.add_dropdown(
            "Scene",
            options=list(self.path_by_label),
            initial_value=scene_label(current_scene, self.scene_root),
        )
        self.scene_status = self.server.gui.add_text(
            "Selected scene",
            initial_value=str(current_scene),
            disabled=True,
        )
        self.refresh_scenes_btn = self.server.gui.add_button(
            "Refresh scenes",
            icon=viser.Icon.REFRESH,
        )

        self.notes = self.server.gui.add_text(
            "Annotator notes",
            initial_value=self.editor.scene.manifest.annotator_notes,
            multiline=True,
            disabled=True,
        )

        self.visible = self.server.gui.add_checkbox(
            "Display annotations",
            initial_value=self.editor.state.prefs.annotations_visible,
        )
        self.place_btn = self.server.gui.add_button("Place from cloud", icon=viser.Icon.CROSSHAIR)
        self.select_btn = self.server.gui.add_button("Click to select", icon=viser.Icon.POINTER)
        self.delete_btn = self.server.gui.add_button("Remove active", icon=viser.Icon.TRASH)
        self.clear_btn = self.server.gui.add_button("Remove all", color="red", icon=viser.Icon.TRASH_X)
        self.annotation_select = self.server.gui.add_dropdown(
            "Active annotation",
            options=["None"],
            initial_value="None",
        )
        self.name_input = self.server.gui.add_text("Name", initial_value="region")
        self.size_x = self._size_slider("Size X")
        self.size_y = self._size_slider("Size Y")
        self.size_z = self._size_slider("Size Z")
        self.save_annotations_btn = self.server.gui.add_button(
            "Write annotations",
            icon=viser.Icon.DEVICE_FLOPPY,
        )

        self._wire_callbacks()

    def _wire_callbacks(self) -> None:
        @self.scene_select.on_update
        def _(_) -> None:
            if self.syncing:
                return
            self._load_selected_scene()

        @self.refresh_scenes_btn.on_click
        def _(_) -> None:
            refreshed = discover_scene_paths(self.scene_root)
            self._reset_scene_options(refreshed)
            self.scene_select.options = list(self.path_by_label)
            current = self.editor.scene.manifest.scene_dir
            if current in refreshed:
                self.scene_select.value = scene_label(current, self.scene_root)
            elif refreshed:
                self.scene_select.value = scene_label(refreshed[0], self.scene_root)

        @self.visible.on_update
        def _(_) -> None:
            if not self.syncing:
                self.editor.set_annotations_visible(bool(self.visible.value))
                self.renderer.refresh_annotations()

        @self.place_btn.on_click
        def _(event: viser.GuiEvent) -> None:
            self.renderer.arm_point_placement(event.client)

        @self.select_btn.on_click
        def _(event: viser.GuiEvent) -> None:
            self.renderer.arm_annotation_selection(event.client)

        @self.delete_btn.on_click
        def _(_) -> None:
            self.editor.delete_selected()
            self.renderer.refresh_annotations()
            self.refresh_annotations()

        @self.clear_btn.on_click
        def _(_) -> None:
            self.editor.delete_all()
            self.renderer.refresh_annotations()
            self.refresh_annotations()

        @self.annotation_select.on_update
        def _(_) -> None:
            if self.syncing:
                return
            self.editor.select_annotation(self._selected_id_from_label(self.annotation_select.value))
            self.renderer.refresh_annotations()
            self.refresh_annotations()

        @self.name_input.on_update
        def _(_) -> None:
            if self.syncing:
                return
            self.editor.rename_selected(self.name_input.value)
            self.renderer.refresh_annotations()
            self.refresh_annotations()

        for slider, axis in ((self.size_x, 0), (self.size_y, 1), (self.size_z, 2)):
            self._wire_size_slider(slider, axis)

        @self.save_annotations_btn.on_click
        def _(_) -> None:
            self.editor.save_annotations()
            self.refresh_all()

    def _wire_size_slider(self, slider, axis: int) -> None:
        @slider.on_update
        def _(_) -> None:
            if self.syncing:
                return
            annotation = self.editor.selected_annotation()
            if annotation is None:
                return
            size = annotation.size_xyz.copy()
            size[axis] = float(slider.value)
            self.editor.resize_selected(size)
            self.renderer.refresh_annotations()
            self.refresh_annotations()

    def _load_selected_scene(self) -> None:
        selected_path = self.path_by_label.get(self.scene_select.value)
        if selected_path is None:
            return
        try:
            self.editor.load_scene(selected_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed loading scene %s: %s", selected_path, exc)
            self.scene_status.value = f"Failed: {selected_path}"
            return
        self.renderer.refresh_all()
        self.refresh_all()

    def refresh_all(self) -> None:
        self.syncing = True
        state = self.editor.state
        scene = self.editor.scene
        self.scene_status.value = str(scene.manifest.scene_dir)
        self.visible.value = state.prefs.annotations_visible
        self.notes.value = scene.manifest.annotator_notes
        self.syncing = False
        self.refresh_annotations()

    def refresh_annotations(self) -> None:
        self.syncing = True
        annotations = self.editor.annotations
        options = ["None"] + [self._annotation_label(item) for item in annotations]
        self.annotation_select.options = options
        selected = self.editor.selected_annotation()
        self.annotation_select.value = "None" if selected is None else self._annotation_label(selected)
        self.name_input.disabled = selected is None
        for slider in (self.size_x, self.size_y, self.size_z):
            slider.disabled = selected is None
        if selected is not None:
            self.name_input.value = selected.name
            self.size_x.value = float(selected.size_xyz[0])
            self.size_y.value = float(selected.size_xyz[1])
            self.size_z.value = float(selected.size_xyz[2])
        self.syncing = False

    def _reset_scene_options(self, paths: list[Path]) -> None:
        self.scene_paths = paths
        self.path_by_label = {scene_label(path, self.scene_root): path for path in paths}

    def _size_slider(self, label: str):
        return self.server.gui.add_slider(label, min=0.02, max=10.0, step=0.01, initial_value=0.6)

    @staticmethod
    def _annotation_label(annotation) -> str:
        return f"{annotation.name} [{annotation.id}]"

    @staticmethod
    def _selected_id_from_label(value: str) -> str | None:
        if value == "None" or "[" not in value or not value.endswith("]"):
            return None
        return value.rsplit("[", 1)[1][:-1]
