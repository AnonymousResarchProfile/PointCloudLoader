from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCENE_CLOUD_FILE = "cloud.ply"
ANNOTATOR_NOTES_FILE = "annotator_notes.txt"
REGION_BOXES_FILE = "region_boxes.npz"

MIN_ANNOTATION_EDGE = 0.02


@dataclass
class Annotation:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    name: str = "region"
    center_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    size_xyz: np.ndarray = field(
        default_factory=lambda: np.array([0.6, 0.6, 0.6], dtype=np.float64)
    )
    orientation_wxyz: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )

@dataclass(frozen=True)
class SceneManifest:
    scene_dir: Path
    annotations_asset: str = REGION_BOXES_FILE
    annotator_notes: str = ""

    @property
    def cloud_path(self) -> Path:
        return self.scene_dir / SCENE_CLOUD_FILE

    @property
    def annotations_path(self) -> Path:
        return self.scene_dir / self.annotations_asset


@dataclass
class LoadedScene:
    manifest: SceneManifest
    points: np.ndarray
    colors: np.ndarray | None
    annotations: list[Annotation]


@dataclass
class ViewerPrefs:
    annotations_visible: bool = True


@dataclass
class EditorState:
    scene: LoadedScene | None = None
    prefs: ViewerPrefs = field(default_factory=ViewerPrefs)
    selected_annotation_id: str | None = None
    dirty_annotations: bool = False


@dataclass(frozen=True)
class EditorChange:
    kind: str
    message: str = ""
