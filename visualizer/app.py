from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import viser

from .editor import SceneEditor
from .viser_gui import ViserGuiAdapter, discover_scene_paths
from .viser_scene import ViserSceneAdapter

log = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone scene viewer with 3D region editing."
    )
    parser.add_argument(
        "scene_path",
        type=Path,
        help="Scene directory, or a folder of scene directories.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_arg_parser().parse_args()

    scene_root = args.scene_path.expanduser().resolve()
    scene_paths = discover_scene_paths(scene_root)
    if not scene_paths:
        raise FileNotFoundError(f"No scenes found under {scene_root}")

    editor = SceneEditor()
    editor.load_scene(scene_paths[0])

    server = viser.ViserServer(host=args.host, port=args.port)
    log.info("Viser server started at http://localhost:%s", args.port)

    renderer = ViserSceneAdapter(server, editor)
    renderer.refresh_all()
    ViserGuiAdapter(
        server,
        editor,
        renderer,
        scene_root=scene_root,
        scene_paths=scene_paths,
    )

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
