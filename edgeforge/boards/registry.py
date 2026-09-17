"""Discovers and loads board profiles from a directory of YAML files.

Adding a new board to EdgeForge means dropping a new ``<id>.yaml`` file into
this directory -- nothing here needs to change, and nothing in codegen/build/
validate needs to change either. ``tests/test_add_board_no_code_changes.py``
exercises exactly this claim for ``esp32_native``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional

import yaml

from edgeforge.boards.schema import BoardProfile, parse_board_profile
from edgeforge.errors import BoardError

_ENV_VAR = "EDGEFORGE_BOARDS_DIR"


def default_boards_dir() -> Path:
    """The repo-root ``boards/`` directory, unless overridden by $EDGEFORGE_BOARDS_DIR."""
    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    # edgeforge/boards/registry.py -> edgeforge/boards -> edgeforge -> <repo root>
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "boards"


class BoardRegistry:
    """Loads and caches every board profile found in a directory."""

    def __init__(self, boards_dir: Optional[Path] = None):
        self.boards_dir = Path(boards_dir) if boards_dir is not None else default_boards_dir()
        self._cache: dict[str, BoardProfile] = {}
        self._loaded_all = False

    def _yaml_files(self) -> list[Path]:
        if not self.boards_dir.is_dir():
            raise BoardError(
                f"board registry directory not found: {self.boards_dir}\n"
                f"Set {_ENV_VAR} to point at a directory of board YAML files, "
                f"or run from within the EdgeForge repo checkout."
            )
        return sorted(self.boards_dir.glob("*.yaml")) + sorted(self.boards_dir.glob("*.yml"))

    def list_ids(self) -> list[str]:
        return [p.stem for p in self._yaml_files()]

    def _load_file(self, path: Path) -> BoardProfile:
        try:
            raw = yaml.safe_load(path.read_text())
        except yaml.YAMLError as e:
            raise BoardError(f"board file '{path}' is not valid YAML: {e}") from e
        if raw is None:
            raise BoardError(f"board file '{path}' is empty")
        return parse_board_profile(raw, path)

    def get(self, board_id: str) -> BoardProfile:
        if board_id in self._cache:
            return self._cache[board_id]
        candidates = {p.stem: p for p in self._yaml_files()}
        if board_id not in candidates:
            available = ", ".join(sorted(candidates)) or "(none found)"
            raise BoardError(
                f"unknown board id '{board_id}'. Available boards in {self.boards_dir}: {available}"
            )
        profile = self._load_file(candidates[board_id])
        self._cache[board_id] = profile
        return profile

    def all(self) -> dict[str, BoardProfile]:
        if not self._loaded_all:
            for path in self._yaml_files():
                if path.stem not in self._cache:
                    self._cache[path.stem] = self._load_file(path)
            self._loaded_all = True
        return dict(self._cache)

    def __iter__(self) -> Iterator[BoardProfile]:
        return iter(self.all().values())


def load_board(board_id: str, boards_dir: Optional[Path] = None) -> BoardProfile:
    """Convenience one-shot loader: resolve+validate a single board by id."""
    return BoardRegistry(boards_dir).get(board_id)
