from __future__ import annotations

from pathlib import Path

from .api import Chronicle


class ChroniclePipeline(Chronicle):
    def __init__(self, repo_path: str | Path, index_dir: str | Path | None = None, **_: object) -> None:
        super().__init__(repo_path=repo_path, index_dir=index_dir)
