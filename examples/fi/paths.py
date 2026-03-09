from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """
    Return the repository root as an absolute Path.

    Assumes this file lives at <repo>/common/fi/paths.py.
    """
    return Path(__file__).resolve().parents[2]


def repo_path(*parts: str) -> Path:
    """Convenience helper to build a path relative to repo root."""
    return repo_root().joinpath(*parts)


