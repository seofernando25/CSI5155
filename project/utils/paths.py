from __future__ import annotations

from pathlib import Path


def require_file(path: Path | str, hint: str | None = None) -> Path:
    path_obj = Path(path) if isinstance(path, str) else path
    if not path_obj.exists():
        msg = f"File not found at {path_obj}"
        if hint:
            msg += f". {hint}"
        raise FileNotFoundError(msg)
    return path_obj

