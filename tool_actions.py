"""Cross-platform tool registry and action implementations.

Every tool is a callable that receives keyword arguments extracted from
user text and returns a dict with at least a ``reply`` key shown to the
user.  The registry maps intent tags to tools so the assistant can look
them up after classifying a request.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

BASE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Tool data-class and registry
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    tag: str
    description: str
    handler: Callable[..., dict[str, Any]]
    parameters: list[str] = field(default_factory=list)


class ToolRegistry:
    """Stores tools by intent tag and by name."""

    def __init__(self) -> None:
        self._by_tag: dict[str, Tool] = {}
        self._by_name: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._by_tag[tool.tag] = tool
        self._by_name[tool.name] = tool

    def get(self, tag: str) -> Tool | None:
        return self._by_tag.get(tag)

    def get_by_name(self, name: str) -> Tool | None:
        return self._by_name.get(name)

    def all_tools(self) -> list[Tool]:
        return list(self._by_tag.values())


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform == "linux"
APP_DIR = BASE / "data" / "apps"


def _app_command(name: str) -> list[str] | None:
    """Return the platform-specific command to open *name*.

    Returns ``None`` if the app is not supported on this platform.
    """
    name = name.strip().casefold()
    mapping: dict[str, dict[str, Any]] = {
        "notepad": {
            "win32": [str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "notepad.exe")],
        },
        "calculator": {
            "win32": [str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "calc.exe")],
            "linux": ["xdg-open", "calculator:"],
        },
    }
    if name not in mapping:
        return None
    entry = mapping[name]
    if IS_WINDOWS and "win32" in entry:
        return entry["win32"]
    if IS_LINUX and "linux" in entry:
        return entry["linux"]
    return None


def open_application(app_name: str) -> dict[str, Any]:
    """Launch a whitelisted application by name."""
    cmd = _app_command(app_name)
    if cmd is None:
        available = ", ".join(sorted(
            n for n, e in {
                "notepad": {"win32": 1},
                "calculator": {"win32": 1, "linux": 1},
            }.items()
            if (IS_WINDOWS and "win32" in e) or (IS_LINUX and "linux" in e)
        ))
        return {"reply": f"App not available on this platform. Supported: {available}.", "ok": False}
    subprocess.Popen(cmd, shell=False)
    return {"reply": f"Opened {app_name}.", "ok": True}


def search_files(folder: str, query: str, limit: int = 200) -> dict[str, Any]:
    """Walk *folder* and return paths matching *query* (case-insensitive)."""
    root = Path(folder).resolve()
    if not root.is_dir():
        return {"reply": "Folder does not exist.", "ok": False, "paths": [], "limited": False}
    if not query.strip():
        return {"reply": "Provide a filename or part of a filename.", "ok": False, "paths": [], "limited": False}
    skip = {".git", "node_modules", ".venv", "__pycache__"}
    results: list[str] = []
    visited = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in skip
                   and not Path(directory, d).is_symlink()
                   and not Path(directory, d).is_junction()]
        for fname in files:
            visited += 1
            if query.casefold() in fname.casefold():
                path = Path(directory, fname).resolve()
                if path.is_relative_to(root):
                    results.append(str(path))
            if len(results) >= limit or visited >= 50000:
                return {"reply": "\n".join(results) + ("\nSearch limit reached." if True else ""),
                        "ok": True, "paths": results, "limited": True}
    return {"reply": "\n".join(results) or "No matching files found.",
            "ok": True, "paths": results, "limited": False}


def get_system_info() -> dict[str, Any]:
    """Return basic system information."""
    info = {
        "platform": platform.system(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    return {"reply": f"{info['platform']} | Python {info['python']} | {info['machine']}", "ok": True, **info}


def get_time() -> dict[str, Any]:
    """Return the current date and time."""
    from datetime import datetime
    now = datetime.now().astimezone()
    return {"reply": now.strftime("%A, %d %B %Y, %H:%M %Z"), "ok": True}


def list_directory(path: str = ".") -> dict[str, Any]:
    """List contents of a directory."""
    target = Path(path).resolve()
    if not target.is_dir():
        return {"reply": f"Not a directory: {path}", "ok": False}
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    if not entries:
        return {"reply": "Directory is empty.", "ok": True, "entries": []}
    return {"reply": "\n".join(entries[:100]), "ok": True, "entries": entries}


def inspect_python(source: str) -> dict[str, Any]:
    """Parse Python source and report functions/classes."""
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"reply": f"Python syntax error on line {exc.lineno}: {exc.msg}", "ok": False}
    functions = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    return {
        "reply": f"Python syntax is valid.\nFunctions: {', '.join(functions) or 'none'}\nClasses: {', '.join(classes) or 'none'}",
        "ok": True,
        "functions": functions,
        "classes": classes,
    }


# ---------------------------------------------------------------------------
# Default registry with built-in tools
# ---------------------------------------------------------------------------

def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(
        name="open_app", tag="open_app",
        description="Open a desktop application (notepad, calculator)",
        handler=open_application, parameters=["app_name"],
    ))
    registry.register(Tool(
        name="find_file", tag="find_file",
        description="Search for files by name in a folder",
        handler=search_files, parameters=["folder", "query"],
    ))
    registry.register(Tool(
        name="system_info", tag="system_info",
        description="Show platform and Python version",
        handler=get_system_info, parameters=[],
    ))
    registry.register(Tool(
        name="get_time", tag="get_time",
        description="Show current date and time",
        handler=get_time, parameters=[],
    ))
    registry.register(Tool(
        name="list_dir", tag="list_dir",
        description="List contents of a directory",
        handler=list_directory, parameters=["path"],
    ))
    registry.register(Tool(
        name="inspect_code", tag="inspect_code",
        description="Parse Python source for syntax and definitions",
        handler=inspect_python, parameters=["source"],
    ))
    return registry
