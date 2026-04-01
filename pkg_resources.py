"""Minimal compatibility shim for projects expecting pkg_resources.

This project uses pandasgui, which still imports ``pkg_resources`` APIs that
were removed from newer setuptools releases.
"""

from __future__ import annotations

from importlib import import_module, metadata
from pathlib import Path


class DistributionNotFound(Exception):
    """Raised when a distribution cannot be found."""


class _Distribution:
    def __init__(self, project_name: str, version: str) -> None:
        self.project_name = project_name
        self.version = version


def get_distribution(dist_name: str) -> _Distribution:
    try:
        version = metadata.version(dist_name)
    except metadata.PackageNotFoundError as exc:
        raise DistributionNotFound(str(exc)) from exc
    return _Distribution(project_name=dist_name, version=version)


def resource_filename(package_or_requirement: str, resource_name: str) -> str:
    module = import_module(package_or_requirement)
    module_path = getattr(module, "__file__", None)
    module_paths = getattr(module, "__path__", None)

    if module_paths:
        base = Path(next(iter(module_paths)))
    elif module_path:
        base = Path(module_path).resolve().parent
    else:
        base = Path.cwd()

    return str((base / resource_name).resolve())


class _CompatEntryPoint:
    """Adapter for importlib.metadata.EntryPoint to mimic pkg_resources API."""

    def __init__(self, entry_point):
        self._entry_point = entry_point
        self.name = getattr(entry_point, "name", None)
        self.group = getattr(entry_point, "group", None)
        self.value = getattr(entry_point, "value", None)

    def load(self):
        return self._entry_point.load()

    def resolve(self):
        # pkg_resources.EntryPoint exposes resolve(); TensorBoard still uses it.
        return self.load()

    def __getattr__(self, item):
        return getattr(self._entry_point, item)


def iter_entry_points(group: str, name: str | None = None):
    """Compatibility shim used by tools like TensorBoard."""
    try:
        entry_points = metadata.entry_points()
    except Exception:
        return iter(())

    if hasattr(entry_points, "select"):
        selected = entry_points.select(group=group)
    else:
        selected = entry_points.get(group, ())

    for ep in selected:
        if name is None or getattr(ep, "name", None) == name:
            yield _CompatEntryPoint(ep)
