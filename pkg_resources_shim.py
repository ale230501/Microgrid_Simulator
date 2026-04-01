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
