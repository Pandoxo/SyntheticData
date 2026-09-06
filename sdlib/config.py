"""Config loading and path resolution."""

import os

import yaml

# Paths in the config files are written relative to the project root so that the
# generators behave the same no matter which directory they were launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path):
    """Expand `~` and anchor relative paths at the project root, not the cwd."""
    if path is None:
        return None
    path = os.path.expanduser(str(path))
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    return os.path.normpath(path)


def resolve_all(paths):
    return [resolve(p) for p in (paths or [])]
