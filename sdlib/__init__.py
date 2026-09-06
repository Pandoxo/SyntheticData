"""Shared building blocks for the synthetic trash dataset generators.

Both entry points (`generate.py` for a bare ground plane, `generate_scene.py` for
a real .blend environment) differ only in how they build the environment; every
other stage - asset loading, drop physics, camera sampling, annotation writing -
lives here so the two datasets stay comparable.

These modules import `blenderproc`, so they can only be imported from a script
launched via `blenderproc run`.
"""
