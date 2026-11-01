from typing import Any

from .backend_base import *
from .backend_scipy import ScipyBackend
from .backend_highspy import HighspyBackend


def make_backend(name: str) -> SolverBackend[Any]:
    if name == "scipy":
        return ScipyBackend()
    if name == "highspy":
        return HighspyBackend()
    raise ValueError(f"unknown backend {name!r}")
