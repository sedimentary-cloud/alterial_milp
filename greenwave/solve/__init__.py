from typing import Any

from .backend_base import *
from ..errors import GWInputError
from .backend_scipy import ScipyBackend
from .backend_highspy import HighspyBackend


def make_backend(name: str) -> SolverBackend[Any]:
    if name == "scipy":
        return ScipyBackend()
    if name == "highspy":
        return HighspyBackend()
    raise GWInputError(f"unknown backend {name!r}",
                       hint="backend 必须是 'scipy' 或 'highspy'")
