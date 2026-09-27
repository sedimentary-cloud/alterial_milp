from .backend_base import *
from .backend_scipy import ScipyBackend
from .backend_highspy import HighspyBackend


def make_backend(name: str):
    if name == "scipy":
        return ScipyBackend()
    if name == "highspy":
        return HighspyBackend()
    raise ValueError(f"unknown backend {name!r}")
