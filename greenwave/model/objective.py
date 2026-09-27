"""Objective assembly helpers.

The actual objective bookkeeping is performed by
:meth:`greenwave.model.builder.ArterialModel.expr_composite` and
:meth:`ArterialModel.expr_loss`.  This module exists to keep the public
architecture aligned with the design document and to provide a small
functional facade.
"""
from __future__ import annotations

from .builder import ArterialModel


def composite_objective(model: ArterialModel):
    return model.expr_composite()


def intersection_loss_objective(model: ArterialModel):
    return model.expr_loss()


__all__ = ["composite_objective", "intersection_loss_objective"]
