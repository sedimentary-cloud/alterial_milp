"""Objective assembly helpers.

The actual objective bookkeeping is performed by
:meth:`greenwave.model.builder.ArterialModel.expr_composite` and
:meth:`ArterialModel.expr_intersection_cost`.  This module exists to keep the public
architecture aligned with the design document and to provide a small
functional facade.
"""
from __future__ import annotations

from .builder import ArterialModel


def composite_objective(model: ArterialModel):
    return model.expr_composite()


def intersection_reward_objective(model: ArterialModel):
    return model.expr_intersection_reward()


def intersection_loss_objective(model: ArterialModel):
    return model.expr_intersection_loss()


def intersection_composite_objective(model: ArterialModel):
    return model.expr_intersection_composite()


def intersection_cost_objective(model: ArterialModel):
    return model.expr_intersection_cost()


__all__ = ["composite_objective", "intersection_reward_objective",
           "intersection_loss_objective", "intersection_composite_objective",
           "intersection_cost_objective"]
