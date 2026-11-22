"""Scenario models. Every model is scored by the same code in `scenario_lab.validation`.

Interface (what a model has to provide; a new model, e.g. the neural SDE, only has to match this):

    fit(changes: DataFrame) -> params
        changes: daily yield changes (T, n_maturities), percent, chronological. Use ONLY this data
        (no look-ahead): the validation calls it with data up to each forecast origin.

    simulate(params, y0, n_paths, n_days, seed) -> ndarray (n_paths, n_days + 1, n_maturities)
        Yield curves in percent; [:, 0] equals y0, the curve at the forecast origin.

Register the model in MODELS below to include it in the comparison notebook.
"""
from typing import Any, Callable, NamedTuple

from scenario_lab.models import var1, var1_garch


class Model(NamedTuple):
    name: str
    fit: Callable[..., Any]
    simulate: Callable[..., Any]


MODELS = {
    "var1": Model("var1", var1.fit, var1.simulate),
    "var1_garch": Model("var1_garch", var1_garch.fit, var1_garch.simulate),
}
