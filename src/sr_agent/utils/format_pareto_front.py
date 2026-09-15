from __future__ import annotations
import pandas as pd
from typing import Literal
from collections.abc import Sequence
from .df_to_3line import df_to_3line


ObjectiveDirection = Literal["minimize", "maximize"]
Objective = tuple[str, ObjectiveDirection]

DEFAULT_PARETO_BALANCE: tuple[Objective, Objective] = (
    ("validation.metrics.complexity", "minimize"),
    ("validation.metrics.r2", "maximize"),
)
DEFAULT_PARETO_COLUMNS: tuple[str, ...] = (
    "validation.metrics.complexity",
    "validation.metrics.r2",
    "train.metrics.r2",
    "formula",
)
DEFAULT_TITLE_MAPPING: dict[str, str] = {
    "validation.metrics.complexity": "Complexity",
    "validation.metrics.r2": "Validation R²",
    "train.metrics.r2": "Train R²",
    "formula": "Formula",
}


def _record_value(record: dict, path: str):
    """Read a dotted field, treating split names as data_split_results children."""
    parts = path.split(".")
    value = record
    if parts[0] in {"train", "validation"}:
        value = value.get("data_split_results", {})
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _format_value(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _truncate_formula(formula, formula_max_length: int | None) -> str:
    formula = str(formula)
    if formula_max_length is None or len(formula) <= formula_max_length:
        return formula
    if formula_max_length < 4:
        raise ValueError("formula_max_length must be at least 4 or None.")
    return formula[: formula_max_length - 3] + "..."


def _validate_balance(balance: Sequence[Objective]) -> tuple[Objective, Objective]:
    if len(balance) != 2:
        raise ValueError("balance must contain exactly two objectives.")
    normalized = tuple(balance)
    for path, direction in normalized:
        if not path:
            raise ValueError("Pareto objective paths cannot be empty.")
        if direction not in {"minimize", "maximize"}:
            raise ValueError(f"Unsupported Pareto objective direction: {direction!r}.")
    return normalized  # type: ignore[return-value]


def format_pareto_front(
    pareto_front: list[dict] | None,
    concise: bool = False,
    formula_max_length: int | None = None,
    balance: Sequence[Objective] = DEFAULT_PARETO_BALANCE,
    columns: Sequence[str] = DEFAULT_PARETO_COLUMNS,
    title_mapping: dict[str, str] = DEFAULT_TITLE_MAPPING,
) -> str:
    """Format candidate records from ``SRAgent.get_pareto_front``.

    Dotted split paths such as ``validation.metrics.r2`` are resolved below
    each record's ``data_split_results`` mapping. ``formula`` and other
    top-level fields are resolved directly from the record.

    Args:
        - balance: records the two objectives used to construct the supplied
        front and their directions. Set ``concise=True`` for an LLM-facing list;
        the default renders a richer three-line table for human-facing logs.
        - formula_max_length: preserves full formulas, while an integer
        truncates formulas to that total length, including the ellipsis.
        - title_mapping: maps column names to their display titles.
    """
    balance = _validate_balance(balance)
    columns = tuple(columns)
    if not columns:
        raise ValueError("columns cannot be empty.")

    if not pareto_front:
        if concise:
            return "(No Pareto front yet, call tools that can return candidate formulas to populate it.)"
        return "(empty)"

    objective_text = "; ".join(
        f"{direction} {title_mapping.get(path, path)}" for path, direction in balance
    )
    rows = []
    for record in pareto_front:
        row = {}
        for column in columns:
            value = _record_value(record, column)
            if column == "formula":
                value = _truncate_formula(value, formula_max_length)
            row[column] = value
        rows.append(row)

    if concise:
        formatted_rows = []
        for index, row in enumerate(rows, 1):
            fields = ", ".join(
                f"{title_mapping.get(column, column)}={_format_value(row[column])}"
                for column in columns
            )
            formatted_rows.append(f"{index}. {fields}")
        return f"Balance: {objective_text}.\n" + "\n".join(formatted_rows)
    else:
        frame = pd.DataFrame(rows, columns=columns)
        frame.index = pd.Index(range(1, len(frame) + 1), name="#")
        frame = frame.rename(columns={column: title_mapping.get(column, column) for column in frame.columns})
        return f"Balance: {objective_text}.\n{df_to_3line(frame, float_format='.6g')}"
