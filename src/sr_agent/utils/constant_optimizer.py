# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Structure-aware numerical optimization for constants in nd2py expressions.

The optimizer implements a practical subset of FunctionEvolve's coefficient
fitting pipeline: variable projection for jointly affine parameters, bounded
multi-start nonlinear least squares, differential-evolution fallback,
L-BFGS-B polishing, and snap-and-refit for exponents.  It deliberately works
on a deep copy so callers can safely reuse the input expression.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict
import warnings

import nd2py as nd
import numpy as np
from scipy.optimize import differential_evolution, least_squares, minimize


@dataclass(frozen=True)
class ConstantOptimizerConfig:
    """Numerical budget and reproducibility controls for ``fit_constants``."""

    random_state: int = 0
    n_restarts: int = 3
    max_nfev: int = 700
    differential_evolution_maxiter: int = 15
    differential_evolution_popsize: int = 5
    use_global_search: bool = True
    snap_exponents: bool = True
    use_eps: float = 1e-8


@dataclass
class _TreeInfo:
    parameters: set[int]
    affine_candidates: set[int]
    conflicts: set[tuple[int, int]]


class _ParameterVector:
    def __init__(self, expression: nd.Symbol, data: Dict[str, np.ndarray]):
        self.expression = expression
        # GroupedParameter values are materialized by evaluation, just as in
        # nd2py.BFGSFit.  Ordinary formula evaluation uses Number nodes only,
        # but retaining this support makes the utility generally usable.
        for node in expression.iter_preorder():
            if isinstance(node, nd.GroupedParameter):
                node.bind(node.by.eval(vars=data))

        self.nodes = [
            node for node in expression.iter_preorder()
            if isinstance(node, (nd.Number, nd.GroupedParameter)) and node.fitable
        ]
        sizes = [int(np.size(node.value)) for node in self.nodes]
        self.slices = [slice(start, start + size) for start, size in zip(np.cumsum([0, *sizes[:-1]]), sizes)]
        self.node_index = {id(node): index for index, node in enumerate(self.nodes)}

    @property
    def size(self) -> int:
        return sum(item.stop - item.start for item in self.slices)

    def get(self) -> np.ndarray:
        if not self.nodes:
            return np.empty(0, dtype=float)
        return np.concatenate([
            np.asarray(node.value, dtype=float).reshape(-1) for node in self.nodes
        ])

    def set(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        for node, item in zip(self.nodes, self.slices):
            replacement = values[item].reshape(np.shape(node.value))
            node.value = float(replacement) if replacement.ndim == 0 else replacement.copy()

    def component_indices(self, node_indices: set[int]) -> np.ndarray:
        result = []
        for node_index in sorted(node_indices):
            item = self.slices[node_index]
            result.extend(range(item.start, item.stop))
        return np.asarray(result, dtype=int)


def _structure_info(node: nd.Symbol, node_index: dict[int, int]) -> _TreeInfo:
    """Find parameters that are affine when all other parameters are fixed.

    Multiplication creates a conflict between affine parameters on its two
    sides: either side can be projected, but both cannot be projected jointly.
    Parameters below nonlinear functions, powers, or denominators are kept as
    nonlinear variables.  This analysis is conservative by design.
    """
    if id(node) in node_index:
        index = node_index[id(node)]
        return _TreeInfo({index}, {index}, set())
    if not node.operands:
        return _TreeInfo(set(), set(), set())

    children = [_structure_info(child, node_index) for child in node.operands]
    parameters = set().union(*(child.parameters for child in children))
    conflicts = set().union(*(child.conflicts for child in children))
    name = type(node).__name__

    if name in {"Add", "Sub"}:
        candidates = set().union(*(child.affine_candidates for child in children))
    elif name == "Mul":
        left, right = children
        candidates = left.affine_candidates | right.affine_candidates
        for left_index in left.affine_candidates:
            for right_index in right.affine_candidates:
                if left_index == right_index:
                    candidates.discard(left_index)
                else:
                    conflicts.add(tuple(sorted((left_index, right_index))))
    elif name == "Div":
        # Numerator coefficients remain affine with denominator coefficients
        # treated as nonlinear shape parameters.
        candidates = set(children[0].affine_candidates)
    elif name in {"Identity", "Neg"}:
        candidates = set(children[0].affine_candidates)
    else:
        candidates = set()
    return _TreeInfo(parameters, candidates, conflicts)


def _maximum_independent_set(candidates: set[int], conflicts: set[tuple[int, int]]) -> set[int]:
    candidates = sorted(candidates)
    if len(candidates) <= 18:
        conflict_set = {tuple(sorted(pair)) for pair in conflicts}
        for size in range(len(candidates), 0, -1):
            for group in combinations(candidates, size):
                if all((group[i], group[j]) not in conflict_set
                       for i in range(size) for j in range(i + 1, size)):
                    return set(group)
        return set()

    adjacency = {candidate: set() for candidate in candidates}
    for left, right in conflicts:
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)
    selected = set()
    for candidate in sorted(candidates, key=lambda item: (len(adjacency[item]), item)):
        if not (adjacency[candidate] & selected):
            selected.add(candidate)
    return selected


def _parameter_bounds(
    parameters: _ParameterVector,
    initial: np.ndarray,
    data: Dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.empty_like(initial)
    upper = np.empty_like(initial)
    for node, item in zip(parameters.nodes, parameters.slices):
        is_exponent = (
            node.parent is not None
            and type(node.parent).__name__ == "Pow"
            and node.parent.operands[1] is node
        )
        base_limit = 10.0 if is_exponent else 100.0
        values = np.abs(initial[item])
        limits = np.maximum(base_limit, 10.0 * values + 1.0)
        lower[item] = -limits
        upper[item] = limits

        # Additive constants immediately inside log(x + c) protect the real
        # domain and should not cross zero during global exploration.
        parent = node.parent
        if (parent is not None and type(parent).__name__ == "Add"
                and parent.parent is not None
                and type(parent.parent).__name__ in {"Log", "LogAbs"}):
            lower[item] = 1e-6
            upper[item] = limits

        # Centers and offsets in forms such as (x - c) should be searched on
        # the observed feature domain instead of the generic coefficient box.
        if parent is not None and type(parent).__name__ == "Sub":
            siblings = [operand for operand in parent.operands if operand is not node]
            if len(siblings) == 1 and isinstance(siblings[0], nd.Variable):
                feature = np.asarray(data[siblings[0].name], dtype=float).reshape(-1)
                feature = feature[np.isfinite(feature)]
                if feature.size:
                    feature_range = float(np.max(feature) - np.min(feature))
                    margin = max(0.2 * feature_range, 10.0)
                    lower[item] = float(np.min(feature) - margin)
                    upper[item] = float(np.max(feature) + margin)
    return lower, upper


def _as_prediction(expression: nd.Symbol, data: Dict[str, np.ndarray], size: int, use_eps: float) -> np.ndarray | None:
    try:
        with np.errstate(all="ignore"):
            prediction = np.asarray(expression.eval(vars=data, use_eps=use_eps), dtype=float).reshape(-1)
        if prediction.size == 1 and size != 1:
            prediction = np.full(size, float(prediction[0]))
        if prediction.size != size or not np.all(np.isfinite(prediction)):
            return None
        return prediction
    except (ArithmeticError, FloatingPointError, KeyError, TypeError, ValueError, OverflowError):
        return None


def fit_constants(
    f: nd.Symbol,
    data: Dict[str, np.ndarray],
    y: np.ndarray,
    *,
    config: ConstantOptimizerConfig | None = None,
) -> nd.Symbol:
    """Return a fitted copy of ``f`` without modifying ``f``.

    Args:
        f: An nd2py expression whose fitable Number/GroupedParameter nodes are
            treated as coefficients.
        data: Mapping from variable names to aligned NumPy arrays.
        y: One-dimensional regression target.
        config: Optional numerical-budget and random-seed configuration.

    Returns:
        A deep-copied expression containing the best fitted coefficients found.
    """
    if not isinstance(f, nd.Symbol):
        raise TypeError("f must be an nd2py.Symbol")
    config = config or ConstantOptimizerConfig()
    expression = f.copy()
    target = np.asarray(y, dtype=float).reshape(-1)
    if target.size == 0 or not np.all(np.isfinite(target)):
        raise ValueError("y must contain at least one finite-only sample")
    arrays = {name: np.asarray(value) for name, value in data.items()}
    if any(np.asarray(value).reshape(-1).size != target.size for value in arrays.values()):
        raise ValueError("all data arrays must have the same number of samples as y")

    parameters = _ParameterVector(expression, arrays)
    if parameters.size == 0:
        return expression

    initial = parameters.get()
    lower, upper = _parameter_bounds(parameters, initial, arrays)
    initial = np.clip(initial, lower, upper)
    scale = max(float(np.std(target)), float(np.max(np.abs(target))) * 1e-12, 1e-12)
    target_mse = max(1e-24, 1e-10 * float(np.var(target)))
    invalid_residual = np.full(target.size, 1e6, dtype=float)
    rng = np.random.default_rng(config.random_state)

    def prediction(values: np.ndarray) -> np.ndarray | None:
        parameters.set(values)
        return _as_prediction(expression, arrays, target.size, config.use_eps)

    def residual(values: np.ndarray) -> np.ndarray:
        predicted = prediction(values)
        return invalid_residual if predicted is None else (predicted - target) / scale

    def mse(values: np.ndarray) -> float:
        predicted = prediction(values)
        if predicted is None:
            return float("inf")
        with np.errstate(all="ignore"):
            value = float(np.mean((predicted - target) ** 2))
        return value if np.isfinite(value) else float("inf")

    best = initial.copy()
    best_mse = mse(best)

    # Determine a maximal set of parameters that can be jointly eliminated by
    # ordinary least squares for every fixed nonlinear-parameter assignment.
    tree_info = _structure_info(expression, parameters.node_index)
    linear_nodes = _maximum_independent_set(tree_info.affine_candidates, tree_info.conflicts)
    linear_indices = parameters.component_indices(linear_nodes)
    nonlinear_indices = np.asarray(
        [index for index in range(parameters.size) if index not in set(linear_indices)], dtype=int
    )

    def project(nonlinear_values: np.ndarray) -> tuple[np.ndarray | None, float]:
        values = initial.copy()
        if nonlinear_indices.size:
            values[nonlinear_indices] = nonlinear_values
        values[linear_indices] = 0.0
        fixed = prediction(values)
        if fixed is None:
            return None, float("inf")
        basis = []
        for index in linear_indices:
            trial = values.copy()
            trial[index] = 1.0
            column = prediction(trial)
            if column is None:
                return None, float("inf")
            basis.append(column - fixed)
        if basis:
            matrix = np.column_stack(basis)
            try:
                coefficients = np.linalg.lstsq(matrix, target - fixed, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None, float("inf")
            values[linear_indices] = coefficients
        value = mse(values)
        return values, value

    def projected_residual(nonlinear_values: np.ndarray) -> np.ndarray:
        values, _ = project(nonlinear_values)
        return invalid_residual if values is None else residual(values)

    if linear_indices.size:
        projected, projected_mse = project(initial[nonlinear_indices])
        if projected is not None and projected_mse < best_mse:
            best, best_mse = projected, projected_mse

        if nonlinear_indices.size:
            nonlinear_lower, nonlinear_upper = lower[nonlinear_indices], upper[nonlinear_indices]
            starts = [initial[nonlinear_indices]]
            for restart in range(max(0, config.n_restarts - 1)):
                if restart == 0:
                    perturbation = rng.normal(0.0, 0.25, nonlinear_indices.size)
                    starts.append(np.clip(
                        starts[0] + perturbation * np.maximum(1.0, np.abs(starts[0])),
                        nonlinear_lower,
                        nonlinear_upper,
                    ))
                else:
                    starts.append(rng.uniform(nonlinear_lower, nonlinear_upper))
            for start in starts:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result = least_squares(
                        projected_residual, start, bounds=(nonlinear_lower, nonlinear_upper),
                        x_scale="jac", max_nfev=config.max_nfev,
                    )
                candidate, candidate_mse = project(result.x)
                if candidate is not None and candidate_mse < best_mse:
                    best, best_mse = candidate, candidate_mse
                if best_mse <= target_mse:
                    break

            if config.use_global_search and best_mse > target_mse:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result = differential_evolution(
                        lambda values: project(values)[1],
                        list(zip(nonlinear_lower, nonlinear_upper)),
                        seed=config.random_state,
                        maxiter=config.differential_evolution_maxiter,
                        popsize=config.differential_evolution_popsize,
                        polish=False,
                        updating="immediate",
                    )
                candidate, candidate_mse = project(result.x)
                if candidate is not None and candidate_mse < best_mse:
                    best, best_mse = candidate, candidate_mse

    # OLS coefficients are intentionally unconstrained.  Expand only their
    # later full-search bounds when projection found values outside the generic
    # box; nonlinear shape-parameter bounds remain structurally constrained.
    if linear_indices.size:
        padding = np.maximum(1.0, 0.1 * np.abs(best[linear_indices]))
        lower[linear_indices] = np.minimum(lower[linear_indices], best[linear_indices] - padding)
        upper[linear_indices] = np.maximum(upper[linear_indices], best[linear_indices] + padding)

    # Full-parameter search is both a fallback when no decomposition exists
    # and a complementary path when the projected search remains insufficient.
    if not linear_indices.size or best_mse > target_mse:
        starts = [np.clip(initial, lower, upper), np.clip(best, lower, upper)]
        for restart in range(max(0, config.n_restarts - 1)):
            if restart == 0:
                starts.append(np.clip(
                    initial
                    + rng.normal(0.0, 0.25, parameters.size)
                    * np.maximum(1.0, np.abs(initial)),
                    lower,
                    upper,
                ))
            else:
                starts.append(rng.uniform(lower, upper))
        for start in starts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = least_squares(
                    residual, start, bounds=(lower, upper), x_scale="jac",
                    max_nfev=config.max_nfev,
                )
            candidate_mse = mse(result.x)
            if candidate_mse < best_mse:
                best, best_mse = result.x.copy(), candidate_mse
            if best_mse <= target_mse:
                break

        if config.use_global_search and best_mse > target_mse and parameters.size <= 16:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = differential_evolution(
                    mse, list(zip(lower, upper)), seed=config.random_state,
                    maxiter=config.differential_evolution_maxiter,
                    popsize=config.differential_evolution_popsize,
                    polish=False, updating="immediate",
                )
            candidate_mse = mse(result.x)
            if candidate_mse < best_mse:
                best, best_mse = result.x.copy(), candidate_mse

    # L-BFGS-B provides a cheap final polish from the best basin found above.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        polished = minimize(
            mse,
            np.clip(best, lower, upper),
            method="L-BFGS-B",
            bounds=list(zip(lower, upper)),
            options={"maxiter": config.max_nfev},
        )
    polished_mse = mse(polished.x)
    if polished_mse < best_mse:
        best, best_mse = polished.x.copy(), polished_mse

    # Snap power exponents to stable low-denominator values, then refit all
    # remaining coefficients.  A snap is retained only when it does not worsen
    # the objective beyond floating-point tolerance.
    if config.snap_exponents:
        grid = np.asarray([
            -5, -4, -3, -2, -1.5, -1, -2 / 3, -0.5, -1 / 3,
            0, 1 / 3, 0.5, 2 / 3, 1, 1.5, 2, 3, 4, 5,
        ], dtype=float)
        exponent_indices = []
        for node, item in zip(parameters.nodes, parameters.slices):
            if (node.parent is not None and type(node.parent).__name__ == "Pow"
                    and node.parent.operands[1] is node):
                exponent_indices.extend(range(item.start, item.stop))
        for exponent_index in exponent_indices:
            nearby = grid[np.abs(grid - best[exponent_index]) <= 3.0]
            nearby = sorted(nearby, key=lambda value: abs(value - best[exponent_index]))[:6]
            free = np.asarray([index for index in range(parameters.size) if index != exponent_index], dtype=int)
            for snapped in nearby:
                trial = best.copy()
                trial[exponent_index] = snapped
                if free.size:
                    def snap_residual(values: np.ndarray) -> np.ndarray:
                        candidate = trial.copy()
                        candidate[free] = values
                        return residual(candidate)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        result = least_squares(
                            snap_residual, np.clip(trial[free], lower[free], upper[free]),
                            bounds=(lower[free], upper[free]),
                            x_scale="jac", max_nfev=max(100, config.max_nfev // 3),
                        )
                    trial[free] = result.x
                trial_mse = mse(trial)
                tolerance = max(1e-18, best_mse * 1e-10)
                if trial_mse <= best_mse + tolerance:
                    best, best_mse = trial, trial_mse
                    break

    parameters.set(best)
    return expression


__all__ = ["ConstantOptimizerConfig", "fit_constants"]
