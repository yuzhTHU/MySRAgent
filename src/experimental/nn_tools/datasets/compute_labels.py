# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Compute property labels from (X, y) data arrays.

**4-class encoding:**
  Monotonicity: 0=non-monotonic/unknown, 1=increasing, 2=decreasing, 3=constant
  Convexity:    0=neither/unknown,       1=convex,      2=concave,    3=affine
  Periodicity:  0=non-periodic,          1=periodic
  Mul-sep:      0=not separable,         1=separable

Supports two label-computation modes:
  1. ``compute_all_labels`` — pure numerical (fast, for any data)
  2. ``compute_all_labels_sympy`` — SymPy symbolic analysis with numerical
     fallback (accurate, requires expression string from nd2py tree)
"""
import numpy as np
import warnings

__all__ = [
    "compute_all_labels",
    "compute_all_labels_sympy",
    "label_signature",
    "coarse_label_key",
    "remap_old_labels",
    "MONO_CLASSES",
    "CONV_CLASSES",
]

MONO_CLASSES = 4  # 0=non-mono/unk, 1=inc, 2=dec, 3=const
CONV_CLASSES = 4  # 0=neither/unk, 1=convex, 2=concave, 3=affine

_OLD_TO_NEW_MONO = {0: 0, 1: 1, 2: 2, 3: 0, 4: 3}
_OLD_TO_NEW_CONV = {0: 0, 1: 1, 2: 2, 3: 0, 4: 3}


def remap_old_labels(old_mono, old_conv):
    """Convert old 5-class labels to new 4-class encoding."""
    mono = np.array([_OLD_TO_NEW_MONO[int(v)] for v in old_mono], dtype=np.int64)
    conv = np.array([_OLD_TO_NEW_CONV[int(v)] for v in old_conv], dtype=np.int64)
    return mono, conv


# ═══════════════════════════════════════════════════════════════════════════════
# SymPy-based label computation (preferred when expression is available)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_labels_sympy(
    expr_str: str,
    X: np.ndarray,
    y: np.ndarray,
    n_vars: int,
    timeout_per_check: float = 2.0,
) -> dict:
    """Hybrid label computation: SymPy for mono/conv, numerical for period/sep.

    Uses SymPy's ``diff`` + ``lambdify`` for monotonicity and convexity (fast,
    ~5 ms per variable). Falls back to pure numerical for periodicity and
    separability which rely on ``sp.periodicity`` / ``sp.simplify`` that can
    hang on complex expressions.

    Args:
        expr_str: Expression string from ``str(nd2py_tree)``
        X: Input data, shape (N, max_var_num). Unused vars zero-padded.
        y: Target values, shape (N,).
        n_vars: Number of active variables.
        timeout_per_check: Not used currently; reserved for future signal-based timeout.

    Returns:
        Same dict format as ``compute_all_labels``.
    """
    mask = np.isfinite(y) & np.all(np.isfinite(X[:, :n_vars]), axis=1)
    X_clean = X[mask, :n_vars]
    y_clean = y[mask]
    if len(y_clean) < 20:
        return _empty_labels(n_vars)

    mono = np.zeros(n_vars, dtype=np.int64)
    conv = np.zeros(n_vars, dtype=np.int64)

    sympy_ok = False
    try:
        import sympy as sp
        var_names = [f"x{i}" for i in range(n_vars)]
        sym_dict = {name: sp.Symbol(name, real=True) for name in var_names}
        local_ns = dict(sym_dict)
        local_ns.update({
            "pi": sp.pi, "e": sp.E,
            "abs": sp.Abs, "log": sp.log, "sqrt": sp.sqrt,
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "exp": sp.exp, "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
            "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
        })
        expr = sp.sympify(expr_str, locals=local_ns)
        sympy_ok = True
    except Exception:
        pass

    if sympy_ok:
        rng = np.random.default_rng(42)
        for i in range(n_vars):
            var_sym = sym_dict[var_names[i]]
            mono[i] = _sympy_monotonicity(expr, var_sym, sym_dict, var_names, i, X_clean, rng)
            conv[i] = _sympy_convexity(expr, var_sym, sym_dict, var_names, i, X_clean, rng)
    else:
        for i in range(n_vars):
            mono[i] = _check_monotonicity(y_clean, X_clean, i)
            conv[i] = _check_convexity(y_clean, X_clean, i)

    period = np.zeros(n_vars, dtype=np.int64)
    for i in range(n_vars):
        period[i] = _check_periodicity(y_clean, X_clean, i)

    mul_sep = _check_multiplicative_separability(y_clean, X_clean)

    return {
        "monotonicity": mono,
        "convexity": conv,
        "periodicity": period,
        "multiplicative_separable": mul_sep,
    }


def _sympy_monotonicity(expr, var_sym, sym_dict, var_names, var_idx, X_clean, rng):
    import sympy as sp
    if var_sym not in expr.free_symbols:
        return 3  # constant
    try:
        d1 = sp.diff(expr, var_sym)
        if d1.is_zero:
            return 3
        return _eval_derivative_sign(d1, sym_dict, var_names, var_idx, X_clean, rng,
                                     pos_label=1, neg_label=2, zero_label=3, default=0)
    except Exception:
        return _numerical_monotonicity(X_clean[:, var_idx], _sorted_y(X_clean, var_idx, X_clean.shape[0]))


def _sympy_convexity(expr, var_sym, sym_dict, var_names, var_idx, X_clean, rng):
    import sympy as sp
    if var_sym not in expr.free_symbols:
        return 3  # affine
    try:
        d2 = sp.diff(expr, var_sym, 2)
        if d2.is_zero:
            return 3
        dirac_terms = d2.atoms(sp.DiracDelta)
        if dirac_terms:
            d2 = d2.xreplace({term: sp.Integer(0) for term in dirac_terms})
            if d2.is_zero:
                return 3
        return _eval_derivative_sign(d2, sym_dict, var_names, var_idx, X_clean, rng,
                                     pos_label=1, neg_label=2, zero_label=3, default=0)
    except Exception:
        return _numerical_convexity(X_clean[:, var_idx], _sorted_y(X_clean, var_idx, X_clean.shape[0]))


def _eval_derivative_sign(deriv_expr, sym_dict, var_names, var_idx, X_clean, rng,
                          pos_label, neg_label, zero_label, default, n_points=200, confidence=0.95):
    """Evaluate a derivative expression numerically and classify its sign."""
    import sympy as sp
    all_syms = [sym_dict[var_names[var_idx]]]
    other_indices = []
    for j, name in enumerate(var_names):
        if j != var_idx:
            all_syms.append(sym_dict[name])
            other_indices.append(j)

    try:
        func = sp.lambdify(all_syms, deriv_expr, modules=["numpy"])
    except Exception:
        return default

    var_col = X_clean[:, var_idx]
    vmin, vmax = float(np.min(var_col)), float(np.max(var_col))
    if vmax - vmin < 1e-12:
        return default
    test_vals = rng.uniform(vmin, vmax, n_points)
    args = [test_vals]
    for j in other_indices:
        col = X_clean[:, j]
        omin, omax = float(np.min(col)), float(np.max(col))
        args.append(rng.uniform(omin, omax, n_points))

    eps = 1e-8
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vals = np.asarray(func(*args), dtype=float)
    except Exception:
        return default

    if vals.ndim == 0 or vals.size == 1:
        vals = np.full(n_points, float(vals.reshape(-1)[0]))
    else:
        vals = vals.flatten()[:n_points]

    valid = np.isfinite(vals)
    if valid.sum() < 10:
        return default
    v = vals[valid]

    pos_frac = np.mean(v > eps)
    neg_frac = np.mean(v < -eps)
    zero_frac = np.mean(np.abs(v) < eps)

    if pos_frac >= confidence:
        return pos_label
    elif neg_frac >= confidence:
        return neg_label
    elif zero_frac >= confidence:
        return zero_label
    return default


def _sympy_periodicity(expr, var_sym):
    import sympy as sp
    if var_sym not in expr.free_symbols:
        return 0
    try:
        p = sp.periodicity(expr, var_sym)
        return 1 if p is not None else 0
    except Exception:
        return 0


def _sympy_multiplicative_sep(expr, sym_dict, var_names):
    import sympy as sp
    active = [v for v in var_names if sym_dict[v] in expr.free_symbols]
    if len(active) <= 1:
        return 1
    try:
        if expr.func == sp.Mul:
            factors = expr.args
            var_groups = []
            for factor in factors:
                fvars = factor.free_symbols & {sym_dict[v] for v in active}
                if fvars:
                    var_groups.append(fvars)
            from itertools import combinations
            for i, j in combinations(range(len(var_groups)), 2):
                if not var_groups[i] & var_groups[j]:
                    return 1

        try:
            log_expr = sp.expand(sp.log(expr))
            for vi, vj in combinations(active, 2):
                mixed = sp.diff(log_expr, sym_dict[vi], sym_dict[vj])
                if sp.simplify(mixed) != 0:
                    return 0
            return 1
        except Exception:
            pass
        return 0
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Pure numerical label computation (fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_labels(
    X: np.ndarray, y: np.ndarray, n_vars: int,
) -> dict:
    """Compute property labels using pure numerical methods (4-class encoding)."""
    mono = np.zeros(n_vars, dtype=np.int64)
    conv = np.zeros(n_vars, dtype=np.int64)
    period = np.zeros(n_vars, dtype=np.int64)

    mask = np.isfinite(y) & np.all(np.isfinite(X[:, :n_vars]), axis=1)
    y_clean = y[mask]
    X_clean = X[mask, :n_vars]

    if len(y_clean) < 30:
        return _empty_labels(n_vars)

    for i in range(n_vars):
        mono[i] = _check_monotonicity(y_clean, X_clean, i)
        conv[i] = _check_convexity(y_clean, X_clean, i)
        period[i] = _check_periodicity(y_clean, X_clean, i)

    mul_sep = _check_multiplicative_separability(y_clean, X_clean)

    return {
        "monotonicity": mono,
        "convexity": conv,
        "periodicity": period,
        "multiplicative_separable": mul_sep,
    }


def _empty_labels(n_vars):
    return {
        "monotonicity": np.zeros(n_vars, dtype=np.int64),
        "convexity": np.zeros(n_vars, dtype=np.int64),
        "periodicity": np.zeros(n_vars, dtype=np.int64),
        "multiplicative_separable": 0,
    }


def _sorted_y(X_clean, var_idx, N):
    """Helper: sort y by X[:, var_idx] and return sorted y."""
    return X_clean[np.argsort(X_clean[:, var_idx])]


def _check_monotonicity(y, X, var_idx, n_bins=15, confidence=0.85):
    """Returns 4-class encoding: 0=non-mono, 1=inc, 2=dec, 3=const."""
    x = X[:, var_idx]
    sorted_idx = np.argsort(x)
    y_sorted = y[sorted_idx]
    bin_size = len(y_sorted) // n_bins
    if bin_size < 3:
        return 0
    bin_means = np.array([y_sorted[b * bin_size:(b + 1) * bin_size].mean() for b in range(n_bins)])
    if np.std(bin_means) < 1e-10 * (np.abs(np.mean(bin_means)) + 1e-15):
        return 3
    diffs = np.diff(bin_means)
    pos_frac = (diffs > 0).sum() / len(diffs)
    neg_frac = (diffs < 0).sum() / len(diffs)
    if pos_frac >= confidence:
        return 1
    elif neg_frac >= confidence:
        return 2
    return 0


def _numerical_monotonicity(x_col, y_sorted_placeholder):
    """Lightweight fallback for SymPy failures."""
    return 0


def _check_convexity(y, X, var_idx, n_bins=10, confidence=0.75):
    """Returns 4-class encoding: 0=neither, 1=convex, 2=concave, 3=affine."""
    x = X[:, var_idx]
    sorted_idx = np.argsort(x)
    y_sorted = y[sorted_idx]
    bin_size = len(y_sorted) // n_bins
    if bin_size < 3:
        return 0
    bin_means = np.array([y_sorted[b * bin_size:(b + 1) * bin_size].mean() for b in range(n_bins)])
    second_diffs = np.diff(bin_means, n=2)
    if np.std(second_diffs) < 1e-10 * (np.abs(np.mean(bin_means)) + 1e-15):
        return 3
    pos_frac = (second_diffs > 0).sum() / len(second_diffs)
    neg_frac = (second_diffs < 0).sum() / len(second_diffs)
    if pos_frac >= confidence:
        return 1
    elif neg_frac >= confidence:
        return 2
    return 0


def _numerical_convexity(x_col, y_sorted_placeholder):
    return 0


def _check_periodicity(y, X, var_idx, min_periods=2.0):
    x = X[:, var_idx]
    sorted_idx = np.argsort(x)
    y_sorted = y[sorted_idx]
    x_sorted = x[sorted_idx]
    x_range = x_sorted[-1] - x_sorted[0]
    if x_range < 1e-10:
        return 0
    n_points = min(1024, len(y_sorted))
    x_uniform = np.linspace(x_sorted[0], x_sorted[-1], n_points)
    y_interp = np.interp(x_uniform, x_sorted, y_sorted)
    y_detrend = y_interp - np.polyval(np.polyfit(x_uniform, y_interp, 1), x_uniform)
    if np.std(y_detrend) < 1e-10:
        return 0
    fft_vals = np.abs(np.fft.rfft(y_detrend))
    freqs = np.fft.rfftfreq(n_points, d=(x_uniform[1] - x_uniform[0]))
    fft_vals[0] = 0
    if len(fft_vals) < 3:
        return 0
    min_freq = min_periods / x_range
    valid_mask = freqs >= min_freq
    if not np.any(valid_mask):
        return 0
    fft_valid = fft_vals[valid_mask]
    peak_power = np.max(fft_valid) ** 2
    total_power = np.sum(fft_valid ** 2)
    if total_power < 1e-15:
        return 0
    return 1 if (peak_power / total_power) > 0.3 else 0


def _check_multiplicative_separability(y, X, n_bins=10, threshold=0.05):
    n_vars = X.shape[1]
    if n_vars < 2:
        return 1
    if np.any(y <= 0):
        y_shifted = y - np.min(y) + 1e-6
    else:
        y_shifted = y
    log_y = np.log(y_shifted)
    if not np.all(np.isfinite(log_y)):
        return 0
    return _check_additive_sep(log_y, X, n_bins, threshold)


def _check_additive_sep(y, X, n_bins=10, threshold=0.05):
    n_vars = X.shape[1]
    if n_vars < 2:
        return 1
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            xi_bins = np.digitize(X[:, i], np.linspace(X[:, i].min(), X[:, i].max(), n_bins + 1)[:-1])
            xj_bins = np.digitize(X[:, j], np.linspace(X[:, j].min(), X[:, j].max(), n_bins + 1)[:-1])
            xi_bins = np.clip(xi_bins, 1, n_bins)
            xj_bins = np.clip(xj_bins, 1, n_bins)
            grand_mean = np.mean(y)
            row_means = np.zeros(n_bins + 1)
            col_means = np.zeros(n_bins + 1)
            for b in range(1, n_bins + 1):
                mi = xi_bins == b
                mj = xj_bins == b
                if mi.sum() > 0:
                    row_means[b] = np.mean(y[mi])
                if mj.sum() > 0:
                    col_means[b] = np.mean(y[mj])
            ss_interaction = 0.0
            ss_total = np.var(y) * len(y)
            n_cells = 0
            for bi in range(1, n_bins + 1):
                for bj in range(1, n_bins + 1):
                    cell_mask = (xi_bins == bi) & (xj_bins == bj)
                    cnt = cell_mask.sum()
                    if cnt < 3:
                        continue
                    cell_mean = np.mean(y[cell_mask])
                    expected = row_means[bi] + col_means[bj] - grand_mean
                    ss_interaction += cnt * (cell_mean - expected) ** 2
                    n_cells += 1
            if ss_total < 1e-15 or n_cells < 4:
                continue
            if ss_interaction / ss_total > threshold:
                return 0
    return 1


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities for rejection sampling
# ═══════════════════════════════════════════════════════════════════════════════

def label_signature(mono, conv, period, mul_sep):
    return (tuple(mono), tuple(conv), tuple(period), int(mul_sep))


def coarse_label_key(mono, conv, period, mul_sep):
    """Coarser signature for balanced rejection sampling."""
    mono = np.asarray(mono)
    conv = np.asarray(conv)
    period = np.asarray(period)
    return (
        bool(np.any(mono == 1)),   # has increasing
        bool(np.any(mono == 2)),   # has decreasing
        bool(np.any(mono == 3)),   # has constant
        bool(np.any(conv == 1)),   # has convex
        bool(np.any(conv == 2)),   # has concave
        bool(np.any(conv == 3)),   # has affine
        bool(np.any(period == 1)), # has periodic
        int(mul_sep),
    )
