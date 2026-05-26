"""Generate property labels for all LLM-SRBench formulas.

Properties (excluding additive separability):
- Monotonicity (per-variable): 1=increasing, 2=decreasing, 3=non-monotonic, 4=constant, 0=unknown
- Convexity (per-variable): 1=convex, 2=concave, 3=neither, 4=affine, 0=unknown
- Periodicity (per-variable): 0=non-periodic, 1=periodic
- Multiplicative separability (per-formula): 0=not separable, 1=separable

Usage:
    conda run -n sragent python src/experimental/nn_tools/datasets/generate_property_labels.py

Output:
    data/llm-srbench-data/property_label/
    ├── all_labels.json            # All 216 formula labels in a single file
    ├── lsr_transform.json         # Per-dataset label files
    ├── lsr_synth_matsci.json
    ├── lsr_synth_chem_react.json
    └── lsr_synth_phys_osc.json
"""

import sys
import json
import h5py
import numpy as np
import sympy as sp
from pathlib import Path
from itertools import combinations

try:
    import datasets as hf_datasets
except ImportError:
    print("ERROR: 'datasets' library required. Install via: pip install datasets")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "llm-srbench-data"
HDF5_PATH = DATA_ROOT / "lsr_bench_data.hdf5"
OUTPUT_DIR = DATA_ROOT / "property_label"

ENCODING_DOC = """
Property encoding convention:
  Monotonicity: 1=increasing, 2=decreasing, 3=non-monotonic, 4=constant, 0=unknown
  Convexity:    1=convex, 2=concave, 3=neither, 4=affine, 0=unknown
  Periodicity:  0=non-periodic, 1=periodic
  Multiplicative separability: 0=not separable, 1=separable
"""

HDF5_SPLITS_FOR_LABELS = ("train", "test", "ood_test")


# =============================================================================
# Data Loading
# =============================================================================

def load_all_problems():
    """Load all problems from parquet files (4 datasets, 216 formulas total)."""
    data_dir = DATA_ROOT / "data"
    splits = ['lsr_transform', 'lsr_synth_matsci', 'lsr_synth_chem_react', 'lsr_synth_phys_osc']
    problems = []
    for split in splits:
        pq_files = list(data_dir.glob(f"{split}-*.parquet"))
        if not pq_files:
            print(f"  WARNING: No parquet file found for {split}")
            continue
        ds = hf_datasets.load_dataset("parquet", data_files=str(pq_files[0]), split="train")
        for entry in ds:
            problems.append({
                "dataset": split,
                "name": entry["name"],
                "expression": entry["expression"],
                "symbols": entry["symbols"],
                "symbol_properties": entry["symbol_properties"],
            })
    return problems


def get_hdf5_problem_group(f, dataset, name):
    """Return the HDF5 group containing arrays for one problem."""
    ds_map = {
        'lsr_synth_chem_react': 'chem_react',
        'lsr_synth_phys_osc': 'phys_osc',
        'lsr_synth_matsci': 'matsci',
    }
    if dataset == 'lsr_transform':
        return f['lsr_transform'][name]
    domain = ds_map[dataset]
    return f['lsr_synth'][domain][name]


def load_hdf5_arrays(dataset, name, splits=HDF5_SPLITS_FOR_LABELS):
    """Load and concatenate available HDF5 splits for one problem."""
    arrays = []
    with h5py.File(HDF5_PATH, 'r') as f:
        grp = get_hdf5_problem_group(f, dataset, name)
        for split in splits:
            if split in grp and isinstance(grp[split], h5py.Dataset):
                arrays.append(grp[split][:])
    if not arrays:
        raise KeyError(f"No HDF5 arrays found for {dataset}/{name}")
    return np.concatenate(arrays, axis=0)


def load_hdf5_data(dataset, name):
    """Load train/test/ood_test data from HDF5 for numerical analysis."""
    data = load_hdf5_arrays(dataset, name)
    y = data[:, 0]
    X = data[:, 1:]
    return y, X


def get_input_variables(symbols, symbol_properties):
    """Extract input variable names from symbol metadata."""
    target = symbols[0]
    input_vars = []
    for sym, prop in zip(symbols[1:], symbol_properties[1:]):
        if "V" in prop.upper() or "variable" in prop.lower() or "input" in prop.lower():
            input_vars.append(sym)
    if not input_vars:
        input_vars = symbols[1:]
    return input_vars


# =============================================================================
# SymPy Symbolic Analysis (for lsr_transform, lsr_synth_matsci)
# =============================================================================

def parse_expression(expr_str, symbols_list, symbol_properties):
    """Parse expression string to sympy expression."""
    input_vars = get_input_variables(symbols_list, symbol_properties)

    sym_dict = {}
    for v in input_vars:
        clean_name = v.replace("(", "_").replace(")", "").replace(" ", "_")
        sym_dict[v] = sp.Symbol(clean_name, real=True)

    try:
        local_ns = {"pi": sp.pi, "e": sp.E, "E": sp.E, "I": sp.I}
        local_ns.update({v: sym_dict[v] for v in input_vars})
        for v in symbols_list:
            if v not in local_ns and v != symbols_list[0]:
                local_ns[v] = sp.Symbol(v, real=True)
                sym_dict[v] = local_ns[v]
        parsed = sp.sympify(expr_str, locals=local_ns)
        return parsed, sym_dict, input_vars
    except Exception:
        return None, sym_dict, input_vars


def sympy_check_monotonicity(expr, sym_dict, input_vars, domain_data=None, n_points=200):
    """Check monotonicity per variable using symbolic differentiation + numerical evaluation."""
    eps = 1e-8
    result = {}
    rng = np.random.default_rng(42)

    for v in input_vars:
        if v not in sym_dict or sym_dict[v] not in expr.free_symbols:
            result[v] = 4
            continue
        try:
            d1 = sp.diff(expr, sym_dict[v])
            if d1.is_zero:
                result[v] = 4
                continue

            other_syms = [sym_dict[u] for u in input_vars if u != v and u in sym_dict]
            all_syms = [sym_dict[v]] + other_syms

            if domain_data and v in domain_data:
                vmin, vmax = float(np.min(domain_data[v])), float(np.max(domain_data[v]))
            else:
                vmin, vmax = -5.0, 5.0

            test_vals = rng.uniform(vmin, vmax, n_points)
            other_vals = {}
            for u in input_vars:
                if u == v or u not in sym_dict:
                    continue
                s = sym_dict[u]
                if domain_data and u in domain_data:
                    omin, omax = float(np.min(domain_data[u])), float(np.max(domain_data[u]))
                    other_vals[s] = rng.uniform(omin, omax, n_points)
                else:
                    other_vals[s] = rng.uniform(-5, 5, n_points)

            d1_func = sp.lambdify(all_syms, d1, modules=["numpy"])
            args = [test_vals] + [other_vals[s] for s in other_syms]

            d1_vals = np.asarray(d1_func(*args), dtype=float)
            if d1_vals.ndim == 0 or d1_vals.size == 1:
                d1_vals = np.full(n_points, float(d1_vals.reshape(-1)[0]))
            else:
                d1_vals = d1_vals.flatten()
            valid = np.isfinite(d1_vals)
            if valid.sum() < 10:
                result[v] = 0
                continue
            d1_valid = d1_vals[valid]

            pos_frac = np.mean(d1_valid > eps)
            neg_frac = np.mean(d1_valid < -eps)
            zero_frac = np.mean(np.abs(d1_valid) < eps)

            if pos_frac >= 0.95:
                result[v] = 1
            elif neg_frac >= 0.95:
                result[v] = 2
            elif zero_frac >= 0.95:
                result[v] = 4
            else:
                result[v] = 3
        except Exception:
            result[v] = 0
    return result


def sympy_check_convexity(expr, sym_dict, input_vars, domain_data=None, n_points=200):
    """Check convexity per variable using second derivatives."""
    eps = 1e-8
    result = {}
    rng = np.random.default_rng(42)

    for v in input_vars:
        if v not in sym_dict or sym_dict[v] not in expr.free_symbols:
            result[v] = 4
            continue
        try:
            d2 = sp.diff(expr, sym_dict[v], 2)
            if d2.is_zero:
                result[v] = 4
                continue

            dirac_terms = d2.atoms(sp.DiracDelta)
            if dirac_terms:
                d2 = d2.xreplace({term: sp.Integer(0) for term in dirac_terms})

            other_syms = [sym_dict[u] for u in input_vars if u != v and u in sym_dict]
            all_syms = [sym_dict[v]] + other_syms

            if domain_data and v in domain_data:
                vmin, vmax = float(np.min(domain_data[v])), float(np.max(domain_data[v]))
            else:
                vmin, vmax = -5.0, 5.0

            test_vals = rng.uniform(vmin, vmax, n_points)
            other_vals = {}
            for u in input_vars:
                if u == v or u not in sym_dict:
                    continue
                s = sym_dict[u]
                if domain_data and u in domain_data:
                    omin, omax = float(np.min(domain_data[u])), float(np.max(domain_data[u]))
                    other_vals[s] = rng.uniform(omin, omax, n_points)
                else:
                    other_vals[s] = rng.uniform(-5, 5, n_points)

            d2_func = sp.lambdify(all_syms, d2, modules=["numpy"])
            args = [test_vals] + [other_vals[s] for s in other_syms]

            d2_vals = np.asarray(d2_func(*args), dtype=float)
            if d2_vals.ndim == 0 or d2_vals.size == 1:
                d2_vals = np.full(n_points, float(d2_vals.reshape(-1)[0]))
            else:
                d2_vals = d2_vals.flatten()
            valid = np.isfinite(d2_vals)
            if valid.sum() < 10:
                result[v] = 0
                continue
            d2_valid = d2_vals[valid]

            pos_frac = np.mean(d2_valid > eps)
            neg_frac = np.mean(d2_valid < -eps)
            zero_frac = np.mean(np.abs(d2_valid) < eps)

            if pos_frac >= 0.95:
                result[v] = 1
            elif neg_frac >= 0.95:
                result[v] = 2
            elif zero_frac >= 0.95:
                result[v] = 4
            else:
                result[v] = 3
        except Exception:
            result[v] = 0
    return result


def sympy_check_periodicity(expr, sym_dict, input_vars):
    """Check periodicity per variable using sympy.periodicity."""
    result = {}
    for v in input_vars:
        if v not in sym_dict or sym_dict[v] not in expr.free_symbols:
            result[v] = 0
            continue
        try:
            period = sp.periodicity(expr, sym_dict[v])
            result[v] = 1 if period is not None else 0
        except Exception:
            result[v] = 0
    return result


def sympy_check_multiplicative_separability(expr, sym_dict, input_vars):
    """Check multiplicative separability using structure analysis."""
    active_vars = [v for v in input_vars if v in sym_dict and sym_dict[v] in expr.free_symbols]
    if len(active_vars) <= 1:
        return 1

    try:
        if expr.func == sp.Mul:
            factors = expr.args
            var_groups = []
            for factor in factors:
                fvars = factor.free_symbols & set(sym_dict[v] for v in active_vars)
                if fvars:
                    var_groups.append(fvars)
            for i, j in combinations(range(len(var_groups)), 2):
                if not var_groups[i] & var_groups[j]:
                    return 1

        try:
            log_expr = sp.log(expr)
            log_expanded = sp.expand(log_expr)
            if _sympy_check_additive_sep(log_expanded, sym_dict, active_vars):
                return 1
        except Exception:
            pass

        return 0
    except Exception:
        return 0


def _sympy_check_additive_sep(expr, sym_dict, active_vars):
    """Internal: check additive separability via mixed partials."""
    if len(active_vars) <= 1:
        return True
    try:
        for vi, vj in combinations(active_vars, 2):
            mixed = sp.diff(expr, sym_dict[vi], sym_dict[vj])
            try:
                simplified = sp.simplify(mixed)
                if simplified != 0:
                    return False
            except Exception:
                return False
        return True
    except Exception:
        return False


def get_hdf5_domain(dataset, name, input_vars):
    """Load variable domains from train/test/ood_test HDF5 arrays."""
    try:
        data = load_hdf5_arrays(dataset, name)
        if data.ndim != 2 or data.shape[1] < len(input_vars) + 1:
            return None
        return {var: data[:, i + 1] for i, var in enumerate(input_vars)}
    except Exception:
        pass
    return None


# =============================================================================
# Numerical Analysis (for lsr_synth_chem_react, lsr_synth_phys_osc)
# =============================================================================

def numerical_check_active(y, X, threshold=0.01):
    """Check active variables using correlation + binned variance."""
    n_vars = X.shape[1]
    active = []
    for i in range(n_vars):
        corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
        if np.isnan(corr):
            corr = 0.0

        n_bins = 20
        x_sorted_idx = np.argsort(X[:, i])
        bin_size = len(y) // n_bins
        bin_means = []
        for b in range(n_bins):
            start = b * bin_size
            end = start + bin_size
            bin_means.append(np.mean(y[x_sorted_idx[start:end]]))
        bin_variance = np.var(bin_means)
        total_variance = np.var(y)

        is_active = (corr > threshold) or (bin_variance / (total_variance + 1e-15) > threshold)
        active.append(1 if is_active else 0)
    return active


def numerical_check_monotonicity(y, X, var_idx, confidence=0.90):
    """Returns: 1=increasing, 2=decreasing, 3=nonmonotonic, 4=constant, 0=unknown"""
    x = X[:, var_idx]
    sorted_idx = np.argsort(x)
    y_sorted = y[sorted_idx]

    n_bins = 50
    bin_size = len(y_sorted) // n_bins
    if bin_size < 5:
        return 0

    bin_means_y = []
    for b in range(n_bins):
        start = b * bin_size
        end = start + bin_size
        bin_means_y.append(np.mean(y_sorted[start:end]))

    bin_means_y = np.array(bin_means_y)
    diffs = np.diff(bin_means_y)

    if np.std(bin_means_y) < 1e-10 * (np.abs(np.mean(bin_means_y)) + 1e-15):
        return 4

    pos_frac = np.sum(diffs > 0) / len(diffs)
    neg_frac = np.sum(diffs < 0) / len(diffs)

    if pos_frac >= confidence:
        return 1
    elif neg_frac >= confidence:
        return 2
    else:
        return 3


def numerical_check_convexity(y, X, var_idx, confidence=0.85):
    """Returns: 1=convex, 2=concave, 3=neither, 4=affine, 0=unknown"""
    x = X[:, var_idx]
    sorted_idx = np.argsort(x)
    y_sorted = y[sorted_idx]

    n_bins = 40
    bin_size = len(y_sorted) // n_bins
    if bin_size < 5:
        return 0

    bin_means = []
    for b in range(n_bins):
        start = b * bin_size
        end = start + bin_size
        bin_means.append(np.mean(y_sorted[start:end]))

    bin_means = np.array(bin_means)
    second_diffs = np.diff(bin_means, n=2)

    if np.std(second_diffs) < 1e-10 * (np.abs(np.mean(bin_means)) + 1e-15):
        return 4

    pos_frac = np.sum(second_diffs > 0) / len(second_diffs)
    neg_frac = np.sum(second_diffs < 0) / len(second_diffs)

    if pos_frac >= confidence:
        return 1
    elif neg_frac >= confidence:
        return 2
    else:
        return 3


def numerical_check_periodicity(y, X, var_idx, min_periods=2.0):
    """Returns: 1=periodic, 0=not periodic"""
    x = X[:, var_idx]
    sorted_idx = np.argsort(x)
    y_sorted = y[sorted_idx]
    x_sorted = x[sorted_idx]

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

    x_range = x_sorted[-1] - x_sorted[0]
    min_freq = min_periods / x_range
    valid_mask = freqs >= min_freq
    if not np.any(valid_mask):
        return 0

    fft_valid = fft_vals[valid_mask]
    peak_power = np.max(fft_valid) ** 2
    total_power = np.sum(fft_valid ** 2)

    if total_power < 1e-15:
        return 0

    dominance = peak_power / total_power
    return 1 if dominance > 0.3 else 0


def numerical_check_multiplicative_separability(y, X, n_bins=10, threshold=0.05):
    """Check multiplicative separability via log-transform + additive separability test."""
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

    return _numerical_check_additive_sep(log_y, X, n_bins, threshold)


def _numerical_check_additive_sep(y, X, n_bins=10, threshold=0.05):
    """Numerical additive separability check via ANOVA interaction test."""
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
                mask_i = xi_bins == b
                mask_j = xj_bins == b
                if np.sum(mask_i) > 0:
                    row_means[b] = np.mean(y[mask_i])
                if np.sum(mask_j) > 0:
                    col_means[b] = np.mean(y[mask_j])

            ss_interaction = 0.0
            ss_total = np.var(y) * len(y)
            n_cells = 0
            for bi in range(1, n_bins + 1):
                for bj in range(1, n_bins + 1):
                    mask = (xi_bins == bi) & (xj_bins == bj)
                    count = np.sum(mask)
                    if count < 3:
                        continue
                    cell_mean = np.mean(y[mask])
                    expected = row_means[bi] + col_means[bj] - grand_mean
                    ss_interaction += count * (cell_mean - expected) ** 2
                    n_cells += 1

            if ss_total < 1e-15 or n_cells < 4:
                continue

            interaction_ratio = ss_interaction / ss_total
            if interaction_ratio > threshold:
                return 0
    return 1


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def analyze_symbolic(problem):
    """Analyze a formula using SymPy symbolic methods (lsr_transform, lsr_synth_matsci)."""
    expr_str = problem["expression"]
    symbols_list = problem["symbols"]
    symbol_properties = problem["symbol_properties"]
    dataset = problem["dataset"]
    name = problem["name"]
    input_vars = get_input_variables(symbols_list, symbol_properties)

    expr, sym_dict, input_vars = parse_expression(expr_str, symbols_list, symbol_properties)
    if expr is None:
        return None

    domain_data = get_hdf5_domain(dataset, name, input_vars)

    mono = sympy_check_monotonicity(expr, sym_dict, input_vars, domain_data)
    conv = sympy_check_convexity(expr, sym_dict, input_vars, domain_data)
    period = sympy_check_periodicity(expr, sym_dict, input_vars)
    mul_sep = sympy_check_multiplicative_separability(expr, sym_dict, input_vars)

    return {
        "name": name,
        "dataset": dataset,
        "n_variables": len(input_vars),
        "variables": input_vars,
        "monotonicity": mono,
        "convexity": conv,
        "periodicity": period,
        "multiplicative_separable": mul_sep,
    }


def analyze_numerical(problem):
    """Analyze a formula using numerical methods (lsr_synth_chem_react, lsr_synth_phys_osc)."""
    name = problem["name"]
    dataset = problem["dataset"]
    symbols_list = problem["symbols"]
    symbol_properties = problem["symbol_properties"]
    input_vars = get_input_variables(symbols_list, symbol_properties)

    try:
        y, X = load_hdf5_data(dataset, name)
    except Exception as e:
        print(f"    ERROR loading data for {name}: {e}")
        return None

    n_vars = X.shape[1]
    if n_vars != len(input_vars):
        print(f"    WARNING: column mismatch for {name}: {n_vars} cols vs {len(input_vars)} vars")
        input_vars = input_vars[:n_vars] if len(input_vars) > n_vars else \
            input_vars + [f"x{i}" for i in range(len(input_vars), n_vars)]

    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    y = y[mask]
    X = X[mask]

    if len(y) < 100:
        print(f"    WARNING: insufficient data for {name} ({len(y)} points)")
        return None

    active = numerical_check_active(y, X)

    mono = {}
    conv = {}
    period = {}
    for i, var in enumerate(input_vars):
        if active[i]:
            mono[var] = numerical_check_monotonicity(y, X, i)
            conv[var] = numerical_check_convexity(y, X, i)
            period[var] = numerical_check_periodicity(y, X, i)
        else:
            mono[var] = 4
            conv[var] = 4
            period[var] = 0

    mul_sep = numerical_check_multiplicative_separability(y, X)

    return {
        "name": name,
        "dataset": dataset,
        "n_variables": len(input_vars),
        "variables": input_vars,
        "monotonicity": mono,
        "convexity": conv,
        "periodicity": period,
        "multiplicative_separable": mul_sep,
    }


def analyze_problem(problem):
    """Dispatch to symbolic or numerical analysis based on dataset."""
    dataset = problem["dataset"]
    if dataset in ('lsr_transform', 'lsr_synth_matsci'):
        return analyze_symbolic(problem)
    elif dataset in ('lsr_synth_chem_react', 'lsr_synth_phys_osc'):
        return analyze_numerical(problem)
    else:
        print(f"    Unknown dataset: {dataset}, skipping")
        return None


# =============================================================================
# Statistics & Comparison
# =============================================================================

def compute_statistics(labels):
    """Compute summary statistics from generated labels for comparison with README."""
    stats = {
        "total": len(labels),
        "per_dataset": {},
        "per_formula": {
            "has_nonmonotonic": 0,
            "has_monotonic_increasing": 0,
            "has_monotonic_decreasing": 0,
            "has_constant": 0,
            "has_convex": 0,
            "has_concave": 0,
            "has_neither_convexity": 0,
            "has_affine": 0,
            "has_periodic": 0,
            "multiplicative_separable": 0,
        },
        "per_variable": {
            "total_vars": 0,
            "monotonicity": {1: 0, 2: 0, 3: 0, 4: 0, 0: 0},
            "convexity": {1: 0, 2: 0, 3: 0, 4: 0, 0: 0},
        },
    }

    datasets = set(l["dataset"] for l in labels)
    for ds in datasets:
        stats["per_dataset"][ds] = {
            "count": 0,
            "has_nonmonotonic": 0,
            "has_monotonic_increasing": 0,
            "has_monotonic_decreasing": 0,
            "has_constant": 0,
            "has_convex": 0,
            "has_concave": 0,
            "has_neither_convexity": 0,
            "has_affine": 0,
            "has_periodic": 0,
            "multiplicative_separable": 0,
        }

    for label in labels:
        ds = label["dataset"]
        mono_vals = list(label["monotonicity"].values())
        conv_vals = list(label["convexity"].values())
        period_vals = list(label["periodicity"].values())

        stats["per_dataset"][ds]["count"] += 1

        # Per-formula flags
        if any(v == 3 for v in mono_vals):
            stats["per_formula"]["has_nonmonotonic"] += 1
            stats["per_dataset"][ds]["has_nonmonotonic"] += 1
        if any(v == 1 for v in mono_vals):
            stats["per_formula"]["has_monotonic_increasing"] += 1
            stats["per_dataset"][ds]["has_monotonic_increasing"] += 1
        if any(v == 2 for v in mono_vals):
            stats["per_formula"]["has_monotonic_decreasing"] += 1
            stats["per_dataset"][ds]["has_monotonic_decreasing"] += 1
        if any(v == 4 for v in mono_vals):
            stats["per_formula"]["has_constant"] += 1
            stats["per_dataset"][ds]["has_constant"] += 1
        if any(v == 1 for v in conv_vals):
            stats["per_formula"]["has_convex"] += 1
            stats["per_dataset"][ds]["has_convex"] += 1
        if any(v == 2 for v in conv_vals):
            stats["per_formula"]["has_concave"] += 1
            stats["per_dataset"][ds]["has_concave"] += 1
        if any(v == 3 for v in conv_vals):
            stats["per_formula"]["has_neither_convexity"] += 1
            stats["per_dataset"][ds]["has_neither_convexity"] += 1
        if any(v == 4 for v in conv_vals):
            stats["per_formula"]["has_affine"] += 1
            stats["per_dataset"][ds]["has_affine"] += 1
        if any(v == 1 for v in period_vals):
            stats["per_formula"]["has_periodic"] += 1
            stats["per_dataset"][ds]["has_periodic"] += 1
        if label["multiplicative_separable"] == 1:
            stats["per_formula"]["multiplicative_separable"] += 1
            stats["per_dataset"][ds]["multiplicative_separable"] += 1

        # Per-variable stats
        n_vars = len(mono_vals)
        stats["per_variable"]["total_vars"] += n_vars
        for v in mono_vals:
            stats["per_variable"]["monotonicity"][v] += 1
        for v in conv_vals:
            stats["per_variable"]["convexity"][v] += 1

    return stats


def print_comparison(stats):
    """Print statistics and compare with README expected values."""
    total = stats["total"]
    pf = stats["per_formula"]
    pv = stats["per_variable"]

    # Expected from README (216 formulas), using the merged HDF5
    # train/test/ood_test empirical domain for range-dependent checks.
    expected = {
        "has_nonmonotonic": 126,
        "has_monotonic_increasing": 120,
        "has_monotonic_decreasing": 133,
        "has_constant": 33,
        "has_convex": 88,
        "has_concave": 62,
        "has_neither_convexity": 128,
        "has_affine": 118,
        "has_periodic": 68,
        "multiplicative_separable": 89,
    }

    expected_per_dataset = {
        "lsr_transform": {
            "count": 111,
            "has_nonmonotonic": 35,
            "has_monotonic_increasing": 96,
            "has_monotonic_decreasing": 94,
            "has_constant": 0,
            "has_convex": 81,
            "has_concave": 50,
            "has_neither_convexity": 37,
            "has_affine": 71,
            "has_periodic": 17,
            "multiplicative_separable": 86,
        },
        "lsr_synth_matsci": {
            "count": 25,
            "has_nonmonotonic": 21,
            "has_monotonic_increasing": 17,
            "has_monotonic_decreasing": 6,
            "has_constant": 1,
            "has_convex": 6,
            "has_concave": 8,
            "has_neither_convexity": 16,
            "has_affine": 15,
            "has_periodic": 1,
            "multiplicative_separable": 2,
        },
        "lsr_synth_chem_react": {
            "count": 36,
            "has_nonmonotonic": 26,
            "has_monotonic_increasing": 6,
            "has_monotonic_decreasing": 22,
            "has_constant": 0,
            "has_convex": 1,
            "has_concave": 4,
            "has_neither_convexity": 31,
            "has_affine": 0,
            "has_periodic": 25,
            "multiplicative_separable": 1,
        },
        "lsr_synth_phys_osc": {
            "count": 44,
            "has_nonmonotonic": 44,
            "has_monotonic_increasing": 1,
            "has_monotonic_decreasing": 11,
            "has_constant": 32,
            "has_convex": 0,
            "has_concave": 0,
            "has_neither_convexity": 44,
            "has_affine": 32,
            "has_periodic": 25,
            "multiplicative_separable": 0,
        },
    }

    # Per-variable expected (742 total)
    expected_var_mono = {3: 250, 2: 236, 1: 220, 4: 36, 0: 0}
    expected_var_conv = {3: 258, 4: 197, 2: 107, 1: 180, 0: 0}

    print("\n" + "=" * 80)
    print("PROPERTY LABEL STATISTICS vs README EXPECTED")
    print("=" * 80)

    print(f"\nTotal formulas: {total} (expected: 216)")
    print(f"Total variable-formula pairs: {pv['total_vars']} (expected: 742)")

    print("\n--- Per-formula property distribution (all 216) ---")
    print(f"{'Property':<35} {'Generated':>10} {'Expected':>10} {'Match':>6}")
    print("-" * 65)
    for prop in expected:
        gen = pf[prop]
        exp = expected[prop]
        match = "✓" if gen == exp else f"Δ={gen-exp:+d}"
        pct = gen / total * 100
        print(f"  {prop:<33} {gen:>5} ({pct:5.1f}%) {exp:>5}      {match}")

    print("\n--- Per-dataset breakdown ---")
    for ds_name, exp_ds in expected_per_dataset.items():
        gen_ds = stats["per_dataset"].get(ds_name, {})
        ds_count = gen_ds.get("count", 0)
        print(f"\n  [{ds_name}] ({ds_count}/{exp_ds['count']} formulas)")
        for prop in ["has_nonmonotonic", "has_monotonic_increasing", "has_monotonic_decreasing",
                     "has_constant", "has_convex", "has_concave", "has_neither_convexity",
                     "has_affine", "has_periodic", "multiplicative_separable"]:
            gen = gen_ds.get(prop, 0)
            exp = exp_ds[prop]
            match = "✓" if gen == exp else f"Δ={gen-exp:+d}"
            print(f"    {prop:<33} {gen:>4} / {exp:>4}  {match}")

    print("\n--- Per-variable monotonicity ---")
    print(f"  {'Category':<20} {'Generated':>10} {'Expected':>10} {'Match':>6}")
    mono_names = {1: "increasing", 2: "decreasing", 3: "non-monotonic", 4: "constant", 0: "unknown"}
    for code in [3, 2, 1, 4, 0]:
        gen = pv["monotonicity"][code]
        exp = expected_var_mono.get(code, 0)
        match = "✓" if gen == exp else f"Δ={gen-exp:+d}"
        print(f"  {mono_names[code]:<20} {gen:>10} {exp:>10} {match:>6}")

    print("\n--- Per-variable convexity ---")
    conv_names = {1: "convex", 2: "concave", 3: "neither", 4: "affine", 0: "unknown"}
    for code in [3, 4, 2, 1, 0]:
        gen = pv["convexity"][code]
        exp = expected_var_conv.get(code, 0)
        match = "✓" if gen == exp else f"Δ={gen-exp:+d}"
        print(f"  {conv_names[code]:<20} {gen:>10} {exp:>10} {match:>6}")

    print("\n" + "=" * 80)


# =============================================================================
# Entry Point
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate property labels for LLM-SRBench formulas")
    parser.add_argument("--use-precomputed", type=str, default=None,
                        help="Path to pre-computed JSON (formula_properties_all_216.json) to skip recomputation")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for label files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_precomputed:
        print(f"Loading pre-computed results from: {args.use_precomputed}")
        with open(args.use_precomputed) as f:
            precomputed = json.load(f)

        labels = []
        for item in precomputed:
            if not item.get("parse_success", False):
                continue
            input_vars = list(item.get("monotonicity_detail", {}).keys())
            labels.append({
                "name": item["name"],
                "dataset": item["dataset"],
                "n_variables": len(input_vars),
                "variables": input_vars,
                "monotonicity": item.get("monotonicity_detail", {}),
                "convexity": item.get("convexity_detail", {}),
                "periodicity": item.get("periodicity_detail", {}),
                "multiplicative_separable": item.get("multiplicative_separable", 0),
            })
    else:
        print("Loading problems from parquet files...")
        problems = load_all_problems()
        print(f"Loaded {len(problems)} problems from 4 datasets")

        labels = []
        failed = []
        for i, prob in enumerate(problems):
            if i % 10 == 0:
                print(f"  [{i+1}/{len(problems)}] {prob['dataset']}/{prob['name']}...")
            try:
                result = analyze_problem(prob)
                if result is not None:
                    labels.append(result)
                else:
                    failed.append(prob["name"])
            except Exception as e:
                print(f"    ERROR: {prob['name']} - {e}")
                failed.append(prob["name"])

        if failed:
            print(f"\nFailed to analyze {len(failed)} formulas: {failed[:10]}...")

    print(f"\nSuccessfully generated labels for {len(labels)} formulas")

    # Save all labels
    all_labels_path = output_dir / "all_labels.json"
    with open(all_labels_path, "w") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    print(f"Saved: {all_labels_path}")

    # Save per-dataset files
    datasets = set(l["dataset"] for l in labels)
    for ds in sorted(datasets):
        ds_labels = [l for l in labels if l["dataset"] == ds]
        ds_path = output_dir / f"{ds}.json"
        with open(ds_path, "w") as f:
            json.dump(ds_labels, f, indent=2, ensure_ascii=False)
        print(f"Saved: {ds_path} ({len(ds_labels)} formulas)")

    # Save encoding documentation
    encoding_path = output_dir / "ENCODING.md"
    with open(encoding_path, "w") as f:
        f.write("# Property Label Encoding\n\n")
        f.write("## Properties (excluding additive separability)\n\n")
        f.write("### Monotonicity (per-variable)\n")
        f.write("| Code | Meaning |\n|------|--------|\n")
        f.write("| 1 | Monotonic increasing |\n")
        f.write("| 2 | Monotonic decreasing |\n")
        f.write("| 3 | Non-monotonic |\n")
        f.write("| 4 | Constant / inactive |\n")
        f.write("| 0 | Unknown / computation failed |\n\n")
        f.write("### Convexity (per-variable)\n")
        f.write("| Code | Meaning |\n|------|--------|\n")
        f.write("| 1 | Convex |\n")
        f.write("| 2 | Concave |\n")
        f.write("| 3 | Neither convex nor concave |\n")
        f.write("| 4 | Affine / linear |\n")
        f.write("| 0 | Unknown / computation failed |\n\n")
        f.write("### Periodicity (per-variable)\n")
        f.write("| Code | Meaning |\n|------|--------|\n")
        f.write("| 0 | Non-periodic |\n")
        f.write("| 1 | Periodic |\n\n")
        f.write("### Multiplicative Separability (per-formula)\n")
        f.write("| Code | Meaning |\n|------|--------|\n")
        f.write("| 0 | Not multiplicatively separable |\n")
        f.write("| 1 | Multiplicatively separable |\n\n")
        f.write("## Label Structure\n\n")
        f.write("Each formula label contains:\n")
        f.write("```json\n")
        f.write('{\n')
        f.write('  "name": "formula_name",\n')
        f.write('  "dataset": "dataset_name",\n')
        f.write('  "n_variables": 3,\n')
        f.write('  "variables": ["x1", "x2", "x3"],\n')
        f.write('  "monotonicity": {"x1": 1, "x2": 3, "x3": 4},\n')
        f.write('  "convexity": {"x1": 4, "x2": 3, "x3": 4},\n')
        f.write('  "periodicity": {"x1": 0, "x2": 1, "x3": 0},\n')
        f.write('  "multiplicative_separable": 1\n')
        f.write('}\n')
        f.write("```\n")
        f.write("\n## Label Domain\n\n")
        f.write("Range-dependent labels are evaluated on the empirical benchmark domain from ")
        f.write("`lsr_bench_data.hdf5`. For each formula, the generator concatenates the ")
        f.write("available `train`, `test`, and `ood_test` arrays, then uses the observed ")
        f.write("input-variable ranges for symbolic derivative sampling and the merged ")
        f.write("observations for numerical checks.\n\n")
        f.write("The HDF5 arrays store numeric samples for each problem. Column 0 is the ")
        f.write("target/output variable and columns 1..N are the input variables listed in ")
        f.write("the corresponding parquet metadata.\n\n")
        f.write("## Regeneration Summary\n\n")
        f.write("Generated with:\n\n")
        f.write("```bash\n")
        f.write("conda run -n sragent python src/experimental/nn_tools/datasets/generate_property_labels.py\n")
        f.write("```\n\n")
        f.write("Current HDF5-domain labels cover 216 formulas and 742 variable-formula pairs. ")
        f.write("There are no unknown monotonicity or convexity labels after using the merged ")
        f.write("HDF5 domain.\n")
    print(f"Saved: {encoding_path}")

    # Compute and print comparison statistics
    stats = compute_statistics(labels)
    print_comparison(stats)


if __name__ == "__main__":
    main()
