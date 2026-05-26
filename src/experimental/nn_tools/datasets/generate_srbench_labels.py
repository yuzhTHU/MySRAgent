#!/usr/bin/env python
# conda run -n sragent python src/experimental/nn_tools/datasets/generate_srbench_labels.py
"""Regenerate LLM-SRBench property labels with UNIFIED methods.

Ensures full consistency between training labels and GT labels:
  - Mono/Conv: SymPy diff + lambdify, confidence=0.95 (same as compute_labels.py)
  - Periodicity: FFT numerical (same as compute_labels.py)
  - Separability: ANOVA numerical (same as compute_labels.py)
  - Encoding: 4-class (0=default, 1=inc/convex, 2=dec/concave, 3=const/affine)

For chem_react/phys_osc (no parseable expression): all numerical.
For lsr_transform/matsci: SymPy for mono/conv, numerical for period/sep.
"""
from __future__ import annotations
import sys
import json
import h5py
import numpy as np
import sympy as sp
from pathlib import Path

try:
    import datasets as hf_datasets
except ImportError:
    print("ERROR: 'datasets' library required.")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experimental"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_ROOT = PROJECT_ROOT / "data" / "llm-srbench-data"
HDF5_PATH = DATA_ROOT / "lsr_bench_data.hdf5"
OUTPUT_DIR = DATA_ROOT / "property_label"

# Import the SAME functions used in training
from nn_tools.datasets.compute_labels import (
    _check_monotonicity,
    _check_convexity,
    _check_periodicity,
    _check_multiplicative_separability,
    _eval_derivative_sign,
    MONO_CLASSES,
    CONV_CLASSES,
)


def load_all_problems():
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


def get_input_variables(symbols, symbol_properties):
    input_vars = []
    for sym, prop in zip(symbols[1:], symbol_properties[1:]):
        if "V" in prop.upper() or "variable" in prop.lower() or "input" in prop.lower():
            input_vars.append(sym)
    if not input_vars:
        input_vars = symbols[1:]
    return input_vars


def load_hdf5_data(dataset, name, splits=("train", "test", "ood_test")):
    ds_map = {
        'lsr_synth_chem_react': 'chem_react',
        'lsr_synth_phys_osc': 'phys_osc',
        'lsr_synth_matsci': 'matsci',
    }
    arrays = []
    with h5py.File(HDF5_PATH, 'r') as f:
        if dataset == 'lsr_transform':
            grp = f['lsr_transform'][name]
        else:
            grp = f['lsr_synth'][ds_map[dataset]][name]
        for split in splits:
            if split in grp and isinstance(grp[split], h5py.Dataset):
                arrays.append(grp[split][:])
    if not arrays:
        raise KeyError(f"No HDF5 arrays found for {dataset}/{name}")
    data = np.concatenate(arrays, axis=0)
    y = data[:, 0]
    X = data[:, 1:]
    return y, X


def parse_expression(expr_str, symbols_list, symbol_properties):
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


def sympy_check_mono_conv(expr, sym_dict, input_vars, domain_data, n_points=200):
    """SymPy diff + lambdify for mono/conv — SAME method as compute_labels.py."""
    rng = np.random.default_rng(42)
    mono = {}
    conv = {}
    eps = 1e-8

    for v in input_vars:
        if v not in sym_dict or sym_dict[v] not in expr.free_symbols:
            mono[v] = 3  # constant
            conv[v] = 3  # affine
            continue

        # Build symbol list and sampling ranges
        var_names_local = list(sym_dict.keys())
        var_idx = var_names_local.index(v)

        # Prepare X_clean-like array from domain_data for _eval_derivative_sign
        other_syms_ordered = [sym_dict[u] for u in var_names_local if u != v]
        all_syms = [sym_dict[v]] + other_syms_ordered

        if domain_data and v in domain_data:
            vmin, vmax = float(np.min(domain_data[v])), float(np.max(domain_data[v]))
        else:
            vmin, vmax = -5.0, 5.0
        if vmax - vmin < 1e-12:
            mono[v] = 3
            conv[v] = 3
            continue

        test_vals = rng.uniform(vmin, vmax, n_points)
        other_vals_list = []
        for u in var_names_local:
            if u == v:
                continue
            if domain_data and u in domain_data:
                omin, omax = float(np.min(domain_data[u])), float(np.max(domain_data[u]))
            else:
                omin, omax = -5.0, 5.0
            other_vals_list.append(rng.uniform(omin, omax, n_points))

        # --- Monotonicity ---
        try:
            d1 = sp.diff(expr, sym_dict[v])
            if d1.is_zero:
                mono[v] = 3
            else:
                d1_func = sp.lambdify(all_syms, d1, modules=["numpy"])
                args = [test_vals] + other_vals_list
                d1_vals = np.asarray(d1_func(*args), dtype=float)
                if d1_vals.ndim == 0 or d1_vals.size == 1:
                    d1_vals = np.full(n_points, float(d1_vals.reshape(-1)[0]))
                else:
                    d1_vals = d1_vals.flatten()
                valid = np.isfinite(d1_vals)
                if valid.sum() < 10:
                    mono[v] = 0
                else:
                    d1_v = d1_vals[valid]
                    pos = np.mean(d1_v > eps)
                    neg = np.mean(d1_v < -eps)
                    zero = np.mean(np.abs(d1_v) < eps)
                    if pos >= 0.95:
                        mono[v] = 1
                    elif neg >= 0.95:
                        mono[v] = 2
                    elif zero >= 0.95:
                        mono[v] = 3
                    else:
                        mono[v] = 0  # non-monotonic → default
        except Exception:
            mono[v] = 0

        # --- Convexity ---
        try:
            d2 = sp.diff(expr, sym_dict[v], 2)
            if d2.is_zero:
                conv[v] = 3
            else:
                dirac_terms = d2.atoms(sp.DiracDelta)
                if dirac_terms:
                    d2 = d2.xreplace({term: sp.Integer(0) for term in dirac_terms})
                if d2.is_zero:
                    conv[v] = 3
                else:
                    d2_func = sp.lambdify(all_syms, d2, modules=["numpy"])
                    args = [test_vals] + other_vals_list
                    d2_vals = np.asarray(d2_func(*args), dtype=float)
                    if d2_vals.ndim == 0 or d2_vals.size == 1:
                        d2_vals = np.full(n_points, float(d2_vals.reshape(-1)[0]))
                    else:
                        d2_vals = d2_vals.flatten()
                    valid = np.isfinite(d2_vals)
                    if valid.sum() < 10:
                        conv[v] = 0
                    else:
                        d2_v = d2_vals[valid]
                        pos = np.mean(d2_v > eps)
                        neg = np.mean(d2_v < -eps)
                        zero = np.mean(np.abs(d2_v) < eps)
                        if pos >= 0.95:
                            conv[v] = 1
                        elif neg >= 0.95:
                            conv[v] = 2
                        elif zero >= 0.95:
                            conv[v] = 3
                        else:
                            conv[v] = 0  # neither → default
        except Exception:
            conv[v] = 0

    return mono, conv


def analyze_problem(problem):
    """Analyze one problem with UNIFIED methods (4-class encoding)."""
    dataset = problem["dataset"]
    name = problem["name"]
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
        input_vars = input_vars[:n_vars] if len(input_vars) > n_vars else \
            input_vars + [f"x{i}" for i in range(len(input_vars), n_vars)]

    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    y = y[mask]
    X = X[mask]
    if len(y) < 100:
        print(f"    WARNING: insufficient data for {name} ({len(y)} points)")
        return None

    # --- Mono/Conv: SymPy for lsr_transform/matsci, numerical for chem_react/phys_osc ---
    mono_dict = {}
    conv_dict = {}

    if dataset in ('lsr_transform', 'lsr_synth_matsci'):
        expr, sym_dict, parsed_vars = parse_expression(
            problem["expression"], symbols_list, symbol_properties)
        if expr is not None:
            domain_data = {var: X[:, i] for i, var in enumerate(input_vars)}
            mono_dict, conv_dict = sympy_check_mono_conv(expr, sym_dict, input_vars, domain_data)
        else:
            for i, var in enumerate(input_vars):
                mono_dict[var] = _check_monotonicity(y, X, i)
                conv_dict[var] = _check_convexity(y, X, i)
    else:
        for i, var in enumerate(input_vars):
            mono_dict[var] = _check_monotonicity(y, X, i)
            conv_dict[var] = _check_convexity(y, X, i)

    # --- Periodicity: ALWAYS numerical FFT (same as training) ---
    period_dict = {}
    for i, var in enumerate(input_vars):
        period_dict[var] = _check_periodicity(y, X, i)

    # --- Separability: ALWAYS numerical ANOVA (same as training) ---
    mul_sep = _check_multiplicative_separability(y, X)

    return {
        "name": name,
        "dataset": dataset,
        "n_variables": len(input_vars),
        "variables": input_vars,
        "monotonicity": mono_dict,
        "convexity": conv_dict,
        "periodicity": period_dict,
        "multiplicative_separable": mul_sep,
    }


def main():
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

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_labels_path = OUTPUT_DIR / "all_labels.json"
    with open(all_labels_path, "w") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    print(f"Saved: {all_labels_path}")

    # Per-dataset files
    datasets = set(l["dataset"] for l in labels)
    for ds in sorted(datasets):
        ds_labels = [l for l in labels if l["dataset"] == ds]
        ds_path = OUTPUT_DIR / f"{ds}.json"
        with open(ds_path, "w") as f:
            json.dump(ds_labels, f, indent=2, ensure_ascii=False)
        print(f"Saved: {ds_path} ({len(ds_labels)} formulas)")

    # Print statistics
    print_stats(labels)


def print_stats(labels):
    total_vars = 0
    mono_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    conv_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    period_counts = {0: 0, 1: 0}
    sep_counts = {0: 0, 1: 0}

    for lab in labels:
        for v in lab["monotonicity"].values():
            mono_counts[v] = mono_counts.get(v, 0) + 1
            total_vars += 1
        for v in lab["convexity"].values():
            conv_counts[v] = conv_counts.get(v, 0) + 1
        for v in lab["periodicity"].values():
            period_counts[v] = period_counts.get(v, 0) + 1
        sep_counts[lab["multiplicative_separable"]] += 1

    print(f"\n=== V2 Label Statistics ({len(labels)} formulas, {total_vars} var-pairs) ===")
    print(f"\nMonotonicity (0=default, 1=inc, 2=dec, 3=const):")
    for k in sorted(mono_counts):
        print(f"  {k}: {mono_counts[k]} ({mono_counts[k]/total_vars*100:.1f}%)")
    print(f"\nConvexity (0=default, 1=convex, 2=concave, 3=affine):")
    for k in sorted(conv_counts):
        print(f"  {k}: {conv_counts[k]} ({conv_counts[k]/total_vars*100:.1f}%)")
    print(f"\nPeriodicity (0=non-periodic, 1=periodic):")
    for k in sorted(period_counts):
        print(f"  {k}: {period_counts[k]} ({period_counts[k]/total_vars*100:.1f}%)")
    print(f"\nMul-sep (0=not sep, 1=sep):")
    for k in sorted(sep_counts):
        print(f"  {k}: {sep_counts[k]} ({sep_counts[k]/len(labels)*100:.1f}%)")


if __name__ == "__main__":
    main()
