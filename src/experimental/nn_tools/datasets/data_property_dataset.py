# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Dataset v5: generates (data, property_labels) pairs with variable combination augmentation.

Each item contains:
  - data: float tensor (sample_num, max_var_num + 1)
  - var_mask: bool tensor (max_var_num,) — True for active variables
  - labels: dict of tensors {monotonicity, convexity, periodicity, mul_sep}

v5 improvements over v4:
  1. Variable combination augmentation: randomly combine pairs of variables
     (x_i*x_j, x_i+x_j, x_i-x_j, x_i/x_j) and compute properties of y
     w.r.t. the combined variable using numerical methods. Applies to BOTH
     synthetic (gplearn) and SRBench data.
  2. All v4 features retained.
"""
from __future__ import annotations
import json
import torch
import logging
import warnings
import itertools
import numpy as np
import torch.utils.data as D
from pathlib import Path
from typing import Optional, List, Dict
from collections import Counter
from .generate_eq import BaseEqGenerator
from .generate_data import BaseDataGenerator
from .compute_labels import (
    compute_all_labels, compute_all_labels_sympy,
    label_signature, coarse_label_key,
    MONO_CLASSES, CONV_CLASSES,
)

__all__ = ["DataPropertyDataset", "SRBenchPropertyDataset", "InfiniteSampler"]
_logger = logging.getLogger(f"sr_agent.{__name__}")
warnings.filterwarnings("ignore", message="overflow encountered in cast")
warnings.filterwarnings("ignore", message="invalid value encountered in cast")


class InfiniteSampler(D.Sampler):
    def __iter__(self):
        return itertools.count()


def _load_srbench_items(
    hdf5_path: str,
    label_path: str,
    max_var_num: int,
    sample_num: int,
    splits: tuple = ("train",),
    seed: int = 42,
) -> List[Dict]:
    """Pre-load LLM-SRBench HDF5 data as list of dicts matching dataset format."""
    import h5py

    hdf5_path = Path(hdf5_path)
    label_path = Path(label_path)
    if not hdf5_path.exists() or not label_path.exists():
        _logger.warning("SRBench data not found, skipping HDF5 mixing.")
        return []

    labels = json.load(open(label_path))
    ds_map = {
        "lsr_synth_chem_react": "chem_react",
        "lsr_synth_phys_osc": "phys_osc",
        "lsr_synth_matsci": "matsci",
    }

    items = []
    rng = np.random.default_rng(seed)

    with h5py.File(hdf5_path, "r") as f:
        for lab in labels:
            ds_name = lab["dataset"]
            name = lab["name"]
            n_vars = lab["n_variables"]
            if n_vars > max_var_num or n_vars == 0:
                continue

            try:
                if ds_name == "lsr_transform":
                    grp = f["lsr_transform"][name]
                else:
                    grp = f["lsr_synth"][ds_map[ds_name]][name]

                arrays = []
                for split in splits:
                    if split in grp and isinstance(grp[split], h5py.Dataset):
                        arrays.append(grp[split][:])
                if not arrays:
                    continue
                raw = np.concatenate(arrays, axis=0)
            except Exception:
                continue

            mask_fin = np.all(np.isfinite(raw), axis=1)
            raw = raw[mask_fin]
            if raw.shape[0] < 20:
                continue

            S = min(raw.shape[0], sample_num)
            idx = rng.choice(raw.shape[0], S, replace=False)
            sampled = raw[idx]

            data = np.zeros((S, max_var_num + 1), dtype=np.float32)
            for i in range(min(n_vars, sampled.shape[1] - 1)):
                data[:, i] = sampled[:, i + 1].astype(np.float32)
            data[:, -1] = sampled[:, 0].astype(np.float32)

            vars_list = lab["variables"]
            mono_raw = np.array([lab["monotonicity"].get(v, 0) for v in vars_list[:n_vars]], dtype=np.int64)
            conv_raw = np.array([lab["convexity"].get(v, 0) for v in vars_list[:n_vars]], dtype=np.int64)
            period = np.array([lab["periodicity"].get(v, 0) for v in vars_list[:n_vars]], dtype=np.int64)
            mono_labels = np.clip(mono_raw, 0, 3)
            mono_labels[mono_raw == 4] = 3
            conv_labels = np.clip(conv_raw, 0, 3)
            conv_labels[conv_raw == 4] = 3

            mono_padded = np.zeros(max_var_num, dtype=np.int64)
            conv_padded = np.zeros(max_var_num, dtype=np.int64)
            period_padded = np.zeros(max_var_num, dtype=np.int64)
            mono_padded[:n_vars] = mono_labels
            conv_padded[:n_vars] = conv_labels
            period_padded[:n_vars] = period

            var_mask = np.zeros(max_var_num, dtype=bool)
            var_mask[:n_vars] = True

            items.append({
                "data": torch.from_numpy(data),
                "var_mask": torch.from_numpy(var_mask),
                "monotonicity": torch.from_numpy(mono_padded),
                "convexity": torch.from_numpy(conv_padded),
                "periodicity": torch.from_numpy(period_padded),
                "mul_sep": torch.tensor(lab["multiplicative_separable"], dtype=torch.long),
            })

    _logger.info(f"Loaded {len(items)} SRBench items from splits={splits}")
    return items


def _is_trivial_sample(mono_labels, conv_labels, n_vars):
    return (
        all(mono_labels[i] == 3 for i in range(n_vars)) and
        all(conv_labels[i] == 3 for i in range(n_vars))
    )


def _is_all_default(mono_labels, conv_labels, n_vars):
    return (
        all(mono_labels[i] == 0 for i in range(n_vars)) and
        all(conv_labels[i] == 0 for i in range(n_vars))
    )


def _create_combo_item(item: dict, max_var_num: int, rng: np.random.Generator) -> dict | None:
    """Create a new training sample by combining two variables into one.

    Given an item with variables [x0, x1, x2, ...], randomly pick a pair (i, j),
    compute x_combo = x_i op x_j (where op in {*, +, -, /}), then produce a new
    item with variables [x_combo, remaining...]. Labels for x_combo are computed
    numerically from the data; labels for remaining variables are kept as-is.
    """
    data = item["data"]       # (S, max_var_num+1), torch
    var_mask = item["var_mask"]  # (max_var_num,), bool
    n_vars = int(var_mask.sum().item())

    if n_vars < 2:
        return None

    S = data.shape[0]
    y_np = data[:, -1].numpy()

    # Collect active variable columns
    active_cols = [data[:, k].numpy().copy() for k in range(n_vars)]

    # Pick two distinct active variables
    pair = rng.choice(n_vars, size=2, replace=False)
    i, j = int(pair[0]), int(pair[1])

    # Pick combination operation (/ less frequent due to div-by-zero risk)
    op_weights = np.array([0.3, 0.25, 0.25, 0.2])  # *, +, -, /
    op = rng.choice(["*", "+", "-", "/"], p=op_weights)

    xi, xj = active_cols[i], active_cols[j]
    if op == "*":
        x_combo = xi * xj
    elif op == "+":
        x_combo = xi + xj
    elif op == "-":
        x_combo = xi - xj
    else:  # /
        safe = np.abs(xj) > 1e-8
        if safe.sum() < 30:
            return None
        x_combo = np.where(safe, xi / xj, 0.0)

    if not np.all(np.isfinite(x_combo)) or np.std(x_combo) < 1e-12:
        return None

    # Build new variable list: [combo, remaining vars (excluding i and j)]
    new_cols = [x_combo]
    old_indices = []  # tracks which original slot each new var came from
    for k in range(n_vars):
        if k != i and k != j:
            new_cols.append(active_cols[k])
            old_indices.append(k)

    new_n_vars = len(new_cols)
    if new_n_vars > max_var_num:
        return None

    # Build new data matrix
    new_data = np.zeros((S, max_var_num + 1), dtype=np.float32)
    for k, col in enumerate(new_cols):
        new_data[:, k] = col.astype(np.float32)
    new_data[:, -1] = y_np

    # Compute ALL labels numerically for the new variable arrangement
    X_new = new_data[:, :max_var_num]
    labels = compute_all_labels(X_new, y_np, new_n_vars)

    new_var_mask = np.zeros(max_var_num, dtype=bool)
    new_var_mask[:new_n_vars] = True

    mono_padded = np.zeros(max_var_num, dtype=np.int64)
    conv_padded = np.zeros(max_var_num, dtype=np.int64)
    period_padded = np.zeros(max_var_num, dtype=np.int64)
    mono_padded[:new_n_vars] = labels["monotonicity"]
    conv_padded[:new_n_vars] = labels["convexity"]
    period_padded[:new_n_vars] = labels["periodicity"]

    return {
        "data": torch.from_numpy(new_data),
        "var_mask": torch.from_numpy(new_var_mask),
        "monotonicity": torch.from_numpy(mono_padded),
        "convexity": torch.from_numpy(conv_padded),
        "periodicity": torch.from_numpy(period_padded),
        "mul_sep": torch.tensor(labels["multiplicative_separable"], dtype=torch.long),
    }


class DataPropertyDataset(D.Dataset):
    """Generate (data, property_labels) pairs with SymPy labels + augmentations.

    v5: variable combination augmentation for both synthetic and SRBench data.
    """

    def __init__(
        self,
        max_var_num: int,
        eq_generator: BaseEqGenerator,
        data_generator: BaseDataGenerator,
        sample_num: int = 200,
        n_samples: Optional[int] = None,
        random_state: Optional[int] = None,
        max_per_signature: int = 50,
        range_augment: bool = True,
        use_sympy_labels: bool = True,
        srbench_items: Optional[List[Dict]] = None,
        srbench_mix_ratio: float = 0.0,
        noise_std: float = 0.0,
        scale_augment: bool = False,
        permute_vars: bool = False,
        reject_trivial_prob: float = 0.0,
        combo_augment_prob: float = 0.0,
    ):
        self.max_var_num = max_var_num
        self.eq_generator = eq_generator
        self.data_generator = data_generator
        self.sample_num = sample_num
        self.n_samples = n_samples
        self.random_state = random_state
        self.max_per_signature = max_per_signature
        self.range_augment = range_augment
        self.use_sympy_labels = use_sympy_labels
        self.sig_counter: Counter = Counter()
        self.srbench_items = srbench_items or []
        self.srbench_mix_ratio = srbench_mix_ratio if self.srbench_items else 0.0
        self.noise_std = noise_std
        self.scale_augment = scale_augment
        self.permute_vars = permute_vars
        self.reject_trivial_prob = reject_trivial_prob
        self.combo_augment_prob = combo_augment_prob

    def __len__(self):
        if self.n_samples is None:
            raise TypeError("Infinite dataset has no length; use InfiniteSampler.")
        return self.n_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng((self.random_state, idx) if self.random_state is not None else None)

        if self.srbench_items and rng.random() < self.srbench_mix_ratio:
            item = self._get_srbench_item(rng)
        else:
            item = self._get_synthetic_item(rng)

        # v5: variable combination augmentation (applies to BOTH sources)
        if self.combo_augment_prob > 0 and rng.random() < self.combo_augment_prob:
            combo = _create_combo_item(item, self.max_var_num, rng)
            if combo is not None:
                item = combo

        item = self._apply_augmentations(item, rng)
        return item

    def _apply_augmentations(self, item: dict, rng: np.random.Generator) -> dict:
        data = item["data"].clone()
        n_vars = int(item["var_mask"].sum().item())

        if self.noise_std > 0 and n_vars > 0:
            y_col = data[:, -1]
            y_std = y_col.std().item()
            if y_std > 1e-10:
                noise = torch.from_numpy(
                    rng.normal(0, self.noise_std * y_std, size=y_col.shape).astype(np.float32)
                )
                data[:, -1] = y_col + noise

        if self.scale_augment and rng.random() < 0.3:
            scale = float(rng.uniform(0.1, 10.0))
            data[:, -1] *= scale

        if self.permute_vars and n_vars > 1 and rng.random() < 0.3:
            perm = rng.permutation(n_vars)
            new_data = data.clone()
            for new_i, old_i in enumerate(perm):
                new_data[:, new_i] = data[:, old_i]

            mono = item["monotonicity"].clone()
            conv = item["convexity"].clone()
            period = item["periodicity"].clone()
            new_mono = mono.clone()
            new_conv = conv.clone()
            new_period = period.clone()
            for new_i, old_i in enumerate(perm):
                new_mono[new_i] = mono[old_i]
                new_conv[new_i] = conv[old_i]
                new_period[new_i] = period[old_i]

            item = dict(item)
            item["data"] = new_data
            item["monotonicity"] = new_mono
            item["convexity"] = new_conv
            item["periodicity"] = new_period
            return item

        item = dict(item)
        item["data"] = data
        return item

    def _get_srbench_item(self, rng):
        item = self.srbench_items[rng.integers(0, len(self.srbench_items))]
        result = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in item.items()}

        data = result["data"]
        S = data.shape[0]
        if S > 30 and rng.random() < 0.3:
            keep = max(20, int(S * rng.uniform(0.7, 0.9)))
            idx = rng.choice(S, keep, replace=False)
            idx.sort()
            new_data = torch.zeros_like(data)
            new_data[:keep] = data[idx]
            result["data"] = new_data

        return result

    def _get_synthetic_item(self, rng):
        max_reject = 30
        for attempt in range(max_reject):
            eqtree = self.eq_generator(_rng=rng)

            if self.range_augment:
                range_type = rng.integers(0, 5)
                if range_type == 0:
                    lo, hi = rng.uniform(-20, -0.5), rng.uniform(0.5, 20)
                elif range_type == 1:
                    center = rng.uniform(-10, 10)
                    width = rng.uniform(0.5, 3.0)
                    lo, hi = center - width, center + width
                elif range_type == 2:
                    lo = rng.uniform(0.1, 5.0)
                    hi = lo + rng.uniform(0.5, 10.0)
                elif range_type == 3:
                    hi = rng.uniform(-5.0, -0.1)
                    lo = hi - rng.uniform(0.5, 10.0)
                else:
                    lo, hi = -10.0, 10.0

                old_range = self.data_generator.kwargs.get("range", (-10, 10))
                self.data_generator.kwargs["range"] = (lo, hi)
                try:
                    data_dict, target, success = self.data_generator(eqtree, _rng=rng)
                finally:
                    self.data_generator.kwargs["range"] = old_range
            else:
                data_dict, target, success = self.data_generator(eqtree, _rng=rng)

            if not success:
                continue

            variables = sorted(data_dict)
            n_vars = len(variables)
            if n_vars > self.max_var_num or n_vars == 0:
                continue

            N = int(target.shape[0])
            data = np.zeros((N, self.max_var_num + 1), dtype=np.float32)
            for i, v in enumerate(variables):
                data[:, i] = np.asarray(data_dict[v], dtype=np.float32).reshape(-1)
            data[:, -1] = np.asarray(target, dtype=np.float32).reshape(-1)

            X = data[:, :self.max_var_num]
            y = data[:, -1]

            if self.use_sympy_labels:
                expr_str = str(eqtree)
                labels = compute_all_labels_sympy(expr_str, X, y, n_vars)
            else:
                labels = compute_all_labels(X, y, n_vars)

            if self.reject_trivial_prob > 0 and attempt < max_reject - 1:
                if _is_trivial_sample(labels["monotonicity"], labels["convexity"], n_vars):
                    if rng.random() < self.reject_trivial_prob:
                        continue
                if _is_all_default(labels["monotonicity"], labels["convexity"], n_vars):
                    if rng.random() < self.reject_trivial_prob * 0.5:
                        continue

            coarse_key = coarse_label_key(
                labels["monotonicity"], labels["convexity"],
                labels["periodicity"], labels["multiplicative_separable"],
            )
            if self.sig_counter[coarse_key] >= self.max_per_signature and attempt < max_reject - 1:
                continue
            self.sig_counter[coarse_key] += 1

            var_mask = np.zeros(self.max_var_num, dtype=bool)
            var_mask[:n_vars] = True

            mono_padded = np.zeros(self.max_var_num, dtype=np.int64)
            conv_padded = np.zeros(self.max_var_num, dtype=np.int64)
            period_padded = np.zeros(self.max_var_num, dtype=np.int64)
            mono_padded[:n_vars] = labels["monotonicity"]
            conv_padded[:n_vars] = labels["convexity"]
            period_padded[:n_vars] = labels["periodicity"]

            return {
                "data": torch.from_numpy(data),
                "var_mask": torch.from_numpy(var_mask),
                "monotonicity": torch.from_numpy(mono_padded),
                "convexity": torch.from_numpy(conv_padded),
                "periodicity": torch.from_numpy(period_padded),
                "mul_sep": torch.tensor(labels["multiplicative_separable"], dtype=torch.long),
            }

        return self._fallback_item()

    def _fallback_item(self):
        data = torch.zeros(self.sample_num, self.max_var_num + 1)
        return {
            "data": data,
            "var_mask": torch.zeros(self.max_var_num, dtype=torch.bool),
            "monotonicity": torch.zeros(self.max_var_num, dtype=torch.long),
            "convexity": torch.zeros(self.max_var_num, dtype=torch.long),
            "periodicity": torch.zeros(self.max_var_num, dtype=torch.long),
            "mul_sep": torch.tensor(0, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch):
        return {
            "data": torch.stack([b["data"] for b in batch]),
            "var_mask": torch.stack([b["var_mask"] for b in batch]),
            "monotonicity": torch.stack([b["monotonicity"] for b in batch]),
            "convexity": torch.stack([b["convexity"] for b in batch]),
            "periodicity": torch.stack([b["periodicity"] for b in batch]),
            "mul_sep": torch.stack([b["mul_sep"] for b in batch]),
        }

    def get_sampler(self):
        return InfiniteSampler() if self.n_samples is None else None


class SRBenchPropertyDataset(D.Dataset):
    """Finite dataset loading LLM-SRBench HDF5 data with GT labels (4-class encoding)."""

    def __init__(
        self,
        hdf5_path: str,
        label_path: str,
        max_var_num: int,
        sample_num: int = 200,
        splits: tuple = ("train",),
        seed: int = 42,
    ):
        self.items = _load_srbench_items(
            hdf5_path, label_path, max_var_num, sample_num, splits, seed,
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
