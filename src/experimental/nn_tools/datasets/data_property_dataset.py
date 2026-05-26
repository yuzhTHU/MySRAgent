# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Dataset that generates (data, property_labels) pairs with rejection sampling.

Each item contains:
  - data: float tensor (sample_num, max_var_num + 1)
  - var_mask: bool tensor (max_var_num,) — True for active variables
  - labels: dict of tensors {monotonicity, convexity, periodicity, mul_sep}

Supports two data sources:
  1. Synthetic formulas via gplearn + nd2py (infinite, with SymPy labels)
  2. LLM-SRBench HDF5 real data (finite, mixed in with probability)

Uses 4-class label encoding for mono/conv.
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
            mono_labels = np.array([lab["monotonicity"].get(v, 0) for v in vars_list[:n_vars]], dtype=np.int64)
            conv_labels = np.array([lab["convexity"].get(v, 0) for v in vars_list[:n_vars]], dtype=np.int64)
            period = np.array([lab["periodicity"].get(v, 0) for v in vars_list[:n_vars]], dtype=np.int64)

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


class DataPropertyDataset(D.Dataset):
    """Generate (data, property_labels) pairs with SymPy labels + rejection sampling.

    Optionally mixes in LLM-SRBench HDF5 real data.
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

    def __len__(self):
        if self.n_samples is None:
            raise TypeError("Infinite dataset has no length; use InfiniteSampler.")
        return self.n_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng((self.random_state, idx) if self.random_state is not None else None)

        if self.srbench_items and rng.random() < self.srbench_mix_ratio:
            return self._get_srbench_item(rng)

        return self._get_synthetic_item(rng)

    def _get_srbench_item(self, rng):
        """Return a random pre-loaded SRBench item (re-sampled)."""
        item = self.srbench_items[rng.integers(0, len(self.srbench_items))]
        return item

    def _get_synthetic_item(self, rng):
        max_reject = 20
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
