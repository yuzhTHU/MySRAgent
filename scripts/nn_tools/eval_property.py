# conda run -n sragent python scripts/nn_tools/eval_property.py --checkpoint <path>
"""Evaluate PropertyPredictionModel on four test sets (4-class encoding):

A) New synthetic formulas (gplearn, balanced rejection sampling)
B) Seen-formula / new-range (gplearn, same seed but different sampling range)
C) LLM-SRBench real formulas — HDF5 **test** split only
D) LLM-SRBench real formulas — HDF5 **ood_test** split only (where available)
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "experimental"))
sys.path.insert(0, str(ROOT / "src"))

import json
import h5py
import torch
import logging
import argparse
import numpy as np
from nn_tools.models import FloatEmbedder, DataEmbedder, PropertyPredictionModel
from nn_tools.datasets.generate_eq import BaseEqGenerator
from nn_tools.datasets.generate_data import BaseDataGenerator
from nn_tools.datasets.data_property_dataset import DataPropertyDataset
from nn_tools.datasets.compute_labels import (
    compute_all_labels, MONO_CLASSES, CONV_CLASSES,
)

try:
    import datasets as hf_datasets
except ImportError:
    hf_datasets = None

_logger = logging.getLogger("sr_agent.eval_property")

DATA_ROOT = ROOT / "data" / "llm-srbench-data"
HDF5_PATH = DATA_ROOT / "lsr_bench_data.hdf5"
LABEL_PATH = DATA_ROOT / "property_label" / "all_labels.json"


def load_model(args, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = argparse.Namespace(**ckpt["args"]) if isinstance(ckpt["args"], dict) else ckpt["args"]
    for attr in ("d_model", "nhead", "num_encoder_layers", "dim_feedforward", "dropout", "max_var_num", "data_pooling"):
        setattr(args, attr, getattr(saved_args, attr, getattr(args, attr, None)))
    float_emb = FloatEmbedder(d_model=args.d_model).to(device)
    data_emb = DataEmbedder(d_model=args.d_model, pooling=args.data_pooling, float_embedder=float_emb).to(device)
    model = PropertyPredictionModel(args).to(device)
    model.load_state_dict(ckpt["model"])
    float_emb.load_state_dict(ckpt["float_embedder"])
    data_emb.load_state_dict(ckpt["data_embedder"])
    model.eval(); float_emb.eval(); data_emb.eval()
    return model, float_emb, data_emb


def predict_one(model, float_emb, data_emb, data_tensor, device):
    with torch.no_grad():
        data = data_tensor.to(device)
        B, S = data.shape[:2]
        val_emb = float_emb(data).flatten(0, 1)
        d_emb = data_emb.pool(val_emb).reshape(B, S, -1)
        out = model(d_emb)
    return {k: v.cpu() for k, v in out.items()}


def per_task_metrics(all_preds, all_gts, all_masks, task, n_classes):
    pred = np.concatenate(all_preds)
    gt = np.concatenate(all_gts)
    mask = np.concatenate(all_masks)
    pred_m = pred[mask]
    gt_m = gt[mask]
    acc = (pred_m == gt_m).mean() if len(gt_m) > 0 else 0.0
    f1s = []
    per_class = {}
    for c in range(n_classes):
        tp = ((pred_m == c) & (gt_m == c)).sum()
        fp = ((pred_m == c) & (gt_m != c)).sum()
        fn = ((pred_m != c) & (gt_m == c)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        f1s.append(f1)
        per_class[int(c)] = {"precision": float(prec), "recall": float(rec), "f1": float(f1),
                             "support": int((gt_m == c).sum())}
    return {"accuracy": float(acc), "macro_f1": float(np.mean(f1s)), "per_class": per_class}


def eval_synthetic(args, model, float_emb, data_emb, seed, n_samples, test_name):
    """Test set A or B: synthetic formulas (labels via SymPy)."""
    eq_gen = BaseEqGenerator.create(
        "gplearn", n_variables=args.max_var_num, random_seed=seed,
        const_range=None, depth_range=(1, 5), n_var_range=(1, args.max_var_num + 1),
    )
    data_gen = BaseDataGenerator.create(
        "uniform", sample_num=args.sample_num, random_seed=seed, range=(args.data_min, args.data_max),
    )
    ds = DataPropertyDataset(
        max_var_num=args.max_var_num, eq_generator=eq_gen, data_generator=data_gen,
        sample_num=args.sample_num, n_samples=n_samples, random_state=seed,
        max_per_signature=9999, range_augment=(test_name != "seen_formula_new_range"),
        use_sympy_labels=True,
    )

    preds_dict = {t: [] for t in ("monotonicity", "convexity", "periodicity")}
    gts_dict = {t: [] for t in ("monotonicity", "convexity", "periodicity")}
    masks_all = []
    sep_preds, sep_gts = [], []

    for i in range(n_samples):
        item = ds[i]
        data_t = item["data"].unsqueeze(0)
        out = predict_one(model, float_emb, data_emb, data_t, args.device)
        mask = item["var_mask"].numpy()
        masks_all.append(mask)
        for t in ("monotonicity", "convexity", "periodicity"):
            preds_dict[t].append(out[t][0].argmax(dim=-1).numpy())
            gts_dict[t].append(item[t].numpy())
        sep_preds.append(out["multiplicative_separable"][0].argmax().item())
        sep_gts.append(item["mul_sep"].item())

    results = {}
    for t, nc in [("monotonicity", MONO_CLASSES), ("convexity", CONV_CLASSES), ("periodicity", 2)]:
        results[t] = per_task_metrics(preds_dict[t], gts_dict[t], masks_all, t, nc)
    sep_preds = np.array(sep_preds)
    sep_gts = np.array(sep_gts)
    results["sep_acc"] = float((sep_preds == sep_gts).mean())
    return results


def eval_llm_srbench(args, model, float_emb, data_emb, splits=("test",), test_name="test"):
    """Test set C/D: real LLM-SRBench formulas from specific HDF5 splits."""
    if not LABEL_PATH.exists() or not HDF5_PATH.exists():
        _logger.warning("LLM-SRBench data not found, skipping.")
        return None
    labels = json.load(open(LABEL_PATH))

    preds_dict = {t: [] for t in ("monotonicity", "convexity", "periodicity")}
    gts_dict = {t: [] for t in ("monotonicity", "convexity", "periodicity")}
    masks_all = []
    sep_preds, sep_gts = [], []

    ds_map = {"lsr_synth_chem_react": "chem_react", "lsr_synth_phys_osc": "phys_osc", "lsr_synth_matsci": "matsci"}
    skipped = 0

    with h5py.File(HDF5_PATH, "r") as f:
        for lab in labels:
            ds_name = lab["dataset"]
            name = lab["name"]
            n_vars = lab["n_variables"]
            if n_vars > args.max_var_num:
                skipped += 1
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
                    skipped += 1
                    continue
                raw = np.concatenate(arrays, axis=0)
            except Exception:
                skipped += 1
                continue

            mask_fin = np.all(np.isfinite(raw), axis=1)
            raw = raw[mask_fin]
            if raw.shape[0] < 10:
                skipped += 1
                continue

            S = min(raw.shape[0], args.sample_num)
            idx = np.random.choice(raw.shape[0], S, replace=False) if raw.shape[0] > S else np.arange(S)
            sampled = raw[idx]

            data = np.zeros((S, args.max_var_num + 1), dtype=np.float32)
            for i in range(min(n_vars, sampled.shape[1] - 1)):
                data[:, i] = sampled[:, i + 1]
            data[:, -1] = sampled[:, 0]
            data_t = torch.from_numpy(data).unsqueeze(0)

            out = predict_one(model, float_emb, data_emb, data_t, args.device)
            mask = np.zeros(args.max_var_num, dtype=bool)
            mask[:n_vars] = True
            masks_all.append(mask)

            vars_list = lab["variables"]
            mono_labels = np.array([lab["monotonicity"].get(v, 0) for v in vars_list[:n_vars]], dtype=np.int64)
            conv_labels = np.array([lab["convexity"].get(v, 0) for v in vars_list[:n_vars]], dtype=np.int64)

            mono_gt = np.zeros(args.max_var_num, dtype=np.int64)
            conv_gt = np.zeros(args.max_var_num, dtype=np.int64)
            period_gt = np.zeros(args.max_var_num, dtype=np.int64)
            mono_gt[:n_vars] = mono_labels
            conv_gt[:n_vars] = conv_labels
            for i, v in enumerate(vars_list[:n_vars]):
                if i < args.max_var_num:
                    period_gt[i] = lab["periodicity"].get(v, 0)

            preds_dict["monotonicity"].append(out["monotonicity"][0].argmax(dim=-1).numpy())
            preds_dict["convexity"].append(out["convexity"][0].argmax(dim=-1).numpy())
            preds_dict["periodicity"].append(out["periodicity"][0].argmax(dim=-1).numpy())
            gts_dict["monotonicity"].append(mono_gt)
            gts_dict["convexity"].append(conv_gt)
            gts_dict["periodicity"].append(period_gt)
            sep_preds.append(out["multiplicative_separable"][0].argmax().item())
            sep_gts.append(lab["multiplicative_separable"])

    if not masks_all:
        _logger.warning(f"No valid formulas for {test_name} (skipped={skipped})")
        return None

    _logger.info(f"  {test_name}: evaluated {len(masks_all)} formulas (skipped {skipped})")

    results = {}
    for t, nc in [("monotonicity", MONO_CLASSES), ("convexity", CONV_CLASSES), ("periodicity", 2)]:
        results[t] = per_task_metrics(preds_dict[t], gts_dict[t], masks_all, t, nc)
    sep_preds = np.array(sep_preds)
    sep_gts = np.array(sep_gts)
    results["sep_acc"] = float((sep_preds == sep_gts).mean())
    return results


def main(args):
    model, float_emb, data_emb = load_model(args, args.checkpoint, args.device)
    _logger.info(f"Model loaded from {args.checkpoint}")

    results = {}

    _logger.info("=== Test A: New synthetic formulas ===")
    results["test_a_new_synthetic"] = eval_synthetic(
        args, model, float_emb, data_emb, seed=9999, n_samples=args.n_test, test_name="new_synthetic")
    for t in ("monotonicity", "convexity", "periodicity"):
        r = results["test_a_new_synthetic"][t]
        _logger.info(f"  {t}: acc={r['accuracy']:.3f}, macro_f1={r['macro_f1']:.3f}")
    _logger.info(f"  sep_acc={results['test_a_new_synthetic']['sep_acc']:.3f}")

    _logger.info("=== Test B: Seen formulas, new range ===")
    results["test_b_seen_new_range"] = eval_synthetic(
        args, model, float_emb, data_emb, seed=args.train_seed, n_samples=args.n_test, test_name="seen_formula_new_range")
    for t in ("monotonicity", "convexity", "periodicity"):
        r = results["test_b_seen_new_range"][t]
        _logger.info(f"  {t}: acc={r['accuracy']:.3f}, macro_f1={r['macro_f1']:.3f}")
    _logger.info(f"  sep_acc={results['test_b_seen_new_range']['sep_acc']:.3f}")

    _logger.info("=== Test C: LLM-SRBench (test split) ===")
    results["test_c_srbench_test"] = eval_llm_srbench(
        args, model, float_emb, data_emb, splits=("test",), test_name="srbench_test")
    if results["test_c_srbench_test"]:
        for t in ("monotonicity", "convexity", "periodicity"):
            r = results["test_c_srbench_test"][t]
            _logger.info(f"  {t}: acc={r['accuracy']:.3f}, macro_f1={r['macro_f1']:.3f}")
        _logger.info(f"  sep_acc={results['test_c_srbench_test']['sep_acc']:.3f}")

    _logger.info("=== Test D: LLM-SRBench (ood_test split) ===")
    results["test_d_srbench_ood"] = eval_llm_srbench(
        args, model, float_emb, data_emb, splits=("ood_test",), test_name="srbench_ood_test")
    if results["test_d_srbench_ood"]:
        for t in ("monotonicity", "convexity", "periodicity"):
            r = results["test_d_srbench_ood"][t]
            _logger.info(f"  {t}: acc={r['accuracy']:.3f}, macro_f1={r['macro_f1']:.3f}")
        _logger.info(f"  sep_acc={results['test_d_srbench_ood']['sep_acc']:.3f}")

    out_path = Path(args.checkpoint).parent / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    _logger.info(f"Results saved to {out_path}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_test", type=int, default=500)
    p.add_argument("--sample_num", type=int, default=200)
    p.add_argument("--data_min", type=float, default=-10.0)
    p.add_argument("--data_max", type=float, default=10.0)
    p.add_argument("--train_seed", type=int, default=42)
    p.add_argument("--max_var_num", type=int, default=5)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_encoder_layers", type=int, default=4)
    p.add_argument("--dim_feedforward", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--data_pooling", default="attention")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(args)
