# conda run -n sragent python scripts/nn_tools/train_property.py
"""Train PropertyPredictionModel to predict formula properties from data.

4-class mono/conv, SymPy labels, optional SRBench HDF5 mixing.

Two modes:
  --mode scratch   : train from random initialization
  --mode finetune  : load encoder from a FoundationModel checkpoint, freeze
                     encoder for --freeze_steps then unfreeze
"""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "experimental"))
sys.path.insert(0, str(ROOT / "src"))

import re
import json
import time
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data as D
from datetime import datetime
from socket import gethostname
from nn_tools.datasets.generate_eq import BaseEqGenerator
from nn_tools.datasets.generate_data import BaseDataGenerator
from nn_tools.datasets.data_property_dataset import (
    DataPropertyDataset, InfiniteSampler, _load_srbench_items,
)
from nn_tools.datasets.compute_labels import MONO_CLASSES, CONV_CLASSES
from nn_tools.models import FloatEmbedder, DataEmbedder, PropertyPredictionModel
from sr_agent.utils import setup_logging, seed_all

SCRIPT_NAME = Path(__file__).stem
_logger = logging.getLogger(f"sr_agent.{SCRIPT_NAME}")

DATA_ROOT = ROOT / "data" / "llm-srbench-data"
HDF5_PATH = DATA_ROOT / "lsr_bench_data.hdf5"
LABEL_PATH = DATA_ROOT / "property_label" / "all_labels.json"


def build_dataloader(args, seed, n_samples, batch_size, sampler=None, shuffle=False,
                     srbench_items=None, srbench_mix_ratio=0.0):
    eq_gen = BaseEqGenerator.create(
        args.eq_generator, n_variables=args.max_var_num,
        random_seed=seed, const_range=None,
        depth_range=(args.min_depth, args.max_depth + 1),
        n_var_range=(1, args.max_var_num + 1),
    )
    data_gen = BaseDataGenerator.create(
        args.data_generator, sample_num=args.sample_num,
        random_seed=seed, range=(args.data_min, args.data_max),
    )
    ds = DataPropertyDataset(
        max_var_num=args.max_var_num,
        eq_generator=eq_gen,
        data_generator=data_gen,
        sample_num=args.sample_num,
        n_samples=n_samples,
        random_state=seed,
        max_per_signature=args.max_per_signature,
        range_augment=True,
        use_sympy_labels=True,
        srbench_items=srbench_items,
        srbench_mix_ratio=srbench_mix_ratio,
    )
    return D.DataLoader(
        ds, batch_size=batch_size, num_workers=args.num_workers,
        collate_fn=ds.collate_fn, sampler=sampler, shuffle=shuffle,
    )


def compute_metrics(preds: dict, labels: dict, var_mask: torch.Tensor) -> dict:
    metrics = {}
    for task, n_cls in [("monotonicity", MONO_CLASSES), ("convexity", CONV_CLASSES), ("periodicity", 2)]:
        pred = preds[task].argmax(dim=-1)
        gt = labels[task]
        mask = var_mask
        correct = ((pred == gt) & mask).sum().item()
        total = mask.sum().item()
        metrics[f"{task}_acc"] = correct / max(total, 1)

        f1s = []
        for c in range(n_cls):
            tp = ((pred == c) & (gt == c) & mask).sum().item()
            fp = ((pred == c) & (gt != c) & mask).sum().item()
            fn = ((pred != c) & (gt == c) & mask).sum().item()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            f1s.append(f1)
        metrics[f"{task}_f1"] = np.mean(f1s)

    pred_sep = preds["multiplicative_separable"].argmax(dim=-1)
    gt_sep = labels["mul_sep"]
    metrics["sep_acc"] = (pred_sep == gt_sep).float().mean().item()
    return metrics


def run_step(args, model, float_embedder, data_embedder, batch, criterion_dict, mode="train"):
    data = batch["data"].to(args.device)
    var_mask = batch["var_mask"].to(args.device)
    labels = {k: batch[k].to(args.device) for k in ("monotonicity", "convexity", "periodicity", "mul_sep")}

    B, S = data.shape[:2]
    val_emb = float_embedder(data).flatten(0, 1)
    data_emb = data_embedder.pool(val_emb).reshape(B, S, -1)

    preds = model(data_emb)

    loss = torch.tensor(0.0, device=args.device)
    for task in ("monotonicity", "convexity", "periodicity"):
        logits = preds[task]
        target = labels[task]
        mask = var_mask
        logits_flat = logits[mask]
        target_flat = target[mask]
        if logits_flat.numel() > 0:
            loss = loss + criterion_dict[task](logits_flat, target_flat)

    loss = loss + criterion_dict["sep"](preds["multiplicative_separable"], labels["mul_sep"])

    metrics = compute_metrics(preds, labels, var_mask)
    metrics["loss"] = loss.item()
    return loss, metrics


def main(args):
    float_embedder = FloatEmbedder(d_model=args.d_model).to(args.device)
    data_embedder = DataEmbedder(
        d_model=args.d_model, pooling=args.data_pooling,
        float_embedder=float_embedder,
    ).to(args.device)
    model = PropertyPredictionModel(args).to(args.device)

    if args.mode == "finetune" and args.pretrain_checkpoint:
        _logger.info(f"Loading encoder from {args.pretrain_checkpoint}")
        missing, unexpected = model.load_encoder_from_foundation(args.pretrain_checkpoint, args.device)
        _logger.info(f"  missing={len(missing)}, unexpected={len(unexpected)}")
        ckpt = torch.load(args.pretrain_checkpoint, map_location=args.device, weights_only=False)
        if "float_embedder" in ckpt:
            float_embedder.load_state_dict(ckpt["float_embedder"])
            _logger.info("  loaded float_embedder weights")
        if "data_embedder" in ckpt:
            data_embedder.load_state_dict(ckpt["data_embedder"])
            _logger.info("  loaded data_embedder weights")

    trainable = list(set(
        list(model.parameters()) +
        list(float_embedder.parameters()) +
        list(data_embedder.parameters())
    ))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # 4-class weights — upweight informative classes (inc/dec/const, convex/concave/affine)
    mono_weights = torch.tensor([0.5, 5.0, 5.0, 5.0], device=args.device)
    conv_weights = torch.tensor([0.5, 5.0, 5.0, 5.0], device=args.device)
    period_weights = torch.tensor([1.0, 5.0], device=args.device)
    criterion_dict = {
        "monotonicity": nn.CrossEntropyLoss(weight=mono_weights),
        "convexity": nn.CrossEntropyLoss(weight=conv_weights),
        "periodicity": nn.CrossEntropyLoss(weight=period_weights),
        "sep": nn.CrossEntropyLoss(),
    }

    # Load SRBench HDF5 train data for mixing
    srbench_items = []
    if args.srbench_mix_ratio > 0:
        srbench_items = _load_srbench_items(
            str(HDF5_PATH), str(LABEL_PATH),
            max_var_num=args.max_var_num,
            sample_num=args.sample_num,
            splits=("train",),
            seed=args.seed,
        )
        _logger.info(f"SRBench mix: {len(srbench_items)} items, ratio={args.srbench_mix_ratio}")

    train_loader = build_dataloader(
        args, seed=args.seed, n_samples=None,
        batch_size=args.batch_size, sampler=InfiniteSampler(),
        srbench_items=srbench_items, srbench_mix_ratio=args.srbench_mix_ratio,
    )
    eval_loader = build_dataloader(
        args, seed=args.eval_seed, n_samples=args.eval_size,
        batch_size=args.eval_batch_size,
    )

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    patience_left = args.patience
    history = []

    encoder_frozen = False
    if args.mode == "finetune" and args.freeze_steps > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False
        encoder_frozen = True
        _logger.info(f"Encoder frozen for first {args.freeze_steps} steps")

    _logger.info(f"Starting training, mode={args.mode}, device={args.device}")
    start_time = time.time()

    for step, batch in enumerate(train_loader):
        if step >= args.max_steps:
            break

        if encoder_frozen and step >= args.freeze_steps:
            for p in model.encoder.parameters():
                p.requires_grad = True
            encoder_frozen = False
            _logger.info(f"Encoder unfrozen at step {step}")

        model.train(); float_embedder.train(); data_embedder.train()
        loss, train_metrics = run_step(args, model, float_embedder, data_embedder, batch, criterion_dict, "train")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()

        if step % args.eval_every == 0:
            model.eval(); float_embedder.eval(); data_embedder.eval()
            eval_metrics_list = []
            with torch.no_grad():
                for eval_batch in eval_loader:
                    _, em = run_step(args, model, float_embedder, data_embedder, eval_batch, criterion_dict, "eval")
                    eval_metrics_list.append(em)
            eval_metrics = {k: np.mean([m[k] for m in eval_metrics_list]) for k in eval_metrics_list[0]}

            record = {"step": step, "train": train_metrics, "eval": eval_metrics}
            history.append(record)

            improved = eval_metrics["loss"] < best_loss
            if improved:
                best_loss = eval_metrics["loss"]
                patience_left = args.patience
                torch.save({
                    "step": step, "args": vars(args),
                    "model": model.state_dict(),
                    "float_embedder": float_embedder.state_dict(),
                    "data_embedder": data_embedder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, save_path / "best.pth")

            marker = " *BEST*" if improved else ""
            _logger.info(
                f"[step {step:>6d}] "
                f"train_loss={train_metrics['loss']:.4f}  "
                f"eval_loss={eval_metrics['loss']:.4f}  "
                f"mono_acc={eval_metrics['monotonicity_acc']:.3f}  "
                f"conv_acc={eval_metrics['convexity_acc']:.3f}  "
                f"period_acc={eval_metrics['periodicity_acc']:.3f}  "
                f"sep_acc={eval_metrics['sep_acc']:.3f}  "
                f"mono_f1={eval_metrics['monotonicity_f1']:.3f}  "
                f"conv_f1={eval_metrics['convexity_f1']:.3f}  "
                f"period_f1={eval_metrics['periodicity_f1']:.3f}"
                f"{marker}"
            )

            if not improved:
                patience_left -= 1
                if patience_left <= 0:
                    _logger.info("Early stopping triggered")
                    break

    elapsed = time.time() - start_time
    torch.save({
        "step": step, "args": vars(args),
        "model": model.state_dict(),
        "float_embedder": float_embedder.state_dict(),
        "data_embedder": data_embedder.state_dict(),
    }, save_path / "last.pth")

    with open(save_path / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    _logger.info(f"Training finished in {elapsed:.0f}s, best eval_loss={best_loss:.4f}")
    return history


def build_argparser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["scratch", "finetune"], default="scratch")
    p.add_argument("--pretrain_checkpoint", default=None)
    p.add_argument("--freeze_steps", type=int, default=2000)
    p.add_argument("--name", default=SCRIPT_NAME)
    p.add_argument("--save_dir", default=f"./logs/nn_tools/{SCRIPT_NAME}")
    p.add_argument("--save_path", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")

    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--eval_size", type=int, default=256)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--sample_num", type=int, default=200)
    p.add_argument("--max_var_num", type=int, default=5)
    p.add_argument("--min_depth", type=int, default=2)
    p.add_argument("--max_depth", type=int, default=5)
    p.add_argument("--data_min", type=float, default=-10.0)
    p.add_argument("--data_max", type=float, default=10.0)
    p.add_argument("--eq_generator", default="gplearn")
    p.add_argument("--data_generator", default="uniform")
    p.add_argument("--max_per_signature", type=int, default=50)
    p.add_argument("--srbench_mix_ratio", type=float, default=0.0,
                   help="Probability of returning a SRBench HDF5 item per training sample")

    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_encoder_layers", type=int, default=4)
    p.add_argument("--dim_feedforward", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--data_pooling", choices=["attention", "average", "sum"], default="attention")

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    if args.save_path is None:
        now = datetime.now()
        exp = f"{now:%Y%m%d}_{args.name}_{args.mode}_{now:%H%M%S}_{gethostname()}"
        exp = re.sub(r'[ <>:"/\\|?*\x00-\x1f]', "_", exp)[:255]
        args.save_path = str(Path(args.save_dir) / exp)
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    setup_logging(info_level="debug" if args.verbose else "info", exp_name=args.name,
                  save_path=Path(args.save_path) / "info.log", force=True)
    _logger.info(f"Args: {args}")
    with open(Path(args.save_path) / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    main(args)
