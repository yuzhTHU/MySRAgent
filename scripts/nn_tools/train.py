# python ./scripts/nn_tools/train.py
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "experimental"))

import re
import os
import json
import torch
import shlex
import logging
import argparse
import torch.nn as nn
import torch.utils.data as D
from tqdm import tqdm
from datetime import datetime
from socket import gethostname
from torch.nn.utils.rnn import pad_sequence
from sr_agent.utils import setup_logging, add_minus_flags, add_negation_flags, seed_all, tag2ansi, NamedTimer, ParallelTimer
from nn_tools.datasets.generate_eq import BaseEqGenerator
from nn_tools.datasets.generate_data import BaseDataGenerator
from nn_tools.datasets.data_eq_dataset import DataEqDataset, InfiniteSampler
from nn_tools.models import EquationEmbedder, FloatEmbedder, FoundationModel, DataEmbedder

SCRIPT_NAME = Path(__file__).stem
_logger = logging.getLogger(f"sr_agent.{SCRIPT_NAME}")


def unique_parameters(*modules: nn.Module):
    seen = set()
    for module in modules:
        for param in module.parameters():
            if id(param) not in seen:
                seen.add(id(param))
                yield param


def make_prefix_batch(
    index: torch.Tensor,
    data_embedding: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prefixes = []
    targets = []
    data_rows = []
    for row_idx, row in enumerate(index):
        valid = row[row != pad_token_id]
        for next_pos in range(1, valid.numel()):
            prefixes.append(valid[:next_pos])
            targets.append(valid[next_pos])
            data_rows.append(data_embedding[row_idx])

    prefix_index = pad_sequence(prefixes, batch_first=True, padding_value=pad_token_id)
    prefix_padding_mask = prefix_index == pad_token_id
    target = torch.stack(targets)
    repeated_data_embedding = torch.stack(data_rows, dim=0)
    return repeated_data_embedding, prefix_index, prefix_padding_mask, target


def embed_data_batch(data: torch.Tensor, float_embedder: FloatEmbedder, data_embedder: DataEmbedder) -> torch.Tensor:
    batch_size, sample_num = data.shape[:2]
    value_embedding = float_embedder(data).flatten(0, 1)
    data_embedding = data_embedder.pool(value_embedding)
    return data_embedding.reshape(batch_size, sample_num, -1)


def build_dataset(args, eq_generator, data_generator, equation_embedder, *, n_samples, random_state):
    return DataEqDataset(
        max_var_num=args.max_var_num,
        eq_generator=eq_generator,
        data_generator=data_generator,
        n_samples=n_samples,
        random_state=random_state,
        equation_embedder=equation_embedder,
    )


def build_batch_loss(
    batch,
    model: FoundationModel,
    float_embedder: FloatEmbedder,
    equation_embedder: EquationEmbedder,
    data_embedder: DataEmbedder,
    criterion: nn.Module,
    args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = batch["data"].to(args.device)
    index = batch["index"].to(args.device)

    data_embedding = embed_data_batch(data, float_embedder, data_embedder)
    (
        repeated_data_embedding,
        prefix_index,
        prefix_padding_mask,
        target,
    ) = make_prefix_batch(index, data_embedding, equation_embedder.pad_token_id)
    partial_equation_embedding = equation_embedder.symbol_embedding(prefix_index)
    logits = model(
        repeated_data_embedding,
        partial_equation_embedding,
        eq_padding_mask=prefix_padding_mask,
    )
    return criterion(logits, target), logits, target


@torch.no_grad()
def evaluate(eval_loader, model, float_embedder, equation_embedder, data_embedder, criterion, args) -> dict:
    model.eval()
    float_embedder.eval()
    equation_embedder.eval()
    data_embedder.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch in eval_loader:
        loss, logits, target = build_batch_loss(
            batch,
            model,
            float_embedder,
            equation_embedder,
            data_embedder,
            criterion,
            args,
        )
        count = int(target.numel())
        total_loss += loss.item() * count
        total_correct += int((logits.argmax(dim=-1) == target).sum().item())
        total_count += count
    model.train()
    float_embedder.train()
    equation_embedder.train()
    data_embedder.train()
    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
    }


def warn_arg_diff(saved_args, current_args):
    if isinstance(saved_args, argparse.Namespace):
        saved_args = vars(saved_args)
    current_args = vars(current_args)
    for key in sorted(set(saved_args) | set(current_args)):
        val1 = saved_args.get(key, None)
        val2 = current_args.get(key, None)
        if val1 != val2:
            _logger.warning(
                f"Argument '{key}' differs from the saved checkpoint: "
                f"saved_args={val1} vs. current_args={val2}"
            )


def load_checkpoint(
    args,
    model: FoundationModel,
    optimizer: torch.optim.Optimizer,
    *,
    float_embedder: FloatEmbedder,
    equation_embedder: EquationEmbedder,
    data_embedder: DataEmbedder,
    scheduler=None,
) -> tuple[int, dict]:
    if args.reload_checkpoint is not None:
        checkpoint_path = Path(args.reload_checkpoint)
    elif (Path(args.save_path) / "checkpoint.pth").exists():
        checkpoint_path = Path(args.save_path) / "checkpoint.pth"
    else:
        return 0, {}

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")

    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    if "args" in checkpoint:
        warn_arg_diff(checkpoint["args"], args)

    model_state = checkpoint["model"]
    if "foundation_model" in model_state:
        model.load_state_dict(model_state["foundation_model"])
        if "float_embedder" in model_state:
            float_embedder.load_state_dict(model_state["float_embedder"])
        if "equation_embedder" in model_state:
            equation_embedder.load_state_dict(model_state["equation_embedder"])
        if "data_embedder" in model_state:
            data_embedder.load_state_dict(model_state["data_embedder"])
    else:
        model.load_state_dict(model_state)

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        _logger.warning("Optimizer state not found in checkpoint; optimizer re-initialized.")

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    elif scheduler is not None:
        _logger.warning("Scheduler is enabled but scheduler state was not found in checkpoint.")

    step = int(checkpoint.get("step", 0))
    _logger.note(tag2ansi(
        f"Checkpoint loaded from [underline green]{checkpoint_path}[reset], "
        f"resume from step [underline green]{step + 1}[reset]."
    ))
    return step, checkpoint.get("training_state", {})


def main(args):
    # 准备 Generator
    eq_generator = BaseEqGenerator.create(
        args.eq_generator,
        n_variables=args.max_var_num,
        random_seed=args.seed,
        const_range=None,
        depth_range=(args.min_depth, args.max_depth + 1),
        n_var_range=(1, args.max_var_num + 1),
    )
    data_generator = BaseDataGenerator.create(
        args.data_generator,
        sample_num=args.sample_num,
        random_seed=args.seed,
        range=(args.data_min, args.data_max),
    )
    _logger.info(
        f"Equation generator: {args.eq_generator}\n"
        f"Data generator: {args.data_generator}"
    )
    
    # 准备 Embedder
    float_embedder = FloatEmbedder(
        d_model=args.d_model
    ).to(args.device)
    equation_embedder = EquationEmbedder(
        d_model=args.d_model,
        max_variables=args.max_var_num,
        operands=eq_generator.symbols,
    ).to(args.device)
    data_embedder = DataEmbedder(
        d_model=args.d_model,
        float_embedder=float_embedder,
        pooling=args.data_pooling,
    ).to(args.device)
    
    # 准备 Dataset / Dataloader
    train_dataset = build_dataset(
        args,
        eq_generator,
        data_generator,
        equation_embedder,
        n_samples=None,
        random_state=args.seed,
    )
    train_loader = D.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
        sampler=InfiniteSampler(),
    )
    eval_data_generator = BaseDataGenerator.create(
        args.data_generator,
        sample_num=args.sample_num,
        random_seed=args.eval_seed,
        range=(args.data_min, args.data_max),
    )
    eval_dataset = build_dataset(
        args,
        eq_generator,
        eval_data_generator,
        equation_embedder,
        n_samples=args.eval_size,
        random_state=args.eval_seed,
    )
    eval_loader = D.DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=eval_dataset.collate_fn,
        shuffle=False,
    )
    
    # 准备 Model / Optimizer / Criterion / (Scheduler)
    model = FoundationModel(
        d_model=args.d_model,
        vocab_size=equation_embedder.symbol_embedding.num_embeddings,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_formula_len=args.max_formula_len,
        output_pooling=args.output_pooling,
    ).to(args.device)
    trainable_params = list(unique_parameters(
        float_embedder,
        equation_embedder,
        data_embedder,
        model,
    ))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.scheduler_t_max,
            eta_min=args.scheduler_eta_min,
        )
    else:
        scheduler = None
    criterion = nn.CrossEntropyLoss()
    _logger.info(
        f"Model initialized with {sum(p.numel() for p in trainable_params):,} parameters. "
        f"Trainable parameters: {sum(p.numel() for p in trainable_params if p.requires_grad):,}. "
        f"Device: {args.device}."
    )
    start_step, training_state = load_checkpoint(
        args,
        model,
        optimizer,
        float_embedder=float_embedder,
        equation_embedder=equation_embedder,
        data_embedder=data_embedder,
        scheduler=scheduler,
    )

    # 训练模型
    started_at = datetime.now()
    result = {
        "start_time": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": None,
        "status": "running",
        "last_train_loss": None,
        "last_accuracy": None,
        "best_eval_loss": training_state.get("best_eval_loss", None),
        "best_eval_accuracy": training_state.get("best_eval_accuracy", None),
        "best_step": training_state.get("best_step", None),
        "steps": start_step,
        "checkpoint": None,
        "best_checkpoint": training_state.get("best_checkpoint", None),
    }
    total_timer = ParallelTimer() # 总用时统计
    named_timer = NamedTimer() # 细粒度用时统计
    best_eval_loss = training_state.get("best_eval_loss", None)
    if best_eval_loss is None:
        best_eval_loss = float("inf")
    patience_left = training_state.get("patience_left", args.patience)
    last_periodic_checkpoint_time = datetime.now()
    try:
        progress = tqdm(enumerate(train_loader, start=start_step + 1), desc="train", dynamic_ncols=True)
        for step, batch in progress:
            named_timer.add("prepare_data")

            # 前向预测
            loss, logits, target = build_batch_loss(
                batch,
                model,
                float_embedder,
                equation_embedder,
                data_embedder,
                criterion,
                args,
            )
            named_timer.add("forward")

            # 反向传播
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    trainable_params,
                    args.grad_clip,
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            named_timer.add("backward")

            # 后处理
            accuracy = (logits.argmax(dim=-1) == target).float().mean().item()
            result["last_train_loss"] = loss.item()
            result["last_accuracy"] = accuracy
            result["steps"] = step
            if step % args.log_every == 0:
                progress.set_postfix(
                    train_loss=f"{loss.item():.4f}",
                    train_acc=f"{accuracy:.3f}",
                    best_eval="nan" if result["best_eval_loss"] is None else f"{result['best_eval_loss']:.4f}",
                    patience=f"{patience_left}/{args.patience}",
                )
            total_timer.add("step", n=1)
            total_timer.add("eq", n=target.numel())
            if (
                args.save_every_seconds > 0
                and (datetime.now() - last_periodic_checkpoint_time).total_seconds() >= args.save_every_seconds
            ):
                save_checkpoint(
                    Path(args.save_path) / "checkpoints" / f"epoch_{step:06d}.pth",
                    step,
                    args,
                    model,
                    optimizer,
                    float_embedder=float_embedder,
                    equation_embedder=equation_embedder,
                    data_embedder=data_embedder,
                    scheduler=scheduler,
                    training_state={
                        "best_eval_loss": result["best_eval_loss"],
                        "best_eval_accuracy": result["best_eval_accuracy"],
                        "best_step": result["best_step"],
                        "best_checkpoint": result["best_checkpoint"],
                        "patience_left": patience_left,
                    },
                )
                last_periodic_checkpoint_time = datetime.now()

            if step % args.eval_every == 0:
                eval_record = evaluate(
                    eval_loader,
                    model,
                    float_embedder,
                    equation_embedder,
                    data_embedder,
                    criterion,
                    args,
                )
                _logger.info(
                    f"[Step {step}] "
                    f"train_loss={loss.item():.6f}, train_acc={accuracy:.3f}, "
                    f"eval_loss={eval_record['loss']:.6f}, eval_acc={eval_record['accuracy']:.3f}, "
                    f"patience_left={patience_left}/{args.patience}, "
                    f"timer={named_timer}, total={total_timer}"
                )
                if eval_record["loss"] < best_eval_loss:
                    best_eval_loss = eval_record["loss"]
                    patience_left = args.patience
                    best_path = Path(args.save_path) / "best.pth"
                    save_checkpoint(
                        best_path,
                        step,
                        args,
                        model,
                        optimizer,
                        float_embedder=float_embedder,
                        equation_embedder=equation_embedder,
                        data_embedder=data_embedder,
                        scheduler=scheduler,
                        training_state={
                            "best_eval_loss": eval_record["loss"],
                            "best_eval_accuracy": eval_record["accuracy"],
                            "best_step": step,
                            "best_checkpoint": str(best_path),
                            "patience_left": patience_left,
                        },
                    )
                    result["best_eval_loss"] = eval_record["loss"]
                    result["best_eval_accuracy"] = eval_record["accuracy"]
                    result["best_step"] = step
                    result["best_checkpoint"] = str(best_path)
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        result["status"] = "early_stopped"
                        break
                progress.set_postfix(
                    train_loss=f"{loss.item():.4f}",
                    eval_loss=f"{eval_record['loss']:.4f}",
                    eval_acc=f"{eval_record['accuracy']:.3f}",
                    patience=f"{patience_left}/{args.patience}",
                )
            named_timer.add("post_process")

        checkpoint_path = Path(args.save_path) / "checkpoint.pth"
        save_checkpoint(
            checkpoint_path,
            result["steps"],
            args,
            model,
            optimizer,
            float_embedder=float_embedder,
            equation_embedder=equation_embedder,
            data_embedder=data_embedder,
            scheduler=scheduler,
            training_state={
                "best_eval_loss": result["best_eval_loss"],
                "best_eval_accuracy": result["best_eval_accuracy"],
                "best_step": result["best_step"],
                "best_checkpoint": result["best_checkpoint"],
                "patience_left": patience_left,
            },
        )
        result["checkpoint"] = str(checkpoint_path)
        if result["status"] == "running":
            result["status"] = "completed"
    except KeyboardInterrupt:
        _logger.note("Experiment interrupted by user.")
        result["status"] = "interrupted"
        checkpoint_path = Path(args.save_path) / "checkpoint.pth"
        save_checkpoint(
            checkpoint_path,
            result["steps"],
            args,
            model,
            optimizer,
            float_embedder=float_embedder,
            equation_embedder=equation_embedder,
            data_embedder=data_embedder,
            scheduler=scheduler,
            training_state={
                "best_eval_loss": result["best_eval_loss"],
                "best_eval_accuracy": result["best_eval_accuracy"],
                "best_step": result["best_step"],
                "best_checkpoint": result["best_checkpoint"],
                "patience_left": patience_left,
            },
        )
        result["checkpoint"] = str(checkpoint_path)
    finally:
        if result["checkpoint"] is None:
            checkpoint_path = Path(args.save_path) / "checkpoint.pth"
            save_checkpoint(
                checkpoint_path,
                result["steps"],
                args,
                model,
                optimizer,
                float_embedder=float_embedder,
                equation_embedder=equation_embedder,
                data_embedder=data_embedder,
                scheduler=scheduler,
                training_state={
                    "best_eval_loss": result["best_eval_loss"],
                    "best_eval_accuracy": result["best_eval_accuracy"],
                    "best_step": result["best_step"],
                    "best_checkpoint": result["best_checkpoint"],
                    "patience_left": patience_left,
                },
            )
            result["checkpoint"] = str(checkpoint_path)
        result["duration_seconds"] = (datetime.now() - started_at).total_seconds()
        result_path = Path(args.save_path) / "result.jsonl"
        with open(result_path, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=True)
            f.write("\n")
        _logger.note(tag2ansi(
            f'\n[gray]{"=" * 50}[reset]\n'
            "[red bold]Foundation Model Training Result[reset]\n"
            + "\n".join([f"[red]{k.replace('_', ' ').title()}[reset]: {v}" for k, v in result.items()])
            + f'\n[gray]{"=" * 50}[reset]'
        ))
        _logger.note(f"Result saved to {result_path}")
    return result


def save_checkpoint(
    path: Path,
    step: int,
    args: argparse.Namespace,
    model: FoundationModel,
    optimizer: torch.optim.Optimizer,
    *,
    float_embedder: FloatEmbedder | None = None,
    equation_embedder: EquationEmbedder | None = None,
    data_embedder: DataEmbedder | None = None,
    scheduler=None,
    training_state: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_state = {"foundation_model": model.state_dict()}
    if float_embedder is not None:
        model_state["float_embedder"] = float_embedder.state_dict()
    if equation_embedder is not None:
        model_state["equation_embedder"] = equation_embedder.state_dict()
    if data_embedder is not None:
        model_state["data_embedder"] = data_embedder.state_dict()
    checkpoint = {
        "step": step,
        "args": args,
        "model": model_state,
        "optimizer": optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if training_state is not None:
        checkpoint["training_state"] = training_state
    if equation_embedder is not None:
        checkpoint["token2index"] = equation_embedder.token2index
        checkpoint["index2token"] = equation_embedder.index2token
    torch.save(
        checkpoint,
        path,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train FoundationModel on generated data-equation pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", default=f"{SCRIPT_NAME}", help="Experiment task name used when auto-generating exp_name.")
    parser.add_argument("--exp_name", default=None, help="Experiment name. Defaults to a timestamped name.")
    parser.add_argument("--save_dir", default=f"./logs/nn_tools/{SCRIPT_NAME}", help="Root directory for logs and checkpoints.")
    parser.add_argument("--save_path", default=None, help="Path to save logs and checkpoints. Default is auto-generated from --save_dir and --exp_name.")
    parser.add_argument("--reload_checkpoint", default=None, help="Checkpoint path to reload. Defaults to save_path/checkpoint.pth when it exists.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed. Default -1 means using current system time.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument("--debug", action="store_true", default=True, help="Enable debug mode.")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max_var_num", type=int, default=3)
    parser.add_argument("--min_depth", type=int, default=2)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--data_min", type=float, default=-10.0)
    parser.add_argument("--data_max", type=float, default=10.0)
    parser.add_argument("--eq_generator", default="gplearn")
    parser.add_argument("--data_generator", default="uniform")

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_formula_len", type=int, default=256)
    parser.add_argument("--output_pooling", choices=["attention", "average", "last"], default="last")
    parser.add_argument("--data_pooling", choices=["attention", "average", "sum"], default="attention")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every_seconds", type=int, default=300)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--scheduler_t_max", type=int, default=10000)
    parser.add_argument("--scheduler_eta_min", type=float, default=0.0)
    parser = add_minus_flags(parser)
    parser = add_negation_flags(parser)
    return parser


def sanitize_filename(value: str) -> str:
    value = re.compile(r'[ <>:"/\\|?*\x00-\x1f]').sub("_", value.strip())
    return (value or "unnamed")[:255]


def save_args(args, args_path: Path):
    if args_path.exists():
        i = 1
        while args_path.with_suffix(f".json.{i}").exists():
            i += 1
        args_path.rename(args_path.with_suffix(f".json.{i}"))
        _logger.warning(f"args.json already exists, backup to args.json.{i}")
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = build_argparser()
    args, unknown = parser.parse_known_args()

    if args.exp_name is None:
        now = datetime.now()
        args.exp_name = sanitize_filename(
            f"{now:%Y%m%d}_{args.name}_{now:%H%M%S}_{gethostname()}"
        )
    else:
        args.exp_name = sanitize_filename(args.exp_name)
    if args.debug:
        args.verbose = True
    if args.seed == -1:
        args.seed = int(datetime.now().timestamp() * 1000) % (2**32 - 1)
    seed_all(args.seed)
    save_path = Path(args.save_dir) / args.exp_name
    save_path.mkdir(parents=True, exist_ok=True)
    args.save_path = str(save_path)
    args.command = " ".join(map(shlex.quote, [sys.executable, *sys.argv]))

    setup_logging(
        info_level="debug" if args.verbose else "info",
        exp_name=args.exp_name,
        save_path=save_path / "info.log",
        force=True,
    )

    if unknown:
        _logger.warning(f"Unknown args: {unknown}")
    _logger.note(f"Args: {args}")

    save_args(args, save_path / "args.json")

    main(args)
    _logger.note(tag2ansi(f"Experiment completed. Re-run the script with [green bold]{args.command}[reset]"))
