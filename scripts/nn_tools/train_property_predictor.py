# python ./scripts/nn_tools/train_property_predictor.py
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "experimental"))

import re
import json
import torch
import shlex
import logging
import argparse
import torch.nn as nn
import torch.utils.data as D
from datetime import datetime
from socket import gethostname
from typing import Any, Dict, Optional
from torch.nn.utils.rnn import pad_sequence
from nn_tools.datasets.generate_eq import BaseEqGenerator
from nn_tools.datasets.generate_data import BaseDataGenerator
from nn_tools.datasets.eq_property_dataset import EqPropertyDataset, InfiniteSampler
from nn_tools.models import EquationEmbedder, FloatEmbedder, PropertyPredictor, DataEmbedder
from sr_agent.utils import setup_logging, add_minus_flags, add_negation_flags, seed_all, tag2ansi, NamedTimer, ParallelTimer, log_exception, format_confusion_matrix, load_model_state

SCRIPT_NAME = Path(__file__).stem
_logger = logging.getLogger(f"sr_agent.{SCRIPT_NAME}")


PROPERTY_SPECS = [
    {"name": "exists",             "scope": "variable",      "task": "binary"},
    {"name": "multiple_splitable", "scope": "variable_pair", "task": "binary"},
    # {"name": "target_mean",        "scope": "target",        "task": "regression"}, # 不太容易预测，不要这个
]


def run_model(
    args, model, criterion, optimizer, scheduler,
    float_embedder, equation_embedder, data_embedder,
    mode, trainable_params, batch=None, data_loader=None,
) -> Dict[str, Any]:
    if not (batch is None) ^ (data_loader is None):
        raise ValueError("Exactly one of batch or data_loader must be provided.")
    elif batch is not None:
        batches = (batch,)
    else:
        batches = data_loader
    if mode not in {'train', 'test'}:
        raise ValueError(f"Invalid mode: {mode!r}. Must be 'train' or 'test'.")
    elif mode == 'train':
        for m in (model, float_embedder, equation_embedder, data_embedder): m.train()
    else:
        for m in (model, float_embedder, equation_embedder, data_embedder): m.eval()

    all_loss = []
    eq_count = 0
    details = {}
    for spec in PROPERTY_SPECS:
        key = spec['name']
        if spec['task'] == 'regression':
            details[key] = {"true": [], 'pred': []}
        else:
            details[key] = {'true': [], 'pred': []}
    with torch.set_grad_enabled(mode == "train"):
        for current_batch in batches:
            data = current_batch["data"].to(args.device)

            batch_size, sample_num = data.shape[:2]
            value_embedding = float_embedder(data).flatten(0, 1)
            data_embedding = data_embedder.pool(value_embedding).reshape(batch_size, sample_num, -1)
            outputs = model(data_embedding)

            loss = data_embedding.new_tensor(0.0)
            for spec in PROPERTY_SPECS:
                name = spec["name"]
                task = spec["task"]
                scope_size = {"target": 1, "variable": args.max_var_num, "variable_pair": args.max_var_num ** 2}[spec["scope"]]
                label_rows = [row[name] for row in current_batch["properties"]]
                if task == "regression":
                    pred = outputs[name].reshape(batch_size, scope_size, 1)
                    true = torch.as_tensor(label_rows, dtype=pred.dtype, device=args.device)
                    property_loss = nn.functional.mse_loss(pred, true)
                    details[name]["true"].extend(true.cpu().tolist())
                    details[name]["pred"].extend(pred.detach().cpu().tolist())
                else:
                    task_size = {"binary": 2, "ternary": 3, "quaternary": 4}[task]
                    logits = outputs[name].reshape(batch_size, scope_size, task_size)
                    true = torch.as_tensor(label_rows, dtype=torch.long, device=args.device)
                    property_loss = nn.functional.cross_entropy(logits.reshape(-1, task_size), true.reshape(-1))
                    details[name]['true'].extend(true.cpu().tolist())
                    details[name]['pred'].extend(logits.argmax(dim=-1).cpu().tolist())
                loss = loss + property_loss
                eq_count += batch_size

            if mode == 'train':
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            all_loss.append(loss.detach())

    for spec in PROPERTY_SPECS:
        name = spec["name"]
        record = details[name]
        true = torch.as_tensor(record.pop("true"), dtype=torch.float32)
        pred = torch.as_tensor(record.pop("pred"), dtype=torch.float32)
        if spec["task"] == "regression":
            ss_res = ((pred - true) ** 2).sum()
            ss_tot = ((true - true.mean()) ** 2).sum()
            details[name]["r2"] = None if ss_tot.item() == 0.0 else (1 - ss_res / ss_tot).item()
        else:
            task_size = {"binary": 2, "ternary": 3, "quaternary": 4}[spec["task"]]
            true = true.long().reshape(-1).tolist()
            pred = pred.long().reshape(-1).tolist()
            confusion_matrix = {}
            for t, p in zip(true, pred):
                confusion_matrix.setdefault(str(t), {})
                confusion_matrix[str(t)].setdefault(str(p), 0)
                confusion_matrix[str(t)][str(p)] += 1
            details[name]["confusion_matrix"] = confusion_matrix

    return {
        "mode": mode,
        "loss": torch.stack(all_loss).mean().item(),
        "eq_count": eq_count,
        "details": details,
    }


def log_epoch(args, train_record, test_record, states) -> str:
    def append_detail_metrics(lines, prefix, record):
        if record is None:
            return
        for name, metrics in record.get("details", {}).items():
            if "confusion_matrix" in metrics:
                confution_matrix_str = format_confusion_matrix(metrics["confusion_matrix"])
                lines.append(f"{prefix} {name} {confution_matrix_str}")
            elif "r2" in metrics:
                r2 = metrics["r2"]
                r2_str = "N/A" if r2 is None else f"{r2:.6f}"
                lines.append(f"{prefix} {name} R2=[cyan]{r2_str}[reset]")

    lines = [f"Step=[bold blue]{states['step']}"]
    if train_record is not None:
        lines.append(f"Train Loss=[green]{train_record['loss']:.6f}[reset]")
        append_detail_metrics(lines, "Train", train_record)
    if test_record is not None:
        lines.append(f"Eval Loss=[red]{test_record['loss']:.6f}[reset]")
        append_detail_metrics(lines, "Eval", test_record)
    if states.get("patience") is not None:
        lines.append(f"Patience=[magenta]{states['patience']}/{args.patience}[reset]")
    if states.get('named_timer') is not None:
        time_usage = states['named_timer'].to_str(mode='time', mode_of_detail='pace', mode_of_percent='by_time')
        lines.append(f"Time Usage=[gray]{time_usage}[reset]")
    if states.get('total_timer') is not None:
        speed = states['total_timer'].to_str(mode='time', mode_of_detail='speed', mode_of_percent=None)
        lines.append(f"Speed=[gray]{speed}[reset]")
    return tag2ansi("\n".join(lines))


def save_checkpoint(
    save_path, step, args, model, optimizer, scheduler, states,
    float_embedder, equation_embedder, data_embedder,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "args": args,
        "model": model.state_dict(),
        'float_embedder': float_embedder.state_dict(),
        "equation_embedder": equation_embedder.state_dict(),
        "data_embedder": data_embedder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "states": states,
    }, save_path)


def build_dataloader(args, seed, n_samples, batch_size, equation_embedder, sampler=None, shuffle=False):
    eq_generator = BaseEqGenerator.create(
        args.eq_generator,
        n_variables=args.max_var_num,
        random_seed=seed,
        const_range=None,
        depth_range=(args.min_depth, args.max_depth + 1),
        n_var_range=(1, args.max_var_num + 1),
    )
    data_generator = BaseDataGenerator.create(
        args.data_generator,
        sample_num=args.sample_num,
        random_seed=seed,
        range=(args.data_min, args.data_max),
    )
    dataset = EqPropertyDataset(
        max_var_num=args.max_var_num,
        eq_generator=eq_generator,
        data_generator=data_generator,
        n_samples=n_samples,
        random_state=seed,
        equation_embedder=equation_embedder,
    )
    dataloader = D.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        sampler=sampler,
        shuffle=shuffle,
    )
    return dataloader


def main(args):
    ## 准备训练 & 测试数据 —— 已经过人类审核，不得擅自修改
    if True:    
        # 准备 Embedder
        tmp_eq_generator = BaseEqGenerator.create(
            args.eq_generator,
            n_variables=args.max_var_num,
            random_seed=args.seed,
            const_range=None,
            depth_range=(args.min_depth, args.max_depth + 1),
            n_var_range=(1, args.max_var_num + 1),
        )
        float_embedder = FloatEmbedder(
            d_model=args.d_model
        ).to(args.device)
        equation_embedder = EquationEmbedder(
            d_model=args.d_model, 
            operands=tmp_eq_generator.symbols,
            max_variables=args.max_var_num, 
        ).to(args.device)
        data_embedder = DataEmbedder(
            d_model=args.d_model,
            pooling=args.data_pooling,
            float_embedder=float_embedder,
        ).to(args.device)
        _logger.info(f"Embedders initialized. Vocab size={equation_embedder.num_symbol_embeddings}")
    
        # 准备 Dataset / Dataloader
        train_loader = build_dataloader(
            args=args, seed=args.seed, n_samples=None,
            batch_size=args.batch_size,
            equation_embedder=equation_embedder,
            sampler=InfiniteSampler(),
        )
        eval_loader = build_dataloader(
            args=args, seed=args.eval_seed, n_samples=args.eval_size,
            batch_size=args.eval_batch_size,
            equation_embedder=equation_embedder,
            shuffle=False,
        )
        test_loader = build_dataloader(
            args=args, seed=666, n_samples=512,
            batch_size=args.eval_batch_size,
            equation_embedder=equation_embedder,
            shuffle=False,
        )
        _logger.info(f"Equation generator={args.eq_generator}, Data generator={args.data_generator}")

    
    ## 准备 Model / Optimizer / Criterion / (Scheduler) —— 已经过人类审核，不得擅自修改
    if True:
        # Model
        args.vocab_size = equation_embedder.num_symbol_embeddings
        model = PropertyPredictor(args=args, properties=PROPERTY_SPECS).to(args.device)

        # Params
        trainable_params = {}
        for module in (float_embedder, equation_embedder, data_embedder, model):
            for param in module.parameters():
                if id(param) not in trainable_params:
                    trainable_params[id(param)] = param
        trainable_params = list(trainable_params.values())

        # Optimizer
        optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )

        # Scheduler
        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=args.scheduler_t_max, 
                eta_min=args.scheduler_eta_min
            )
        else:
            scheduler = None
        
        # Criterion is chosen per property inside compute_property_loss.
        criterion = None
        _logger.info(
            f"Model initialized with {sum(p.numel() for p in trainable_params):,} parameters. "
            f"Trainable parameters: {sum(p.numel() for p in trainable_params if p.requires_grad):,}. "
            f"Device: {args.device}."
        )
    
    ## 加载检查点 & 恢复训练状态 (如果有的话) —— 已经过人类审核，不得擅自修改
    if True:
        if args.reload_checkpoint is None:
            checkpoint_path = Path(args.save_path) / "checkpoint.pth"
        elif Path(args.reload_checkpoint).exists():
            checkpoint_path = Path(args.reload_checkpoint)
        else:
            raise ValueError(f"Checkpoint path {args.reload_checkpoint!r} does not exist.")
        if not checkpoint_path.exists():
            states = {}
        else:
            # Saved args
            checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
            if (saved_args := vars(checkpoint["args"])) is not None:
                current_args = vars(args)
                for key in sorted(set(saved_args) | set(current_args)):
                    saved_value, current_value = saved_args.get(key), current_args.get(key)
                    if saved_value != current_value:
                        _logger.warning(
                            f"Argument {key!r} differs from the saved checkpoint: "
                            f"saved_args={saved_value!r} vs. current_args={current_value!r}"
                        )
            # Saved embedders, model, optimizer, and scheduler            
            if 'float_embedder' in checkpoint and float_embedder is not None:
                float_embedder.load_state_dict(checkpoint['float_embedder'])
            if 'equation_embedder' in checkpoint and equation_embedder is not None:
                equation_embedder.load_state_dict(checkpoint['equation_embedder'])
            if 'data_embedder' in checkpoint and data_embedder is not None:
                data_embedder.load_state_dict(checkpoint['data_embedder'])
            if 'model' in checkpoint:
                missing_keys, unexpected_keys, mismatched_keys = load_model_state(model, checkpoint['model'])
                checkpoint_is_full_resume = not missing_keys and not unexpected_keys and not mismatched_keys
                if missing_keys:
                    _logger.warning(
                        "Model checkpoint is missing keys for this PropertyPredictor; "
                        f"these parameters keep their random initialization: {missing_keys}"
                    )
                if unexpected_keys:
                    _logger.warning(
                        "Model checkpoint has keys not used by this PropertyPredictor; "
                        f"ignored keys: {unexpected_keys}"
                    )
                if mismatched_keys:
                    _logger.warning(
                        "Model checkpoint has shape-mismatched keys for this PropertyPredictor; "
                        f"these parameters keep their random initialization: {mismatched_keys}"
                    )
            else:
                checkpoint_is_full_resume = False
                _logger.warning("Model state not found in checkpoint; model re-initialized.")

            if checkpoint_is_full_resume and 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            elif 'optimizer' in checkpoint:
                _logger.warning("Optimizer state skipped because the checkpoint is not a full PropertyPredictor resume.")
            else:
                _logger.warning("Optimizer state not found in checkpoint; optimizer re-initialized.")

            if checkpoint_is_full_resume and checkpoint.get('scheduler') is not None and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
            elif checkpoint_is_full_resume and scheduler is not None:
                _logger.warning("Scheduler is enabled but scheduler state was not found in checkpoint.")
            elif scheduler is not None:
                _logger.warning("Scheduler state skipped because the checkpoint is not a full PropertyPredictor resume.")

            if checkpoint_is_full_resume:
                states = checkpoint["states"]
                _logger.note(tag2ansi(
                    f"Checkpoint loaded from [underline green]{checkpoint_path}[reset], "
                    f"resume from step [underline green]{states['step']}[reset]."
                ))
            else:
                states = {}
                _logger.note(tag2ansi(
                    f"Checkpoint loaded from [underline green]{checkpoint_path}[reset], "
                    f"but not fully compatible with the current model/optimizer/scheduler; "
                    f"training resumed with random initialization."
                ))

    ## 训练循环
    start_time = datetime.now()
    states.setdefault("step", 0) # 当前训练步
    states.setdefault('total_timer', ParallelTimer(unit=''))  # 总用时统计
    states.setdefault('named_timer', NamedTimer())  # 细粒度用时统计
    states.setdefault("patience", args.patience)       # 早停耐心值
    states.setdefault("best_record", {"loss": None, "step": None})
    start_step = states["step"]
    result = {'status': 'running'}
    try:
        named_timer = states['named_timer']
        total_timer = states['total_timer']

        for step, batch in enumerate(train_loader, start=start_step):
            states['step'] = step
            named_timer.add("prepare_data")

            ## 训练模型
            if args.test_before_train and states['step'] == 0:
                train_record = None
            else:
                train_record = run_model(
                    args,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    float_embedder,
                    equation_embedder,
                    data_embedder,
                    mode='train',
                    trainable_params=trainable_params,
                    batch=batch,
                )
                named_timer.add("train")

                # 后处理
                total_timer.add("step", n=1)
                total_timer.add("eq", n=train_record["eq_count"])

                # 保存检查点
                if states['step'] >= 100 and set(str(states['step'])[1:]) == {'0'}: # 只在 100, 200, ..., 1000, 2000, ... 这样的整数倍步数保存
                    save_checkpoint(
                        Path(args.save_path) / "checkpoints" / f"epoch_{states['step']:06d}.pth",
                        states['step'], args, model, optimizer, scheduler, states,
                        float_embedder, equation_embedder, data_embedder,
                    )
                named_timer.add("post_process")

            ## 评估模型
            if states['step'] % args.eval_every != 0:
                eval_record = None
            else:
                # 计算评测指标
                eval_record = run_model(
                    args,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    float_embedder,
                    equation_embedder,
                    data_embedder,
                    mode='test',
                    trainable_params=trainable_params,
                    data_loader=eval_loader,
                )

                # 更新最佳记录 & 耐心值
                best_record = states["best_record"]
                if best_record["loss"] is None or eval_record["loss"] < best_record["loss"]:
                    states["best_record"] = {**eval_record, "step": step}
                    states["patience"] = args.patience
                    save_checkpoint(
                        Path(args.save_path) / "best.pth",
                        step, args, model, optimizer, scheduler, states,
                        float_embedder, equation_embedder, data_embedder,
                    )
                    log = log_epoch(args, train_record, eval_record, states)
                    _logger.note(f"Best record updated.\n{log}")
                elif states["patience"] > 0:
                    states["patience"] -= 1
                    log = log_epoch(args, train_record, eval_record, states)
                    _logger.info(f"Patience decreased.\n{log}")
                else:
                    log = log_epoch(args, train_record, eval_record, states)
                    _logger.info(f"Early stopped.\n{log}")
                    result['status'] = 'early_stopped'
                    break
                named_timer.add("eval")

                # 保存结果
                with open(Path(args.save_path) / "records.jsonl", "a", encoding="utf-8") as f:
                    if train_record is not None:
                        json.dump({**train_record, "step": states['step'], 'mode': 'train'}, f, ensure_ascii=False)
                        f.write("\n")
                    if eval_record is not None:
                        json.dump({**eval_record, "step": states['step'], 'mode': 'eval'}, f, ensure_ascii=False)
                        f.write("\n")
        else:
            result["status"] = "completed"
    except KeyboardInterrupt:
        _logger.note("Experiment interrupted by user.")
        result["status"] = "interrupted"
    except Exception as e:
        _logger.error(f"Experiment failed with exception: {log_exception(e)}")
        result["status"] = "failed"
        if args.debug:
            raise
    finally:
        _logger.info("Exiting training loop, saving final checkpoint and result...")

        # 保存模型
        save_checkpoint(
            Path(args.save_path) / "checkpoint.pth",
            states['step'], args, model, optimizer, scheduler, states,
            float_embedder, equation_embedder, data_embedder,
        )
        _logger.info(f"Final checkpoint saved to {Path(args.save_path) / 'checkpoint.pth'}")

        # 保存结果
        result["duration_seconds"] = (datetime.now() - start_time).total_seconds()
        result['best_eval_step'] = states["best_record"]["step"]
        result["best_eval_loss"] = states["best_record"]["loss"]
        result["total_steps"] = f"{start_step} -> {states['step']}"
        result_path = Path(args.save_path) / "result.jsonl"
        _logger.info(f"Result saved to {result_path}")
        
        # 打印日志
        with open(result_path, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")
        _logger.note(tag2ansi(
            f'\n[gray]{"=" * 50}[reset]\n'
            "[red bold]Property Predictor Training Result[reset]\n"
            + "\n".join([f"[red]{k.replace('_', ' ').title()}[reset]: {v}" for k, v in result.items()])
            + f'\n[gray]{"=" * 50}[reset]'
        ))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train PropertyPredictor on generated data-equation pairs.",
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
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--test_before_train", action="store_true", help="Run evaluation at step 0 before the first training update.")
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument('--confusion_matrix', action='store_true', default=True, help="Whether to compute and log the confusion matrix during evaluation.")
    parser.add_argument("--confusion_matrix_max_size", type=int, default=20, help="Print the full confusion matrix when vocab size is at most this value.")
    parser.add_argument("--confusion_topk", type=int, default=20, help="Print this many top confusion errors when the matrix is too large.")
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
    parser.add_argument("--output_pooling", choices=["attention", "average", "last"], default="attention")
    parser.add_argument("--data_pooling", choices=["attention", "average", "sum"], default="attention")
    parser.add_argument("--encoder_pooling", choices=["attention", "average", "last"], default="attention")
    parser.add_argument("--mlp_hidden_dims", type=int, nargs="*", default=[128], help="MLP hidden dims, e.g. 128 128.")
    parser.add_argument("--shared_decoder_mlp", action="store_true", help="Use one shared MLP for all property outputs.")

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
