import re
import sys
import time
import json
import torch
import shlex
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as D
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from socket import gethostname
from argparse import ArgumentParser
from setproctitle import setproctitle
from src.dataset import ETHDataset, UCYDataset, SDDDataset, GCDataset, WayMoDataset, ORCADataset
from src.model import FunctionalWorldModel as Model
from ...utils.logger import init_logger
from ...utils.seed import seed_all
from ...utils.timer import NamedTimer
from ...utils.plot import get_fig
from ...utils.auto_gpu import AutoGPU
from ...utils.fix_parser import add_negation_flags, add_minus_flags
from ...utils.tag2ansi import tag2ansi
from ...utils.use_npu import USE_NPU, npu_attention_fallback_context
from ...utils.calc_xy_error import calc_xy_error

_logger = logging.getLogger("sr_agent.train")

def main(args):
    ## Load Dataset
    # 加载数据集
    dataset_list = []
    if 'All' in args.datasets:
        args.datasets.remove('All')
        args.datasets += ['ETH', 'UCY', 'GC', 'SDD', 'WayMo']
    if "ETH" in args.datasets:
        dataset_list += ETHDataset.load_data_batch(args, "./data/ETH/")
        args.datasets.remove("ETH")
    if "UCY" in args.datasets:
        dataset_list += UCYDataset.load_data_batch(args, "./data/UCY/data/")
        args.datasets.remove("UCY")
    if "GC" in args.datasets:
        dataset_list += [GCDataset.load_data(args, "./data/GC/Annotation")]
        args.datasets.remove("GC")
    if "SDD" in args.datasets:
        dataset_list += SDDDataset.load_data_batch(args, "./data/SDD/annotations/")
        args.datasets.remove("SDD")
    if 'WayMo' in args.datasets:
        dataset_list += WayMoDataset.load_data_batch(args, "./data/WayMo/Processed/", total=100)
        args.datasets.remove("WayMo")
    if 'ORCA' in args.datasets:
        dataset_list += ORCADataset.load_data_batch(args, "./data/ORCA/")
        args.datasets.remove("ORCA")
    if 'debug' in args.datasets:
        # dataset_list += [UCYDataset.load_data(args, './data/UCY/data/data_university_students/students003.vsp')]
        # dataset_list += [SDDDataset.load_data(args, "./data/SDD/annotations/hyang/video0/annotations.txt")]
        dataset_list += [WayMoDataset.load_data(args, './data/WayMo/Processed/00000_1_2aa43fad083efbf3/data.csv.gz')]
        # dataset_list[0].samples = dataset_list[0].samples[int(len(dataset_list[0].samples) * 0.8):]
        args.datasets.remove('debug')
    if len(args.datasets) > 0:
        raise ValueError(f"Unknown datase: {args.datasets}!")
    # 检查地图
    for dataset in dataset_list:
        map_data = dataset.map_data
        delta_x = map_data.xmax - map_data.xmin
        delta_y = map_data.ymax - map_data.ymin
        w, h = map_data.map.shape
        if not (0.8 < (ratio := (delta_x / w) / (delta_y / h)) < 1.2):
            _logger.warning(
                f"Map aspect ratio of {dataset.name} mismatch: "
                f"data ratio={ratio:.4f} (xrange={delta_x:.4f}, yrange={delta_y:.4f}, "
                f"map shape={map_data.map.shape}), may cause distortion."
            )
            exit(1)
    # 划分训练集和测试集
    if args.test_name is not None:
        # 将名称中包含指定字符串的场景划分到测试集
        train_dataset = []
        test_dataset = []
        for d in dataset_list:
            if any(test_name in d.name for test_name in args.test_name):
                test_dataset.append(d)
            else:
                train_dataset.append(d)
    elif args.test_ratio is not None and args.split_by_scenario:
        # 将特定比例的场景划分到测试集
        random.shuffle(dataset_list)
        test_size = max(1, int(len(dataset_list) * args.test_ratio))
        train_dataset = dataset_list[:-test_size]
        test_dataset = dataset_list[-test_size:]
    elif args.test_ratio is not None and not args.split_by_scenario:
        # 将每个场景的特定比例样本划分到测试集
        train_dataset = []
        test_dataset = []
        for d in dataset_list:
            d1 = d
            d2 = deepcopy(d)
            train_num = int(len(d) * (1-args.test_ratio))
            d1.name = d1.name + f"_{100-args.test_ratio*100:.0f}train"
            d2.name = d2.name + f"_{args.test_ratio*100:.0f}test"
            d1.samples = d1.samples[:train_num]
            d2.samples = d2.samples[train_num:]
            train_dataset.append(d1)
            test_dataset.append(d2)
    else:
        # 训练集和测试集相同
        train_dataset = dataset_list
        test_dataset = dataset_list
    # 创建数据加载器
    train_loaders = []
    test_loaders = []
    for dataset in train_dataset:
        if len(dataset) == 0:
            _logger.warning(f"Dataset {dataset.name} has no training samples!")
            continue
        train_loaders.append(D.DataLoader(
            dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn,
        ))
    for dataset in test_dataset:
        if len(dataset) == 0:
            _logger.warning(f"Dataset {dataset.name} has no testing samples!")
            continue
        test_loaders.append(D.DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.batch_size // args.sample_num,  # 在实际测试时 batch_size 会乘上 sample_num，可能会很大导致 OOM
            num_workers=args.num_workers,
            collate_fn=dataset.collate_fn,
        ))
    _logger.note(
        "Datasets:\n"
        f"Train on {[d.name for d in train_dataset]} datasets ({sum([len(d) for d in train_dataset]):,} samples in total)\n"
        f"Test on {[d.name for d in test_dataset]} datasets ({sum([len(d) for d in test_dataset]):,} samples in total)"
    )

    ## Load Model
    if args.use_new_model:
        model = NewModel(args).to(args.device)
    elif args.use_relative_model:
        model = RelativeModel(args).to(args.device)
    else:
        model = Model(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    if args.sampling_method == "DDIM":
        diffusion = DDIM(args)
    elif args.sampling_method == "DDPM":
        diffusion = DDPM(args, flexibility=0.0)
    else:
        raise ValueError(f"Unknown sampling_method {args.sampling_method}!")
    _logger.note(
        "Model Parameters:\n"
        f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n"
        f"Total: {sum(p.numel() for p in model.parameters()):,}"
    )

    ## Reload Checkpoint
    if args.reload_checkpoint is not None:
        # 如果指定了 checkpoint 路径，则从该路径加载
        checkpoint_path = Path(args.reload_checkpoint)
    elif (Path(args.save_path) / "checkpoint.pth").exists():
        # 如果当前保存路径下存在 checkpoint，则从该路径加载
        checkpoint_path = Path(args.save_path) / "checkpoint.pth"
    else:
        checkpoint_path = None
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        start_epoch = checkpoint["epoch"] + 1
        if 'args' in checkpoint:
            saved_args = checkpoint['args']
            for key in sorted(set(saved_args.keys()) | set(vars(args).keys())):
                val1 = saved_args.get(key, None)
                val2 = getattr(args, key, None)
                if val1 != val2:
                    _logger.warning(
                        f"Argument '{key}' differs from the saved checkpoint: "
                        f"saved_args={val1} vs. current_args={val2}"
                    )
        model.load_state_dict(checkpoint["model"])
        _logger.note(tag2ansi(f"Checkpoint loaded from [underline green]{checkpoint_path}[reset], resume from epoch [underline green]{start_epoch}[reset]."))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            _logger.warning("Optimizer state not found in checkpoint, optimizer re-initialized.")
    else:
        start_epoch = 0

    ## Train
    timer = NamedTimer()
    for epoch in range(start_epoch, args.epochs+1):
        # 训练一个 epoch
        if epoch > 0:
            torch.set_grad_enabled(True)
            model.train()
            train_records = train_once(args, train_loaders, model, optimizer, criterion, diffusion, epoch)
            timer.add('train')
        else:
            train_records = None

        # 测试一个 epoch
        if (epoch > 0 and not epoch % args.test_per_epoch) or (epoch == 0 and args.test_before_train):
            torch.set_grad_enabled(False)
            model.eval()
            with npu_attention_fallback_context(model, enable=USE_NPU):
                test_records = test_once(args, test_loaders, model, criterion, diffusion, epoch)
            timer.add('test')
        else:
            test_records = None

        # 保存日志
        with open(f"{args.save_path}/records.jsonl", "a") as f:
            if train_records is not None:
                f.write(json.dumps(train_records) + "\n")
            if test_records is not None:
                f.write(json.dumps(test_records) + "\n")

        # 保存加载点
        if '_last_checkpoint_time' not in locals() or (datetime.now() - _last_checkpoint_time).seconds  > 300:
            # 只在间隔超过 5 min 时保存
            _last_checkpoint_time = datetime.now()
            save_path = f"{args.save_path}/checkpoint.pth"
            torch.save({
                "epoch": epoch,
                "args": vars(args),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, save_path)
            _logger.info(tag2ansi(f"Checkpoint saved to [underline green]{save_path}[reset]."))
            timer.add('save_checkpoint')
        
        # 定期保存
        if set(str(epoch)[1:]) == {'0'}:
            # 只在 epoch=10,20,...,100,...,1000,... 时保存
            save_path = Path(args.save_path) / "checkpoints" / f"epoch{epoch}.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "args": vars(args),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, save_path)
            _logger.note(tag2ansi(f"Model saved to [underline green]{save_path}[reset]"))
            timer.add('save_periodly')

        # 保存最佳模型
        if test_records is not None:
            if (
                'best_records' not in locals() or 
                np.mean(test_records['ade']) < np.mean(best_records['ade'])
            ):
                patience = args.patience
                best_records = test_records
                save_path = f"{args.save_path}/best.pth"
                torch.save({
                    "epoch": epoch,
                    "args": vars(args),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, save_path)
                _logger.note(tag2ansi(f"Best model saved to [underline green]{save_path}[reset]"))
            else:
                patience -= 1
                _logger.info(tag2ansi(
                    f"Patience left: [brightred]{patience}/{args.patience}[reset] ("
                    f"[bold underline orange]best Accuracy={best_records['accuracy']:.2%}[reset] "
                    f"at epoch [#66CCFF]{best_records['epoch']}[reset]. "
                    f"[#66CCFF]ADE={np.mean(best_records['ade']):.4f}, "
                    f"[#66CCFF]FDE={np.mean(best_records['fde']):.4f}, "
                    f"[#66CCFF]AvgLen={np.mean(best_records['trajlen']):.4f}, "
                    f"[#66CCFF]Loss={np.mean(best_records['loss']):.4f}, "
                    f"[#66CCFF]PedNum={np.mean(best_records['ped_num']):.1f}, "
                    f"[#66CCFF]VehNum={np.mean(best_records['veh_num']):.1f})"
                ))
            timer.add('save_best')

        # 打印用时
        allocated = torch.cuda.memory_allocated(args.device) / 1024 / 1024 / 1024
        reserved = torch.cuda.memory_reserved(args.device) / 1024 / 1024 / 1024
        peak = torch.cuda.max_memory_allocated(args.device) / 1024 / 1024 / 1024
        _logger.info(tag2ansi(
            f"[pink][Epoch {epoch}/{args.epochs}] finished. "
            f"Time Usage={timer}, "
            f"CUDA ({args.device}) usage: allocated={allocated:.1f}GiB, peak={peak:.1f}GiB, reserved={reserved:.1f}GiB"
            # f"adjust reserved memory from {reserved_raw/1024:.1f}GiB to {reserved_new/1024:.1f}GiB"
            "[reset]"
        ))

        # 释放额外的显存
        if train_records is not None:
            peak = torch.cuda.max_memory_allocated(args.device) / 1024 / 1024
            reserved_raw = torch.cuda.memory_reserved(args.device) / 1024 / 1024
            torch.cuda.empty_cache() # 释放 reserved 但是未被 allocated 的 block
            reserved_new = torch.cuda.memory_reserved(args.device) / 1024 / 1024
            if reserved_new < peak: # 释放了过多的显存，之后可能会 OOM
                allocated = torch.cuda.memory_allocated(args.device) / 1024 / 1024
                if (keep_MB := int(np.ceil(peak - allocated))) > 0: # 把需要的显存再占回来
                    tmp = AutoGPU.allocate_gpu(device=args.device, memory_MB=keep_MB, block_MB=None)
                    del tmp
                reserved_new = torch.cuda.memory_reserved(args.device) / 1024 / 1024
            _logger.info(tag2ansi(
                f"[brown]Adjust reserved memory from {reserved_raw/1024:.1f}GiB to {reserved_new/1024:.1f}GiB. [reset]"
            ))
        
        # 提前终止
        if 'patience' in locals() and patience <= 0:
            _logger.warning(tag2ansi(f"Early stopping at epoch [lightred]{epoch}/{args.epochs}[reset], "))
            break

    ## Log Best Result
    _logger.note(tag2ansi(
        f"[bold underline orange]best Accuracy={best_records['accuracy']:.2%}[reset] "
        f"at [#66CCFF]epoch {best_records['epoch']}[reset]. "
        f"[#66CCFF]ADE={np.mean(best_records['ade']):.4f}, "
        f"[#66CCFF]FDE={np.mean(best_records['fde']):.4f}, "
        f"[#66CCFF]AvgLen={np.mean(best_records['trajlen']):.4f}, "
        f"[#66CCFF]Loss={np.mean(best_records['loss']):.4f}, "
        f"[#66CCFF]PedNum={np.mean(best_records['ped_num']):.1f}, "
        f"[#66CCFF]VehNum={np.mean(best_records['veh_num']):.1f}"
    ))
    if len(set(best_records['dataset_class'])) > 1:
        for klass in sorted(list(set(best_records['dataset_class']))):
            idxs = [i for i, k in enumerate(best_records['dataset_class']) if k == klass]
            ade = np.array([best_records['ade'][i] for i in idxs])
            fde = np.array([best_records['fde'][i] for i in idxs])
            trajlen = np.array([best_records['trajlen'][i] for i in idxs])
            ped_num = np.array([best_records['ped_num'][i] for i in idxs])
            veh_num = np.array([best_records['veh_num'][i] for i in idxs])
            rollout_time = np.array([best_records['rollout_time'][i] for i in idxs])
            w = np.array([best_records['sample_nums'][i] for i in idxs], dtype=float)
            w /= w.sum()
            acc = 1 - np.sum(w * ade) / np.sum(w * trajlen)
            _logger.info(tag2ansi(
                f"[#66CCFF][Epoch {best_records['epoch']}/{args.epochs}] Overall on {klass} datasets: "
                f"[bold underline orange]Accuracy={acc:.2%}[reset], "
                f"[#66CCFF]ADE={np.sum(w * ade):.4f}, "
                f"[#66CCFF]FDE={np.sum(w * fde):.4f}, "
                f"[#66CCFF]AvgLen={np.sum(w * trajlen):.4f}, "
                f"[#66CCFF]PedNum={np.sum(w * ped_num):.4f}, "
                f"[#66CCFF]VehNum={np.sum(w * veh_num):.4f}, "
                f"[#66CCFF]RolloutTime={np.mean(rollout_time)*1000:.2f}ms "
                f"([bold underline orange]FPS={1/np.mean(rollout_time):.2f} Hz[reset])"
            ))
    _logger.note(f"Training finished. Re-run: {args.command}")


def train_once(args, train_loaders, model, optimizer, criterion, diffusion, epoch):
    train_timer = NamedTimer(unit='it', mode='pace')
    records_list = []
    for loader in train_loaders:
        map_data = loader.dataset.map_data
        map = torch.from_numpy(map_data.map).to(args.device).float()
        records = dict(loss=[], rollout_loss=[])
        for batch in tqdm(loader, total=len(loader), disable=False, leave=False, dynamic_ncols=True):
            optimizer.zero_grad()
            pos = batch['pos'].to(args.device)  # (batch_size, #pedestrian, 2)
            vel = batch['vel'].to(args.device)  # (batch_size, #pedestrian, 2)
            hst = batch['hst'].to(args.device)  # (batch_size, #pedestrian, hist_step, 2)
            des = batch['des'].to(args.device)  # (batch_size, #pedestrian, 2)
            spd = batch['spd'].to(args.device)  # (batch_size, #pedestrian)
            veh = batch['veh'].to(args.device)  # (batch_size, #vehicle, hist_step + 1, 2)
            future_acc = batch['future_acc'].to(args.device)  # (batch_size, #pedestrian, pred_step*roll_step, 2)
            future_pos = batch['future_pos'].to(args.device)  # (batch_size, #pedestrian, pred_step*roll_step, 2)
            future_veh = batch['future_veh'].to(args.device)  # (batch_size, #vehicle, pred_step*roll_step, 2)
            ped_length = batch['ped_length'].to(args.device)  # (batch_size,)
            veh_length = batch['veh_length'].to(args.device)  # (batch_size,)
            train_timer.add('prepare data')

            # Rollout
            pos_now = pos
            vel_now = vel
            hst_now = hst
            des_now = des
            spd_now = spd
            veh_now = veh
            rollout_loss = []
            for step in range(args.multi_frame_rollout):
                # DDPM forward
                # pos_true = future_pos[:, :, args.pred_step*step:args.pred_step*(step+1), :] # (B, #pedestrian, pred_step, 2)
                # vel_true = pos_true.diff(dim=-2, prepend=pos_now.unsqueeze(-2)) * args.fps  # (B, #pedestrian, roll_step*pred_step, 2)
                # acc_true = vel_true.diff(dim=-2, prepend=vel_now.unsqueeze(-2)) * args.fps  # (B, #pedestrian, roll_step*pred_step, 2)
                acc_true = future_acc[:, :, args.pred_step*step:args.pred_step*(step+1), :] # (B, #pedestrian, pred_step, 2)
                noisy_acc, noise_true, denoise_t = diffusion.add_noise(acc_true * args.scale_accelerate)
                train_timer.add('add noise')

                if args.p_drop_map and random.random() < args.p_drop_map:
                    map = torch.full_like(map, torch.nan, device=map.device)
                if args.p_drop_destination and random.random() < args.p_drop_destination:
                    des_now = torch.full_like(des_now, torch.nan, device=des_now.device)
                if args.p_drop_speed and random.random() < args.p_drop_speed:
                    spd_now = torch.full_like(spd_now, torch.nan, device=spd_now.device)

                # DDPM backward
                model.set_map_embedding(
                    map=map,
                    xmin=map_data.xmin,
                    xmax=map_data.xmax,
                    ymin=map_data.ymin,
                    ymax=map_data.ymax,
                )
                model.set_veh_embedding(veh=veh_now)
                model.set_ped_embedding(pos=pos_now, vel=vel_now, hst=hst_now, des=des_now, spd=spd_now)
                model.set_sur_info()
                output = model(
                    noisy_acc=noisy_acc, 
                    denoise_t=denoise_t,
                    ped_length=ped_length, 
                    veh_length=veh_length,
                )  # (B, #pedestrian, pred_step, 2)
                train_timer.add('forward')

                # Compute Loss
                if args.predict_noise:
                    noise_pred = output
                    acc_pred = diffusion.noise_to_x0(xt=noisy_acc, denoise_t=denoise_t, noise=noise_pred) / args.scale_accelerate
                else:
                    acc_pred = output / args.scale_accelerate
                
                if args.loss_type == 'accelerate':
                    loss = criterion(acc_pred, acc_true)
                elif args.loss_type == 'position':
                    # acc_true 是从 pos_true 算出来的，因此不用再返回去计算 pos_true 了
                    vel_true = vel_now.unsqueeze(-2) + acc_true.cumsum(dim=-2) / args.fps
                    pos_true = pos_now.unsqueeze(-2) + vel_true.cumsum(dim=-2) / args.fps
                    vel_pred = vel_now.unsqueeze(-2) + acc_pred.cumsum(dim=-2) / args.fps
                    pos_pred = pos_now.unsqueeze(-2) + vel_pred.cumsum(dim=-2) / args.fps
                    loss = criterion(pos_pred, pos_true)
                elif args.loss_type == 'noise':
                    if not args.predict_noise:
                        raise ValueError("When using noise prediction loss, the model must predict noise!")
                    loss = criterion(noise_pred, noise_true)
                else:
                    raise ValueError(f"Unknown loss type {args.loss_type}!")
                rollout_loss.append(loss.detach().cpu().tolist())
                train_timer.add('compute loss')
                (args.rollout_lambda ** (args.multi_frame_rollout - step) * loss).backward()

                acc_new = acc_pred.detach()  # (B, #pedestrian, pred_step, 2)
                vel_new = vel_now.unsqueeze(-2) + acc_new.cumsum(dim=-2) / args.fps  # (B, #pedestrian, pred_step, 2)
                pos_new = pos_now.unsqueeze(-2) + vel_new.cumsum(dim=-2) / args.fps  # (B, #pedestrian, pred_step, 2)
                veh_new = future_veh[:, :, step*args.pred_step:(step+1)*args.pred_step, :]  # (B, #vehicle, pred_step, 2)

                hst_now = torch.cat([hst_now, pos_now.unsqueeze(-2), pos_new], dim=-2)[:, :, -args.hist_step-1:-1, :] # (B, #pedestrian, hist_step, 2)
                veh_now = torch.cat([veh_now, veh_new], dim=-2)[:, :, -args.hist_step-1:, :]  # (B, #vehicle, hist_step + 1, 2)
                pos_now = pos_new[:, :, -1, :] # (B, #pedestrian, 2)
                vel_now = vel_new[:, :, -1, :] # (B, #pedestrian, 2)
                train_timer.add('rollout')

            ## Backpropagate
            optimizer.step()
            records['loss'].extend([loss.item()] * acc_true.shape[0])
            records['rollout_loss'].extend([rollout_loss] * acc_true.shape[0])
            train_timer.add('backpropagate')
        records_list.append(records)
        _logger.debug(
            f"[Epoch {epoch}/{args.epochs}] Train on {loader.dataset.name}: "
            f"Loss={np.mean(records['loss']):.4f}"
        )
    all_records = {
        'epoch': epoch,
        'dataset_names': [loader.dataset.name for loader in train_loaders],
        'sample_nums': [len(loader.dataset) for loader in train_loaders],
    }
    for records in records_list:
        for k, v in records.items():
            if k not in all_records:
                all_records[k] = []
            if isinstance(v[0], (int, float)):
                mean_v = np.mean(v)
            else: 
                mean_v = np.mean(v, axis=0).tolist()
            all_records[k].append(mean_v)
    _logger.info(tag2ansi(
        f"[#66CCFF][Epoch {epoch}/{args.epochs}] "
        f"[#66CCFF]Loss={np.mean(all_records['loss']):.4f} "
        f"[#66CCFF]Rollout Loss={np.mean(all_records['rollout_loss'], axis=0).round(4).tolist()} "
        f"[#66CCFF]Time={train_timer}"
    ))
    return all_records


def test_once(args, test_loaders, model, criterion, diffusion, epoch):
    test_timer = NamedTimer(unit='it', mode='pace')
    records_list = []
    for loader in test_loaders:
        map_data = loader.dataset.map_data
        map = torch.from_numpy(map_data.map).to(args.device).float()
        test_timer.add('prepare data')
        model.set_map_embedding(
            map=map,
            xmin=map_data.xmin,
            xmax=map_data.xmax,
            ymin=map_data.ymin,
            ymax=map_data.ymax,
        )
        test_timer.add('embed map')
        records = dict(loss=[], ade=[], fde=[], trajlen=[], ped_num=[], veh_num=[], rollout_time=[])
        for batch_idx, batch in enumerate(tqdm(loader, disable=False, leave=False, dynamic_ncols=True)):
            pos = batch['pos'].to(args.device)  # (batch_size, #pedestrian, 2)
            vel = batch['vel'].to(args.device)  # (batch_size, #pedestrian, 2)
            hst = batch['hst'].to(args.device)  # (batch_size, #pedestrian, hist_step, 2)
            des = batch['des'].to(args.device)  # (batch_size, #pedestrian, 2)
            spd = batch['spd'].to(args.device)  # (batch_size, #pedestrian, 1)
            veh = batch['veh'].to(args.device)  # (batch_size, #vehicle, hist_step + 1, 2)
            future_acc = batch['future_acc'].to(args.device)  # (batch_size, #pedestrian, pred_step*roll_step, 2)
            future_pos = batch['future_pos'].to(args.device)  # (batch_size, #pedestrian, pred_step*roll_step, 2)
            future_veh = batch['future_veh'].to(args.device)  # (batch_size, #vehicle, pred_step*roll_step, 2)
            ped_length = batch['ped_length'].to(args.device)  # (batch_size,)
            veh_length = batch['veh_length'].to(args.device)  # (batch_size,)

            S = args.sample_num  # 采样次数
            N = args.denoise_step  # 采样步数
            assert args.T % N == 0, f"试图使用 {N} 步采样，然而训练步数 {args.T} mod {N} 不等于 0!"
            assert 1 <= args.step_offset <= args.T // N, f"step_offset 应该取值于 {{1, ..., {args.T // N}}}!"
            pos_now = pos.repeat(S, 1, 1)  # (S*B, #pedestrian, 2)
            vel_now = vel.repeat(S, 1, 1)  # (S*B, #pedestrian, 2)
            hst_now = hst.repeat(S, 1, 1, 1)  # (S*B, #pedestrian, hist_step, 2)
            des_now = des.repeat(S, 1, 1)  # (S*B, #pedestrian, 2)
            spd_now = spd.repeat(S, 1, 1)  # (S*B, #pedestrian, 1)
            veh_now = veh.repeat(S, 1, 1, 1)  # (S*B, #vehicle, hist_step + 1, 2)
            ped_length_repeat = ped_length.repeat(S)  # (S*B,)
            veh_length_repeat = veh_length.repeat(S)  # (S*B,)
            test_timer.add('prepare data', n=0)

            for_plot = []
            acc_pred = []
            start_time = time.time()
            for step in range(args.roll_step):
                model.set_veh_embedding(veh=veh_now)
                model.set_ped_embedding(pos=pos_now, vel=vel_now, hst=hst_now, des=des_now, spd=spd_now)
                model.set_sur_info()
                test_timer.add('embed data')

                shape = list(future_acc.shape)
                shape[0] *= S
                shape[2] = args.pred_step
                xt = torch.randn(shape, device=args.device)  # 从噪声开始
                for_plot.append([diffusion.noise_to_x0(xt=xt, denoise_t=args.T, noise=0) / args.scale_accelerate])
                stride = args.T // N
                steps = reversed(range(args.step_offset, args.T+1, stride))
                for t in tqdm(steps, disable=True, leave=False, dynamic_ncols=True):
                    noisy_acc = xt
                    denoise_t = torch.full((xt.shape[0],), t, device=args.device, dtype=torch.long)
                    output = model(
                        noisy_acc=noisy_acc, 
                        denoise_t=denoise_t,
                        ped_length=ped_length_repeat, 
                        veh_length=veh_length_repeat,
                    )  # (S*B, #pedestrian, pred_step, 2)
                    if args.predict_noise:
                        for_plot[-1].append(diffusion.noise_to_x0(xt=xt, denoise_t=t, noise=output) / args.scale_accelerate)
                        xt = diffusion.denoise(xt, t, noise=output, stride=min(stride, t))
                    else:
                        for_plot[-1].append(output / args.scale_accelerate)
                        xt = diffusion.denoise(xt, t, x0=output, stride=min(stride, t))
                acc_new = xt / args.scale_accelerate  # (S*B, #pedestrian, pred_step, 2)
                acc_pred.append(acc_new)
                test_timer.add('denoise')

                vel_new = vel_now.unsqueeze(-2) + acc_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
                pos_new = pos_now.unsqueeze(-2) + vel_new.cumsum(dim=-2) / args.fps  # (S*B, #pedestrian, pred_step, 2)
                veh_new = future_veh[:, :, step*args.pred_step:(step+1)*args.pred_step, :].repeat(S, 1, 1, 1)  # (S*B, #vehicle, pred_step, 2)

                hst_now = torch.cat([hst_now, pos_now.unsqueeze(-2), pos_new], dim=-2)[:, :, -args.hist_step-1:-1, :] # (S*B, #pedestrian, hist_step, 2)
                veh_now = torch.cat([veh_now, veh_new], dim=-2)[:, :, -args.hist_step-1:, :]  # (S*B, #vehicle, hist_step + 1, 2)
                pos_now = pos_new[:, :, -1, :] # (S*B, #pedestrian, 2)
                vel_now = vel_new[:, :, -1, :] # (S*B, #pedestrian, 2)
                test_timer.add('rollout')
            rollout_time = (time.time() - start_time) / args.roll_step

            acc_pred = torch.concat(acc_pred, dim=-2)  # (S*B, #pedestrian, roll_step*pred_step, 2)
            batch_size, ped_num, _, _ = future_acc.shape
            acc_pred = acc_pred.view(S, batch_size, ped_num, args.roll_step*args.pred_step, 2)  # (S, B, #pedestrian, roll_step*pred_step, 2)
            # 获取有效的行人掩模
            mask = torch.arange(ped_num, device=args.device).expand(batch_size, ped_num) < ped_length.unsqueeze(-1)  # (B, #pedestrian)
            # 计算 pos_true 和 vel_true
            acc_true = future_acc # (B, #pedestrian, roll_step*pred_step, 2)
            vel_true = vel.unsqueeze(-2) + acc_true.cumsum(dim=-2) / args.fps # (B, #pedestrian, roll_step*pred_step, 2)
            pos_true = pos.unsqueeze(-2) + vel_true.cumsum(dim=-2) / args.fps # (B, #pedestrian, roll_step*pred_step, 2)
            max_err = np.nanmax((pos_true - future_pos).abs().cpu().numpy(), axis=(-2, -1))
            _logger.debug(
                f"pos_true 和 future 最大差距 > 1: {(max_err > 1).mean():.2%}, "
                f"pos_true 和 future 最大差距 > 1e-6: {(max_err > 1e-6).mean():.2%}"
            )
            # pos_true = future_pos # (B, #pedestrian, roll_step*pred_step, 2)
            # 计算 loss
            loss = criterion(acc_pred, acc_true.expand(acc_pred.shape)) # float
            records['loss'].extend([loss.item()] * future_acc.shape[0]) # List[float]
            # 计算 distance error
            vel_pred = vel.unsqueeze(-2) + acc_pred.cumsum(dim=-2) / args.fps  # (S, B, #pedestrian, pred_step, 2)
            pos_pred = pos.unsqueeze(-2) + vel_pred.cumsum(dim=-2) / args.fps  # (S, B, #pedestrian, pred_step, 2)
            dis_err = (pos_pred - pos_true).norm(dim=-1) # (S, B, #pedestrian, pred_step)
            test_timer.add('evaluate')
            # 可视化
            if batch_idx == 0:
                pid = 0
                save_path = f"{args.save_path}/visualize/epoch{epoch}_{loader.dataset.name}_idx{batch_idx}_pid{pid}.png"
                visualize(args, pos, vel, hst, for_plot, mask, pos_true, pos_pred, save_path, pid)
                test_timer.add('visualize')
            # 移除 padding 的行人
            dis_err = dis_err[:, mask, :] # (S, valid{B*#pedestrian}, pred_step)
            # 选择 ade 最佳的 sample
            sample_idx = dis_err.mean(dim=-1).argmin(dim=0)  # (valid{B*#pedestrian},)
            valid_idx = torch.arange(dis_err.shape[1], device=args.device)  # (valid{B*#pedestrian},)
            dis_err = dis_err[sample_idx, valid_idx, :]  # (valid{B*#pedestrian}, pred_step)
            # 计算 ade, fde
            ade = dis_err.mean(dim=-1) # (valid{B*#pedestrian})
            fde = dis_err[..., -1] # (valid{B*#pedestrian})
            records['ade'].extend(ade.cpu().tolist()) # List[float]
            records['fde'].extend(fde.cpu().tolist()) # List[float]
            # 计算切向误差和法向误差
            traj_diff = (pos_pred - pos_true)[:, mask, :, :][sample_idx, valid_idx, :, :] # (valid{B*#pedestrian}, pred_step, 2)
            ped_pos = pos[mask, :] # (valid{B*#pedestrian}, 2)
            batch_indices = torch.nonzero(mask)[:, 0]  # 有效行人所属的 batch index (valid{B*#pedestrian},)
            num_valid_ped = batch_indices.shape[0]
            if veh.shape[1] == 0: # 场景中完全没有车辆数据
                veh_pos = torch.full((num_valid_ped, 2), float('nan'), device=pos.device)
                veh_vel = torch.full((num_valid_ped, 2), float('nan'), device=pos.device)
            else: # 找到每个场景中距离各个行人最近的车辆
                all_veh_pos = veh[..., -1, :] # (batch_size, #vehicle, 2)
                all_veh_vel = (veh[..., -1, :] - veh[..., -2, :]) * args.fps  # (batch_size, #vehicle, 2)
                # 取出每个有效行人对应场景的车辆数据
                batch_veh_pos = all_veh_pos[batch_indices] # (valid{B*#pedestrian}, #vehicle, 2)
                batch_veh_vel = all_veh_vel[batch_indices] # (valid{B*#pedestrian}, #vehicle, 2)
                # 计算行人到同场景所有车辆的距离 (将无效车辆的距离设为无穷大)
                dist = (ped_pos.unsqueeze(1) - batch_veh_pos).norm(dim=-1).nan_to_num_(nan=float('inf')) # (valid{B*#pedestrian}, #vehicle)
                # 找到最近车辆的索引
                min_dist, nearest_idx = torch.min(dist, dim=1) # (valid{B*#pedestrian},)
                has_vehicle = min_dist != float('inf')
                # Gather 最近车辆的位置和速度
                gather_idx = nearest_idx.view(-1, 1, 1).expand(-1, 1, 2)
                veh_pos = torch.gather(batch_veh_pos, 1, gather_idx).squeeze(1) # (valid{B*#pedestrian}, 2)
                veh_vel = torch.gather(batch_veh_vel, 1, gather_idx).squeeze(1) # (valid{B*#pedestrian}, 2)
                # 如果该行人所在的场景没有任何车辆，设为 NaN
                veh_pos[~has_vehicle] = float('nan')
                veh_vel[~has_vehicle] = float('nan')
            records['norm_err'], records['tan_err'] = calc_xy_error(traj_diff, ped_pos, veh_pos, veh_vel)
            # 计算轨迹长度
            trajlen = pos_true.diff(dim=-2).norm(dim=-1).sum(dim=-1)[mask] # (valid{B*#pedestrian})
            records['trajlen'].extend(trajlen.cpu().tolist()) # List[float]
            # 统计行人和车辆数量
            records['ped_num'].extend(ped_length.cpu().tolist()) # List[int]
            records['veh_num'].extend(veh_length.cpu().tolist()) # List[int]
            # 统计 Rollout 用时
            records['rollout_time'].append(rollout_time)  # List[float]
            test_timer.add('evaluate', n=0)
        records_list.append(records)
        _logger.info(tag2ansi(
            f"[#66CCFF][Epoch {epoch}/{args.epochs}] Eval on {loader.dataset.name}: "
            f"[bold underline orange]Accuracy={1 - np.mean(records['ade']) / np.mean(records['trajlen']):.2%}[reset], "
            f"[#66CCFF]Loss={np.mean(records['loss']):.4f}, "
            f"[#66CCFF]ADE={np.mean(records['ade']):.4f}, "
            f"[#66CCFF]FDE={np.mean(records['fde']):.4f}, "
            f"[#66CCFF]X_ERROR (normal)={np.nanmean(records['norm_err']):.4f}, "
            f"[#66CCFF]Y_ERROR (tangential)={np.nanmean(records['tan_err']):.4f}, "
            f"[#66CCFF]AvgLen={np.mean(records['trajlen']):.4f}, "
            f"[#66CCFF]PedNum={np.mean(records['ped_num']):.1f}, "
            f"[#66CCFF]VehNum={np.mean(records['veh_num']):.1f}, "
            f"[#66CCFF]RolloutTime={np.mean(records['rollout_time'])*1000:.2f}ms "
            f"([bold underline orange]FPS={1/np.mean(records['rollout_time']):.2f} Hz[reset])"
        ))
    all_records = {
        'epoch': epoch,
        'dataset_class': [type(loader.dataset).__name__.removesuffix('Dataset') for loader in test_loaders],
        'dataset_names': [loader.dataset.name for loader in test_loaders],
        'sample_nums': [len(loader.dataset) for loader in test_loaders],
    }
    for records in records_list:
        for k, v in records.items():
            if k not in all_records:
                all_records[k] = []
            all_records[k].append(np.mean(v))
    w = np.array(all_records['sample_nums'], dtype=float)
    w /= w.sum()
    all_records['accuracy'] = 1 - np.sum(w * all_records['ade']) / np.sum(w * all_records['trajlen'])
    all_records['unweighted_accuracy'] = 1 - np.mean(all_records['ade']) / np.mean(all_records['trajlen'])
    _logger.note(tag2ansi(
        f"[#66CCFF][Epoch {epoch}/{args.epochs}] Overall: "
        f"[bold underline orange]Accuracy={all_records['accuracy']:.2%}[reset] (unweighted={all_records['unweighted_accuracy']:.2%}), "
        f"[#66CCFF]Loss={np.sum(w * all_records['loss']):.4f}, "
        f"[#66CCFF]ADE={np.sum(w * all_records['ade']):.4f}, "
        f"[#66CCFF]FDE={np.sum(w * all_records['fde']):.4f}, "
        f"[#66CCFF]X_ERROR (normal)={np.nansum(w * all_records['norm_err']) / np.sum(w * np.isfinite(all_records['norm_err'])):.4f}, "
        f"[#66CCFF]Y_ERROR (tangential)={np.nansum(w * all_records['tan_err']) / np.sum(w * np.isfinite(all_records['tan_err'])):.4f}, "
        f"[#66CCFF]AvgLen={np.sum(w * all_records['trajlen']):.4f}, "
        f"[#66CCFF]PedNum={np.sum(w * all_records['ped_num']):.4f}, "
        f"[#66CCFF]VehNum={np.sum(w * all_records['veh_num']):.4f}, "
        f"[#66CCFF]RolloutTime={np.mean(all_records['rollout_time'])*1000:.2}ms "
        f"([bold underline orange]FPS={1/np.mean(all_records['rollout_time']):.2f} Hz[reset]), "
        f"[#66CCFF]Time={test_timer}"
    ))
    if len(set(all_records['dataset_class'])) > 1:
        for klass in sorted(list(set(all_records['dataset_class']))):
            idxs = [i for i, k in enumerate(all_records['dataset_class']) if k == klass]
            ade = np.array([all_records['ade'][i] for i in idxs])
            fde = np.array([all_records['fde'][i] for i in idxs])
            norm_err = np.array([all_records['norm_err'][i] for i in idxs])
            tan_err = np.array([all_records['tan_err'][i] for i in idxs])
            trajlen = np.array([all_records['trajlen'][i] for i in idxs])
            ped_num = np.array([all_records['ped_num'][i] for i in idxs])
            veh_num = np.array([all_records['veh_num'][i] for i in idxs])
            rollout_time = np.array([all_records['rollout_time'][i] for i in idxs])
            w = np.array([all_records['sample_nums'][i] for i in idxs], dtype=float)
            w /= w.sum()
            acc = 1 - np.sum(w * ade) / np.sum(w * trajlen)
            _logger.note(tag2ansi(
                f"[#66CCFF][Epoch {epoch}/{args.epochs}] Overall on {klass} datasets: "
                f"[bold underline orange]Accuracy={acc:.2%}[reset], "
                f"[#66CCFF]ADE={np.sum(w * ade):.4f}, "
                f"[#66CCFF]FDE={np.sum(w * fde):.4f}, "
                f"[#66CCFF]X_ERROR (normal)={np.nansum(w * norm_err) / np.sum(w * np.isfinite(norm_err)):.4f}, "
                f"[#66CCFF]Y_ERROR (tangential)={np.nansum(w * tan_err) / np.sum(w * np.isfinite(tan_err)):.4f}, "
                f"[#66CCFF]AvgLen={np.sum(w * trajlen):.4f}, "
                f"[#66CCFF]PedNum={np.sum(w * ped_num):.4f}, "
                f"[#66CCFF]VehNum={np.sum(w * veh_num):.4f}, "
                f"[#66CCFF]RolloutTime={np.mean(rollout_time)*1000:.2f}ms "
                f"([bold underline orange]FPS={1/np.mean(rollout_time):.2f} Hz[reset])"
            ))
    return all_records


if __name__ == "__main__":
    parser = ArgumentParser()
    # 基础配置
    parser.add_argument("--name", type=str, default="train", help="实验任务名称，用于生成实验ID")
    parser.add_argument("--exp_name", type=str, default=None, help="手动指定实验名称（若指定则覆盖自动生成的名称）")
    parser.add_argument("--device", type=str, default="auto", help="计算设备，可选 'cpu', 'cuda:0' 或 'auto'（自动选择显存充足的 GPU）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，固定以复现实验结果")
    parser.add_argument("--save_dir", type=str, default="./logs/train", help="日志和模型权重的保存根目录")
    parser.add_argument("--debug", action="store_true", help="是否开启调试模式（输出更多日志，不保存部分文件）")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader 的工作线程数（0 表示主线程）")
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=128, help="训练批次大小")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率 (Learning Rate)")
    parser.add_argument("--epochs", type=int, default=10000, help="最大训练轮数")
    parser.add_argument('--patience', type=int, default=20, help="Early Stopping 的耐心值（多少个 epoch 验证集指标不提升则停止）")
    parser.add_argument('--loss_type', type=str, default='noise', choices=['position', 'accelerate', 'noise'], help="损失函数计算的目标类型")
    parser.add_argument('--reload_checkpoint', type=str, default=None, help="断点续训的 checkpoint 路径（.pth 文件）")
    parser.add_argument('--required_memory_MB', type=int, default=6000, help="自动选择 GPU 时要求的最小剩余显存 (MB)")

    parser = add_minus_flags(parser) ## --key_name -> --key-name
    parser = add_negation_flags(parser) ## --action-as-true -> --no-action-as-true
    args, unknown = parser.parse_known_args()

    ## Build Save Path
    if args.exp_name is None:
        now = datetime.now()
        date = now.strftime("%Y%m%d")
        curr = now.strftime("%H%M%S")
        host = gethostname()
        exp_name = f'{date}_{args.name}_{curr}_{host}'
        exp_name = re.compile(r'[ <>:"/\\|?*\x00-\x1f]').sub('_', exp_name.strip())
        exp_name = exp_name or 'unnamed'
        exp_name = exp_name[:255] # Max filename length on most filesystems
        args.exp_name = exp_name
    save_path = Path(args.save_dir) / args.exp_name
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        _logger.warning(f"Save path {save_path} already exists.")
    args.save_path = str(save_path)

    ## Init Logger
    init_logger(
        "src",
        exp_name=args.exp_name,
        log_file=save_path / "info.log",
        info_level="debug" if args.debug else "info",
    )

    ## Warm Unknown Args
    if unknown:
        _logger.warning(f"Unknown args: {unknown}")

    ## Set Seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    seed_all(args.seed)
    ## Set Command
    args.command = ' '.join(map(shlex.quote, [sys.executable, *sys.argv]))
    ## Select GPU
    if args.device == "auto":
        args.device = AutoGPU().choice_gpu(memory_MB=args.required_memory_MB, interval=15) if not USE_NPU else 'npu'

    ## Save Args
    args_path = save_path / "args.json"
    if args_path.exists():
        i = 1
        while args_path.with_suffix(f".json.{i}").exists(): i += 1
        args_path.rename(args_path.with_suffix(f".json.{i}"))
        _logger.warning(f"args.json already exists, backup to args.json.{i}")
    _logger.note(f"Args: {args}")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    ## Start Training
    setproctitle(f"{args.exp_name}@ZihanYu")
    main(args)
