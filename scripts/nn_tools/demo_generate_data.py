# python ./scripts/nn_tools/demo_generate_data.py
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "experimental"))

import os
import time
import logging
import torch.utils.data as D
from argparse import ArgumentParser
from sr_agent.utils import setup_logging
from nn_tools.models import EquationEmbedder
from nn_tools.datasets.generate_eq import BaseEqGenerator
from nn_tools.datasets.generate_data import BaseDataGenerator
from nn_tools.datasets.data_eq_dataset import DataEqDataset, InfiniteSampler

setup_logging(info_level='info', exp_name='demo_generate_data')
_logger = logging.getLogger(f"sr_agent.{__name__}")
MAX_VAR_NUM = 3
SAMPLE_NUM = 100
BATCH_SIZE = 4
RANDOM_STATE = 42

# 生成随机公式
_logger.info("Creating equation generator...")
eq_generator = BaseEqGenerator.create(
    "gplearn",
    n_variables=MAX_VAR_NUM,
    random_seed=RANDOM_STATE,
    const_range=(-1.0, 1.0),
    depth_range=(2, 4),
    n_var_range=(1, MAX_VAR_NUM + 1),
)

# 生成随机数据
_logger.info("Creating data generator...")
data_generator = BaseDataGenerator.create(
    "uniform",
    sample_num=SAMPLE_NUM,
    random_seed=RANDOM_STATE,
    range=(-10.0, 10.0),
)

# 将公式编码成 token 和 index，供后续模型训练使用
_logger.info("Creating equation embedder...")
equation_embedder = EquationEmbedder(
    d_model=128, 
    max_variables=MAX_VAR_NUM,
    operands=eq_generator.symbols
)

# 将上面两个 generator 拼起来，生成公式-数据对
_logger.info("Creating dataset and dataloader...")
dataset = DataEqDataset(
    max_var_num=MAX_VAR_NUM,
    eq_generator=eq_generator,
    data_generator=data_generator,
    n_samples=None, # 默认 None 表示无限生成公式-数据对
    random_state=RANDOM_STATE,
    equation_embedder=equation_embedder, # 可选，如果提供则返回 token 和 index
)
dataloader = D.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=dataset.collate_fn,
    sampler=InfiniteSampler(), # 无限采样器，配合 n_samples=None 可以无限生成数据
)

cnt = 0
start_time = time.time()
_logger.info("Start iterating dataloader...")
for batch in dataloader:
    for formula in batch["formula"]:
        cnt += 1
        print(f"{cnt:03d}. {formula.to_str()}")
    print(f">>> batch data shape: {tuple(batch['data'].shape)}")

    # 可以无限生成，但作为 demo 1 秒就够了
    if time.time() - start_time > 1:
        break

_logger.info(
    f"Finished iterating dataloader.\n"
    f"Total samples: {cnt}.\n"
    f"Time elapsed: {time.time() - start_time:.2f} seconds.\n"
    f"Average speed: {cnt / (time.time() - start_time):.2f} samples/second."
)