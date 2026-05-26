# %%
import os
import sys
from pathlib import Path
ROOT = Path('..' if Path(os.getcwd()).name == "analysis" else '.').absolute()
os.chdir(ROOT)
sys.path.append(str(ROOT))
print(os.getcwd())

# %%
import dotenv
dotenv.load_dotenv()

# %%
import json
import pandas as pd

df_list = []
for exp_path in sorted(Path("logs/bench_sr_agent/bench_my_sr_agent/results/").glob('*.jsonl'), key=lambda x: x.stem):
    lines = [json.loads(line) for line in exp_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    successful_lines = [line for line in lines if 'error' not in line]
    result = successful_lines[-1] if successful_lines else lines[-1]
    df_list.append(result)
df_results = pd.DataFrame(df_list)

df_id = pd.DataFrame(df_results.pop('id_metrics').tolist(), index=df_results.index).add_prefix('id_')
df_results = df_results.join(df_id)

df_ood = pd.DataFrame(df_results.pop('ood_metrics').tolist(), index=df_results.index).add_prefix('ood_')
df_results = df_results.join(df_ood)

# %%
from bench_sr_agent import DATASET_SPLITS, load_problems, build_argparser

parser = build_argparser()
args, _ = parser.parse_known_args()

datasets = list(DATASET_SPLITS.keys())
problems = []
for dataset in datasets:
    problems.extend(load_problems(dataset, args.data_root))


# %%
import logging
import nd2py as nd
from tqdm import tqdm
from sr_agent.utils import symbolic_acc, setup_logging

_logger = logging.getLogger(f"sr_agent.{__name__}")
setup_logging(info_level='debug')


for idx, row in tqdm(df_results.iterrows(), total=len(df_results), disable=True):
    for problem in problems:
        if row['equation_id'] == problem.equation_idx:
            break
    else:
        raise ValueError(f"Problem with equation_id {row['equation_id']} not found in loaded problems.")
    
    f_true = nd.parse(row['gt_expression'])
    f_pred = nd.parse(row['discovered_expression'])
    data = {var: problem.samples['train'][:, idx] for idx, var in enumerate(problem.symbols)}
    result = symbolic_acc(f_true, f_pred, data, return_details=True, llm_provider='openrouter', llm_model='deepseek/deepseek-v4-flash')
    df_results.loc[idx, 'symbolic_acc'] = result['equivalent']
    _logger.info(f"[{idx+1}/{len(df_results)}] Problem ID: {row['equation_id']}, Symbolic Accuracy: {result['equivalent']}, Reason: {result['reason']}")

_logger.note(f"Symbolic accuracy evaluation completed for {len(df_results)} problems, accuracy: {df_results['symbolic_acc'].mean():.2%}")