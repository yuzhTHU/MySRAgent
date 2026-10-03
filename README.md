# SRHarness: A Harness for Agentic Symbolic Regression

[English](README.md) | [简体中文](README.zh.md)

SRHarness is a domain-specific runtime for **agentic symbolic regression**. It lets a large language model inspect numerical observations, choose scientific operations, evaluate competing hypotheses, and refine a symbolic expression over a long search trajectory.

This repository contains the research code for **“SRHarness: A Harness for Agentic Symbolic Regression.”** The Python package is `sr_harness`, and its primary agent class remains `SRAgent`.

> **Research-code status:** the project is under active development. Experiment-scale runs can make many paid LLM requests and may invoke external solvers. Start with a small `R-C-L-K` configuration and inspect the generated logs before launching a benchmark campaign.

## Why SRHarness?

SRHarness organizes agentic equation discovery around three mechanisms:

- **Composable scientific actions.** Analysis, fitting, evaluation, and search tools share a common interface. Actions can operate on raw variables, transformed expressions, residuals, and other candidate-derived views.
- **Persistent scientific state.** Candidate formulas, numerical metrics, complexity, evidence, and provenance survive beyond a single conversation. Compact Pareto and top-candidate views expose useful state back to the model.
- **Trajectory lifecycle management.** A configurable `R-C-L-K` scheduler coordinates restarts, independent branches, refinement steps, and local response sampling while preserving useful intermediate results.

The included action library covers statistical and relationship analysis, formula/code evaluation, constant and structured fitting, PySR and SINDy integration, code execution, skills, and final formula submission. Tools can be enabled, disabled, or extended without changing the main agent loop.

## Results at a Glance

The accompanying paper evaluates SRHarness on LLM-SRBench, including LSR-Synth, LSR-Transform, and an anonymized LSR-Transform variant that removes scientific descriptions and variable semantics.

| Method / backbone | LSR-Transform SA | LSR-Transform-Anon SA |
|---|---:|---:|
| SRHarness + DeepSeek-v4-flash-0731 | **93.69%** | **72.97%** |
| SR-Scientist + DeepSeek-v4-flash-0731 | 62.16% | 39.64% |
| Codex + DeepSeek-v4-flash-0731 | — | 20.72% |

These are symbolic-accuracy results reported in the manuscript. See the paper for the complete numerical, symbolic, complexity, resource, and ablation results, as well as the exact evaluation protocol.

## Installation

### Requirements

- Linux is the primary tested platform.
- Python **3.12 or newer** is required.
- Git and a working C/C++ toolchain are recommended.
- Some optional actions have additional requirements, such as Julia for PySR or PyTorch for neural components.

The following setup mirrors [`scripts/install.sh`](scripts/install.sh) while using HTTPS clone URLs:

```bash
git clone https://github.com/yuzhTHU/MySRAgent.git SRHarness
cd SRHarness

conda create -p ./venv python=3.12 -y
conda activate ./venv

# nd2py is currently installed from source.
git clone https://github.com/yuzhTHU/nd2py.git ./third-party/nd2py
pip install -e ./third-party/nd2py

# Core package plus development/test dependencies.
pip install -e ".[dev]"
```

Install optional components as needed:

```bash
pip install -e ".[web]"       # Web search-tree viewer
pip install -e ".[tools]"     # PySR, gplearn, PySINDy, and PDF integrations
pip install -e ".[nn]"        # Experimental neural components
pip install -e ".[all]"       # Everything above
```

## Provider Configuration

Copy the tracked environment template once, then fill in only the providers you use:

```bash
test -f .env || cp .env.sample .env
```

For example, OpenRouter requires:

```dotenv
OPENROUTER_API_KEY="sk-or-v1-..."
```

The code also contains adapters for DeepSeek, Gemini, OpenAI/Azure OpenAI, SiliconFlow, LM Studio, and manual interaction. See [`.env.sample`](.env.sample) for the corresponding variables. Never commit `.env`; it is ignored by Git.

## Quick Start

Run a small synthetic problem:

```bash
conda activate ./venv

sr-harness run \
  --equation "y = sin(x1 - x2)" \
  --x-low -10 \
  --x-high 10 \
  --llm-provider openrouter \
  --llm-model deepseek/deepseek-v4-flash \
  --force-initial-diagnostics \
  -R 1 -C 1 -L 3 -K 1
```

This command performs paid API calls. Its search budget is controlled by:

| Symbol | Meaning |
|---|---|
| `R` | restart rounds initialized from persistent historical candidates |
| `C` | independent conversational branches per restart |
| `L` | refinement steps per branch |
| `K` | locally sampled responses per refinement step |

The nominal number of model responses is approximately `R × C × L × K`, although retries and provider behavior can affect actual usage.

### Python API

```python
import numpy as np
from sr_harness import SRAgent

x1 = np.linspace(-3.0, 3.0, 100)
x2 = np.linspace(3.0, -3.0, 100)

agent = SRAgent(
    llm_provider="openrouter",
    llm_model="deepseek/deepseek-v4-flash",
    max_restart_loop=1,
    global_width=1,
    max_refinement_depth=3,
    local_sample_size=1,
    save_path="logs/python_api_demo",
)

result = agent.run(
    X={"x1": x1, "x2": x2},
    y={"y": np.sin(x1 - x2)},
    problem_description="Discover y as a function of x1 and x2.",
)
best = result["candidates"][result["best_candidate"]]
print(best["formula"])
```

## LLM-SRBench Evaluation

Download the benchmark data. Git LFS may be required:

```bash
git lfs install
git clone https://huggingface.co/datasets/nnheui/llm-srbench \
  ./data/llm-srbench-data
```

Run one LSR-Transform problem before scaling up:

```bash
sr-harness bench \
  --algorithm my_sr_agent \
  --datasets lsrtransform \
  --problem-names II.6.15b_1_0 \
  --exp-name smoke_lsrtransform \
  --llm-provider openrouter \
  --llm-model deepseek/deepseek-v4-flash \
  -R 1 -C 1 -L 3 -K 1
```

Add `--anonymize` to replace variable names and scientific descriptions with generic input/output labels while leaving the numerical observations unchanged:

```bash
sr-harness bench \
  --algorithm my_sr_agent \
  --datasets lsrtransform \
  --problem-names II.6.15b_1_0 \
  --exp-name smoke_lsrtransform_anon \
  --anonymize \
  --llm-provider openrouter \
  --llm-model deepseek/deepseek-v4-flash \
  -R 1 -C 1 -L 3 -K 1
```

The benchmark entry point also contains adapters for conventional and LLM-based baselines; `sr-harness bench --help` lists its general options, while each adapter defines its method-specific flags. Paper-scale reproduction requires the exact model, toolset, data split, token limit, seed, and `R-C-L-K` configuration reported with each experiment; the smoke commands above intentionally use a much smaller budget.

## Logs and Web Visualization

`SearchRunState` always keeps the live run in memory. When `save_path` is enabled, it also writes:

- `run.json`: the globally unique run ID and agent metadata;
- `nodes.jsonl`: search nodes, parent relations, prompts, actions, results, and usage;
- `result.json`: candidates plus the Pareto-front and best-candidate indices;
- `response.jsonl`: raw model responses and token/cost accounting;
- `tool_calls.jsonl`: tool invocations and outputs;
- text logs and entry-point-specific result files.

With `save_path=None`, search identity, parent relations, candidates, and results remain fully
available through `agent.run_state`, while no search-state files are created.

Install and launch the web viewer:

```bash
pip install -e ".[web]"
sr-harness web --log-dir logs --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000/>. The Web API and search viewer read the active session's
in-memory `SearchRunState`; they do not depend on persisted run files.

![SRHarness Web search-tree viewer](assets/web.png)

The workbench opens on **Data & Problem**. Select or upload a CSV, assign one target and one or
more features, edit variable descriptions, and drag variables into the X/Y/Hue/Size slots for a
quick relationship preview. SRHarness generates the initial system and user prompts from that
configuration; either prompt remains editable before the run starts. The included `demo.csv`
contains three input columns (including one categorical column) and one numeric target.

During a run, **Timeline** shows model reasoning, tool calls, results, token/cost usage, and control
events, while **Current Context** exposes the messages associated with each R-C-L node. The search
tree and candidate panel stay linked to those nodes and can switch between all ranked candidates
and the Pareto front. Guidance, model changes, pause/resume, stop, and inline `ask_human` replies
take effect at safe operation boundaries. The interface supports Chinese/English text, light/dark
themes, and resizable or collapsible side panels.

### Research backends, subagents, and live control

The default tool set includes recursive per-subtree EIC diagnostics (`evaluate_eic`), an actual
MDLformer-guided SR4MDL search (`sr4mdl`), NDformer-guided network-dynamics search (`nd2`), bounded
symbolic-regression hypothesis/critique delegation (`delegate_subagent`), web search, and PDF
reading. Configure heavyweight external projects with `SR4MDL_HOME` and `ND2_HOME`, and point
`SR4MDL_CHECKPOINT` to the trained MDLformer checkpoint. Repositories placed at
`third-party/SR4MDL` and `third-party/ND2` are discovered automatically. When `evaluate_eic` is
enabled, each newly generated scalar candidate receives a lightweight structural audit whose
diagnostics are retained in candidate state. Documentation for EIC, SR4MDL, and ND2 is exposed as
runtime read-only skills by each tool's `get_doc()` method.

`SRAgentInteractive` accepts an `InteractionManager` that connects its shared search loop to a
frontend. Its default `TerminalInteractionManager` handles `ask_human` in a terminal. The Web
workbench injects a `WebInteractionManager`, which binds the run state and workspace to its session,
handles pause/resume/stop and queued guidance at safe boundaries, and publishes model, tool, and
candidate events. Frontend adapters do not own the scientific search state or duplicate the R-C-L
loop. Model auto-routing can use a cheap base backend for simple/early requests and an optional
strong backend for complex or stagnated searches. Configure
`strong_llm_provider`/`strong_llm_model`, or pass `auto_routing=False` to keep every request on the
base backend.

## Evaluation and Reproducibility Notes

- Benchmark test observations are not exposed during search or candidate selection.
- The agent can reserve part of the visible training data for random or OOD-style validation using `--validation-fraction` and `--split-by`.
- Numerical predictions are evaluated through the shared benchmark pipeline. Symbolic equivalence is implemented in [`src/sr_harness/utils/symbolic_acc.py`](src/sr_harness/utils/symbolic_acc.py).
- Logs preserve prompts, model responses, tool calls, candidate provenance, token usage, and recorded cost so that a run can be audited after completion.
- API behavior, model aliases, prices, and stochastic outputs can change over time. Record the exact provider model identifier, source revision, arguments, and environment for serious comparisons.

## Extending SRHarness

New scientific actions inherit `BaseTool`, declare stable metadata, and return a serializable result. Candidate-producing actions should use the shared evaluation contract so their formulas, train/validation metrics, complexity, diagnostics, and provenance can enter persistent scientific state consistently.

See:

- [`src/sr_harness/README.md`](src/sr_harness/README.md) for the agent loop and internal architecture;
- [`src/sr_harness/tools/README.md`](src/sr_harness/tools/README.md) for the action API and custom-tool guide;
- [`tests/README.md`](tests/README.md) for testing conventions.

## Project Layout

```text
├── src/sr_harness/          # Python package
│   ├── agents/              # Batch and interactive SRAgent implementations
│   ├── api/                 # BaseAPI and LLM provider adapters
│   ├── core/                # API, tool, candidate, node, and run-state structures
│   ├── interaction/         # Terminal and Web interaction managers
│   ├── runtime/             # Model routing and interaction control
│   ├── cli/                 # sr-harness subcommands
│   ├── parser/              # Native/text/JSON/XML tool-call parsing
│   ├── tools/               # Scientific actions and shared evaluation contract
│   ├── skills/              # Reusable agent-facing scientific instructions
│   ├── web/                 # Interactive workbench backend and static UI
│   ├── utils/               # Metrics, symbolic accuracy, logging, and utilities
│   └── _vendor/             # Integrated benchmark/baseline adapters
├── tests/                   # Unit and integration tests
├── scripts/                 # Experiment and analysis utilities
├── analysis/                # Analysis notebooks
├── data/                    # Local datasets; ignored by Git
├── logs/                    # Run artifacts; ignored by Git
└── playground/              # Temporary experiments; ignored by Git
```

Repository conventions:

- Add user-facing commands as `sr-harness` subcommands under `src/sr_harness/cli/`.
- Put experiment and analysis utilities under `scripts/`.
- Name analysis notebooks as `YYMMDD_description.ipynb` and avoid committing large outputs.
- Treat `data/`, `logs/`, and `playground/` as local working directories.

## Testing

The default test configuration excludes tests marked `slow` or `paid`:

```bash
python -m pytest tests/ -v
```

Run paid or slow integration tests only when the required services and budget are available.

## Citation

If you use this code, please cite **“SRHarness: A Harness for Agentic Symbolic Regression.”** A copy-ready BibTeX entry and public paper link will be added when the paper record becomes publicly available.

## License

SRHarness is released under the [MIT License](LICENSE).
