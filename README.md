# Declarative vs Imperative Agent Orchestration 
@Application on tau-knowledge

Comparing natural-language workflow guidance (skill files) vs programmatic workflow control (state machine) for LLM agents on the [tau3-bench](https://github.com/sierra-research/tau2-bench) banking_knowledge domain.


## Research Question

Does declarative orchestration (skill files appended to the system prompt) match or exceed imperative orchestration (8-phase state machine with tool restriction) for agent task completion, and does retrieval quality moderate this effect?

## Experiment Design

3×2 factorial: 3 agents × 2 retrieval configs = 6 conditions, 97 tasks × 3 trials = 1,746 simulations.

|  | golden_retrieval | bm25 |
|--|--|--|
| **Baseline** (tau3-bench default) | Condition A | Condition B |
| **Declarative** (+ skill files) | Condition C | Condition D |
| **Imperative** (state machine) | Condition E | Condition F |

## Quick Start

```bash
# 1. Setup
make setup                          # clones tau2-bench v1.0.0, installs deps

# 2. Add API keys
cp tau2-bench/.env.example tau2-bench/.env
# Edit tau2-bench/.env with OPENAI_API_KEY (user sim) and DASHSCOPE_API_KEY (agent LLM)

# 3. Verify
make check-agents                   # confirms custom agents register

# 4. Smoke test
make smoke-test-baseline            # 1 task with baseline agent

# 5. Pilot
make pilot                          # 5 tasks × 6 conditions (30 sims)

# 6. Full experiment
make experiment                     # 97 tasks × 6 conditions × 3 trials (1,746 sims)
```

## Project Structure

```
src/
  agents/
    declarative_agent.py    # Baseline + skill file injection
    imperative_agent.py     # 8-phase state machine with tool restriction
    state.py                # Shared Pydantic state model
    register.py             # Agent factory + tau2 registry integration
  skills/
    customer-interaction.md # Workflow: greet, identify, verify, confirm
    knowledge-discovery.md  # KB search + discoverable tool unlock pattern
    banking-procedures.md   # Domain knowledge per operation type
  analysis/
    extract_metrics.py      # Parse results.json -> per-simulation CSVs
    statistical_tests.py    # Chi-squared, McNemar, bootstrap CI, Cohen's h
    error_analysis.py       # Error taxonomy detection (E1-E9)
    plots.py                # Generate experiment figures
scripts/
  run.py                    # Wrapper: path setup + registration + tau2 CLI
  run_pilot.py              # Pilot orchestrator
  run_experiment.py         # Full experiment orchestrator
  day1_model_test.py        # Model selection validation (historical)
configs/
  experiment.yaml           # Main experiment config (seed=300, 6 conditions, 3 trials)
  generalizability.yaml     # GPT-4o-mini generalizability check (10 tasks, 1 trial)
tau2-bench/                 # Cloned dependency (gitignored)
```

## Models

| Role | Model | Cost |
|------|-------|------|
| Agent LLM | Qwen3.5-Flash via DashScope | ~$0.10/$0.40 per M tokens |
| User simulator | GPT-4o-mini | ~$5 total |

## Architecture

Our code plugs into tau3-bench via its registry system — **zero modifications to tau3-bench source code**. A wrapper script (`scripts/run.py`) adds our agents to the import path, registers them, then delegates to the standard `tau2` CLI.

## Reproducibility

### Headline numbers in the paper

The paper reports results for three agents (Baseline, DeclarativeAgent, ImperativeAgent) on five language models (Qwen3.5-Flash, Claude Haiku-4.5, Gemini-3.1-Flash-Lite, DeepSeek-v4-Flash, DeepSeek-v4-Pro) under two retrieval regimes (golden and a local `all-MiniLM-L6-v2` embedding retriever). All conditions share:

- **Domain:** `banking_knowledge` (97 tasks from $\tau$-Knowledge)
- **Random seed:** 300
- **Trials:** 1
- **Agent `max_tokens`:** 4096
- **User simulator:** `deepseek/deepseek-v4-pro` for all conditions *except* the Claude Haiku-4.5 conditions, which use `anthropic/claude-haiku-4-5-20251001` on both sides (matches the original benchmark pairing and avoids a cross-provider user channel for Anthropic-model rows)
- **Embedding retriever:** local `sentence-transformers` `all-MiniLM-L6-v2`, served via the `deepseek_embeddings` retrieval variant registered in `scripts/run.py`

### Configs that produced each table row

| Paper cells | Config |
|---|---|
| DeepSeek-v4-Pro rows (golden & embedding) | `configs/experiment-v2.yaml` |
| DeepSeek-v4-Flash *imperative* rows | `configs/imp-deepseek-flash-golden.yaml`, `configs/imp-deepseek-flash-emb.yaml` |
| DeepSeek-v4-Flash baseline + declarative rows | produced earlier in the project; sim records preserved under `baseline_golden_flash`, `baseline_embedding_flash`, `declarative_v2_golden_flash`, `declarative_v2_embedding_flash`. To re-run, copy one of the `imp-deepseek-flash-*.yaml` configs and swap the `agent:` field |
| Gemini-3.1-Flash-Lite rows | `configs/scaling-flash-lite.yaml`, `configs/embed-retry-flashlite-c1.yaml` (concurrency=1 top-up for the imperative-embedding cell) |
| Claude Haiku-4.5 golden Baseline | `configs/baseline-haiku.yaml` |
| Claude Haiku-4.5 golden Declarative + Imperative | `configs/haiku-golden-rerun.yaml` *(see note below)* |
| Claude Haiku-4.5 embedding (all three agents) | `configs/embed-missing-haiku.yaml` |
| Qwen3.5-Flash rows | `configs/experiment-research.yaml` |

**Note on `haiku-golden-rerun.yaml`.** An earlier batch of Haiku-golden runs (saved to `*_haiku_backup` directories) was inadvertently resumed with `--agent-llm deepseek-v4-pro` on 2026-05-05, contaminating 86 of 91 declarative-golden sims and 87 of 92 imperative-golden sims with DeepSeek-Pro responses while the metadata still claimed Haiku. The clean rerun uses fresh save names (`declarative_v2_golden_haiku`, `imperative_v2_golden_haiku`) so tau2's auto-resume cannot reach the contaminated checkpoints. The numbers in the paper come from the clean rerun; the `_backup` dirs are preserved on disk as forensic evidence.

### Running an experiment

```bash
# One-time setup
make setup                                    # clones tau2-bench v1.0.0, installs deps

# Add API keys to tau2-bench/.env as needed:
#   ANTHROPIC_API_KEY    (Haiku conditions)
#   DEEPSEEK_API_KEY     (DeepSeek conditions + user simulator on non-Haiku rows)
#   GEMINI_API_KEY       (Flash-Lite conditions)
#   DASHSCOPE_API_KEY    (Qwen3.5-Flash conditions)

# Run one scaling-probe leg, e.g. Flash-Lite golden
cd tau2-bench && uv run python ../scripts/run_experiment.py \
    --config ../configs/scaling-flash-lite.yaml \
    --wave-size 1

# Clean Haiku-golden rerun
cd tau2-bench && uv run python ../scripts/run_experiment.py \
    --config ../configs/haiku-golden-rerun.yaml \
    --wave-size 1
```

`scripts/run_experiment.py` bakes in `--auto-resume`: rerunning the same command picks up evaluable sims already on disk and retries only the infrastructure-error rows.

### Extracting the table numbers

After all conditions in scope finish, generate the scaling-probe summary CSV:

```bash
python3 scripts/compute_scaling_summary.py
# writes results/scaling_flashlite_qwen.csv
```

The raw per-sim records live in `tau2-bench/data/simulations/<condition>/results.json`. The summary script reduces each condition to mean Pass^1, DB-match, mean turns, mean tokens, and mean Cost/Task (from per-message `agent_cost`). The `CONDITIONS` list inside `scripts/compute_scaling_summary.py` controls which conditions get summarised.

### Verifying that the configured model actually answered

A run can silently swap models if `--agent-llm` is mis-passed during a resume (the Haiku contamination above is exactly this failure mode). The top-level `info.agent_info.llm` records what the run *should* have called; to confirm what was *actually* called, walk the per-message `raw_data.model` field:

```python
import json
d = json.load(open('tau2-bench/data/simulations/<condition>/results.json'))
for s in d['simulations']:
    for m in s['messages']:
        if m.get('role') == 'assistant' and m.get('raw_data'):
            print(s['task_id'], m['raw_data'].get('model'))
            break
```

If the printed model does not match the configured `agent_llm` for every task, the condition is contaminated and needs a clean rerun under a fresh save name.

### Software pinning

- Python 3.12 via [`uv`](https://docs.astral.sh/uv/)
- `tau2-bench` v1.0.0 (cloned by `make setup`, gitignored)
- `litellm` (transitive) for provider abstraction and cost tracking
- `sentence-transformers` for the local MiniLM embedder

## Requirements

- Python ≥3.12, <3.14
- [uv](https://docs.astral.sh/uv/) package manager
- macOS or Linux
- API keys: OpenAI (user simulator), DashScope (agent LLM)
