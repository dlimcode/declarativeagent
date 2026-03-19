# Declarative vs Imperative Agent Orchestration on tau3-bench

Comparing natural-language workflow guidance (skill files) vs programmatic workflow control (state machine) for LLM agents on the [tau3-bench](https://github.com/sierra-research/tau2-bench) banking_knowledge domain.

**Course:** Gen AI with LLMs, SMU Singapore
**Deadline:** April 11, 2026

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
# Edit tau2-bench/.env with OPENAI_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY

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
  analysis/                 # Post-experiment analysis (TBD)
scripts/
  run.py                    # Wrapper: path setup + registration + tau2 CLI
  run_pilot.py              # Pilot orchestrator
  run_experiment.py         # Full experiment orchestrator
  day1_model_test.py        # Model validation script
configs/
  experiment.yaml           # Experiment parameters (model, conditions, seed)
tau2-bench/                 # Cloned dependency (gitignored)
research/                   # Design docs and literature review
```

## Models

| Role | Model | Cost |
|------|-------|------|
| Agent LLM | GLM-4.7-Flash via OpenRouter | Free |
| User simulator | GPT-4o-mini | ~$5 total |

## Architecture

Our code plugs into tau3-bench via its registry system — **zero modifications to tau3-bench source code**. A wrapper script (`scripts/run.py`) adds our agents to the import path, registers them, then delegates to the standard `tau2` CLI.

## Key Docs

| Document | Purpose |
|----------|---------|
| `research/experiment-design.md` | Full experiment design |
| `report-outline.md` | Report structure with section owners |
| `team-briefing.md` | Teammate info pack |
| `configs/experiment.yaml` | Experiment configuration |

## Requirements

- Python ≥3.12, <3.14
- [uv](https://docs.astral.sh/uv/) package manager
- macOS or Linux
- API keys: OpenAI, OpenRouter (Gemini optional as fallback)
