# When Skills Beat Scale: Declarative Agent Skills Improve Smaller AI Agents in Tool-Use Workflows

Code and experiment configurations for the paper *When Skills Beat Scale: Declarative Agent Skills Improve Smaller AI Agents in Tool-Use Workflows* (M. Danish Lim, I. Danial Bin Sharudin, Wen Han Chen, Cedric Lim, Laura Wynter — SMU, 2026).

We compare three orchestration paradigms for LLM agents on the τ-Knowledge banking benchmark: an unscaffolded **BaselineAgent**, a **DeclarativeAgent** that loads three natural-language skill files into its system prompt, and an **ImperativeAgent** built around a finite-state machine with deterministic phase transitions, verification gates, and per-tool retry policies. Each is evaluated across four LLMs (Claude-Haiku-4.5, Gemini-3.1-Flash-Lite, DeepSeek-v4-Flash, DeepSeek-v4-Pro) and two retrieval methods (golden oracle and `all-MiniLM-L6-v2` dense embedding).

## Headline findings

- **Skills act as a low-cost capability prosthetic for smaller models.** DeclarativeAgent lifts pass¹ on Claude-Haiku-4.5 by **+0.325** (0.126 → 0.451, a 3.6× multiplier) under golden retrieval, with non-negative gains on every stronger model.
- **Programmatic orchestration is brittle.** ImperativeAgent underperforms baseline on every model under both retrieval methods and fails to deliver its expected safety/compliance benefits (over-retry rate 4–7× higher than baseline; unauthorized-write rate not lower).
- **Retrieval quality dominates.** Switching from oracle to dense-embedding retrieval drops pass¹ sharply for all agents on all models; skill files cannot compensate for fundamentally wrong evidence.

See Tables 3–6 of the paper for the full breakdown.

## Repository layout

```
src/
  agents/
    declarative_agent.py    # LLMAgent + <skills> block injection
    imperative_agent.py     # finite-state machine: GREETING → TRIAGE → VERIFICATION →
                            # PLANNING → EXECUTION → CONFIRMATION → COMPLETE
                            # (+ ADVISORY branch, ESCALATE terminal)
    state.py                # AgentState Pydantic model (gates, task queue, retry counts)
    cached_generate.py      # generate() wrapper with prompt caching + reasoning passthrough
    register.py             # factories registered into the tau2 agent registry
  skills/
    banking-procedures.md   # operations → preconditions, ordering, argument shapes
    customer-interaction.md # four-step conversational structure with multi-request inventory
    knowledge-discovery.md  # when/how to query the KB; recovery on miss
  analysis/
    safety_metrics.py       # offline replay: unauthorized-write, over-retry, trajectory length
scripts/
  run.py                    # adds src/ to path, registers agents, delegates to tau2 CLI
  run_experiment.py         # multi-condition orchestrator driven by a YAML config
configs/
  experiment.yaml           # main experiment (3 agents × 2 retrievals × 97 tasks)
  experiment-embedding.yaml # embedding-retrieval comparison (DeepSeek-v4-Pro)
  ...                       # additional per-model and ablation configs
tau2-bench/                 # cloned dependency; gitignored
```

## Quick start

```bash
# 1. Clone tau2-bench (v1.0.0) and install dependencies
make setup

# 2. API keys (one or more of the agent providers, plus the user-sim provider)
cp tau2-bench/.env.example tau2-bench/.env
# Edit tau2-bench/.env to add: OPENAI_API_KEY (user sim) and the agent-side keys
# you need: ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, DASHSCOPE_API_KEY

# 3. Sanity-check that the custom agents are registered
make check-agents

# 4. Single-task smoke tests (gpt-4o-mini, fast and cheap)
make smoke-test-baseline
make smoke-test-declarative
make smoke-test-imperative

# 5. Five-task pilot
make pilot

# 6. Full experiment (97 tasks, configurable agent/retrieval set)
make experiment
make experiment-embedding   # MiniLM dense-retrieval comparison

# 7. Aggregate + analyse results
make analyze
```

## Reproducing the paper

- **Benchmark:** [τ-Knowledge](https://github.com/sierra-research/tau2-bench), `banking_knowledge` domain (97 tasks, 698 KB documents, 14 permanent tools + 51 discoverable tools).
- **Primary metric:** pass¹ (single-trial task success via final database-state match against the gold trajectory).
- **Seed:** 300 (set in every experiment config).
- **User simulator:** `gpt-4o-mini` (held constant across all conditions to avoid self-play confounds).
- **Agent models:** Claude-Haiku-4.5, Gemini-3.1-Flash-Lite, DeepSeek-v4-Flash, DeepSeek-v4-Pro (each invoked with `max_tokens: 4096`).
- **Retrieval:** `golden_retrieval` (oracle minimal document set) and a dense retriever backed by `all-MiniLM-L6-v2` (registered as `deepseek_embeddings` in `scripts/run.py` for historical reasons).
- **Infrastructure-error policy:** simulations that exhaust LiteLLM auth retries are excluded from the denominator (≤1 task per condition in published runs).
- **Analysis:** `make analyze` parses `tau2-bench/data/simulations/*/results.json`, computes pass¹, DB-match, write accuracy, and the safety metrics in §6 of the paper, and writes CSV summaries under `reports/`.

## Architecture notes

Our code plugs into τ-Knowledge via its registry system with **no modifications to upstream tau2-bench source**. `scripts/run.py` adds `src/` to the Python path, registers our agent factories, and then delegates to the standard `tau2` CLI. Each agent subclasses `HalfDuplexAgent[AgentState]` and implements `get_init_state` and `generate_next_message` — the orchestrator owns all tool execution; the agent never touches the environment directly.

The ImperativeAgent's six structural strategies (explicit task queue, topological task ordering via Kahn's algorithm, state-driven phase transitions, hard verification gate, per-tool retry policy with deterministic escalation, and strict response-type enforcement) are described in §4 of the paper and implemented across `imperative_agent.py` and `state.py`.

## Requirements

- Python ≥ 3.12, < 3.14
- [`uv`](https://docs.astral.sh/uv/) package manager
- macOS or Linux
- API keys for at least one agent provider, plus OpenAI for the user simulator

## Citation

A bibtex entry will be added here when the paper is finalised. In the meantime, please cite as:

```
Lim, M. D., Sharudin, I. D. B., Chen, W. H., Lim, C., & Wynter, L. (2026).
When Skills Beat Scale: Declarative Agent Skills Improve Smaller AI Agents
in Tool-Use Workflows. Singapore Management University.
```
