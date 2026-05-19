#!/usr/bin/env python3
"""Compute pass^1 / DB-match / cost-per-conv / mean-turns for the flashlite + qwen
declarative scaling probe and write a CSV + console summary.

Reads:  tau2-bench/data/simulations/<condition>/results.json
Writes: results/scaling_flashlite_qwen.csv (machine-readable summary)
Stdout: human-readable table

Cost is computed from per-message usage where available; for gemini-3.1-flash-lite
we fall back to published Gemini 2.5 Flash-Lite pricing (Google AI Studio, paid
tier, non-cached input).  Qwen 3.5 on local Ollama has no API charge so cost is
$0.00 by definition; we still report mean tokens-in/tokens-out so the comparison
is meaningful.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM_ROOT = ROOT / "tau2-bench" / "data" / "simulations"
OUT_CSV = ROOT / "results" / "scaling_flashlite_qwen.csv"

# (Agent label, Retrieval label, Model label, sim directory)
CONDITIONS = [
    ("Baseline",    "Golden",    "gemini-3.1-flash-lite", "baseline_golden_flashlite"),
    ("Declarative", "Golden",    "gemini-3.1-flash-lite", "declarative_v2_golden_flashlite"),
    ("Baseline",    "Embedding", "gemini-3.1-flash-lite", "baseline_embedding_flashlite"),
    ("Declarative", "Embedding", "gemini-3.1-flash-lite", "declarative_v2_embedding_flashlite"),
    ("Imperative",  "Golden",    "gemini-3.1-flash-lite", "imperative_v2_golden_flashlite"),
    ("Imperative",  "Embedding", "gemini-3.1-flash-lite", "imperative_v2_embedding_flashlite"),
    ("Baseline",    "Golden",    "qwen3.5:9b (ollama)",   "baseline_golden_qwen"),
    ("Declarative", "Golden",    "qwen3.5:9b (ollama)",   "declarative_v2_golden_qwen"),
]

# Published Google AI Studio paid-tier pricing for Gemini 2.5 Flash-Lite
# (gemini-3.1-flash-lite is the same Flash-Lite family).  Source: Google
# pricing page, non-cached.  Implicit context caching would reduce input cost
# further; the figure here is therefore an upper bound.
FLASHLITE_IN_PER_TOKEN  = 0.10 / 1_000_000   # $0.10 / 1M input tokens
FLASHLITE_OUT_PER_TOKEN = 0.40 / 1_000_000   # $0.40 / 1M output tokens

# Local Ollama qwen3.5:9b: no API cost.
QWEN_IN_PER_TOKEN  = 0.0
QWEN_OUT_PER_TOKEN = 0.0


@dataclass
class Summary:
    agent: str
    retrieval: str
    model: str
    condition: str
    total_sims: int
    infra_errors: int
    n_eval: int
    pass1: float | None
    db_match: float | None
    mean_turns: float | None
    mean_tokens_in: float | None
    mean_tokens_out: float | None
    mean_cost_usd: float | None
    note: str = ""


def _per_sim_tokens(sim: dict) -> tuple[int, int]:
    tin = tout = 0
    for m in sim.get("messages") or []:
        u = m.get("usage") or {}
        if not isinstance(u, dict):
            continue
        tin  += int(u.get("prompt_tokens")     or u.get("input_tokens")  or 0)
        tout += int(u.get("completion_tokens") or u.get("output_tokens") or 0)
    return tin, tout


def _is_success(reward: float | None) -> bool:
    return reward is not None and abs(reward - 1.0) < 1e-6


def analyze(condition_dir: str, model: str) -> dict | None:
    path = SIM_ROOT / condition_dir / "results.json"
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    sims = data.get("simulations") or []
    total = len(sims)
    infra = sum(1 for s in sims if s.get("termination_reason") == "infrastructure_error")
    evaluated = [
        s for s in sims
        if s.get("termination_reason") != "infrastructure_error"
        and s.get("reward_info") is not None
    ]
    n_eval = len(evaluated)
    if n_eval == 0:
        return dict(total=total, infra=infra, n_eval=0)

    pass1 = sum(1 for s in evaluated if _is_success(s["reward_info"].get("reward"))) / n_eval
    db_match = sum(
        1 for s in evaluated
        if s["reward_info"].get("db_check") and s["reward_info"]["db_check"].get("db_match")
    ) / n_eval

    turns, tins, touts, costs = [], [], [], []
    for s in evaluated:
        n_turn = sum(1 for m in (s.get("messages") or []) if m.get("role") in ("user", "assistant"))
        turns.append(n_turn)
        tin, tout = _per_sim_tokens(s)
        tins.append(tin); touts.append(tout)

        # Prefer tau2's tracked agent_cost when populated.
        cost = s.get("agent_cost")
        if not cost:
            if "flash-lite" in model:
                cost = tin * FLASHLITE_IN_PER_TOKEN + tout * FLASHLITE_OUT_PER_TOKEN
            elif "ollama" in model or "qwen" in model.lower():
                cost = 0.0
            else:
                cost = 0.0
        costs.append(cost)

    mean = lambda xs: (sum(xs) / len(xs)) if xs else None
    return dict(
        total=total, infra=infra, n_eval=n_eval,
        pass1=pass1, db_match=db_match,
        mean_turns=mean(turns), mean_tin=mean(tins), mean_tout=mean(touts),
        mean_cost=mean(costs),
    )


def fmt(x, fmt_str):
    return fmt_str.format(x) if x is not None else "n/a"


def main():
    rows: list[Summary] = []
    print()
    header = (
        f"{'Agent':12s} {'Retrieval':10s} {'Model':24s} "
        f"{'Tot':>3} {'Inf':>3} {'N':>3} {'Pass^1':>7} {'DBmtc':>7} "
        f"{'Turns':>6} {'In_tok':>7} {'Out_tok':>7} {'Cost':>9}"
    )
    print(header)
    print("-" * len(header))

    for agent, retr, model, cdir in CONDITIONS:
        r = analyze(cdir, model)
        if r is None:
            print(f"{agent:12s} {retr:10s} {model:24s} {'-':>3} {'-':>3} {'-':>3}   NOT RUN")
            rows.append(Summary(agent, retr, model, cdir, 0, 0, 0,
                                None, None, None, None, None, None,
                                note="not run"))
            continue
        if r["n_eval"] == 0:
            print(f"{agent:12s} {retr:10s} {model:24s} {r['total']:>3d} {r['infra']:>3d} {r['n_eval']:>3d}   "
                  f"all infra-error")
            rows.append(Summary(agent, retr, model, cdir, r["total"], r["infra"], 0,
                                None, None, None, None, None, None,
                                note="all sims infra-error"))
            continue
        print(
            f"{agent:12s} {retr:10s} {model:24s} "
            f"{r['total']:>3d} {r['infra']:>3d} {r['n_eval']:>3d} "
            f"{r['pass1']:>7.3f} {r['db_match']*100:>6.1f}% "
            f"{r['mean_turns']:>6.1f} {r['mean_tin']:>7.0f} {r['mean_tout']:>7.0f} "
            f"${r['mean_cost']:>7.4f}"
        )
        rows.append(Summary(
            agent, retr, model, cdir,
            r["total"], r["infra"], r["n_eval"],
            r["pass1"], r["db_match"], r["mean_turns"],
            r["mean_tin"], r["mean_tout"], r["mean_cost"],
        ))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "agent", "retrieval", "model", "condition",
            "total_sims", "infra_errors", "n_eval",
            "pass1", "db_match", "mean_turns",
            "mean_tokens_in", "mean_tokens_out", "mean_cost_usd",
            "note",
        ])
        for r in rows:
            w.writerow([
                r.agent, r.retrieval, r.model, r.condition,
                r.total_sims, r.infra_errors, r.n_eval,
                "" if r.pass1 is None else f"{r.pass1:.4f}",
                "" if r.db_match is None else f"{r.db_match:.4f}",
                "" if r.mean_turns is None else f"{r.mean_turns:.2f}",
                "" if r.mean_tokens_in is None else f"{r.mean_tokens_in:.0f}",
                "" if r.mean_tokens_out is None else f"{r.mean_tokens_out:.0f}",
                "" if r.mean_cost_usd is None else f"{r.mean_cost_usd:.6f}",
                r.note,
            ])

    print()
    print(f"CSV written: {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
