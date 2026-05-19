#!/usr/bin/env python3
"""Render a standalone LaTeX report from results/scaling_flashlite_qwen.csv.

Output: reports/scaling-flashlite-qwen.tex (compiles with pdflatex, no
external bibliography).  Re-run after qwen sims finish to refresh the table.
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "scaling_flashlite_qwen.csv"
TEX_PATH = ROOT / "reports" / "scaling-flashlite-qwen.tex"


def load_rows():
    rows = []
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def f3(x):  return "--" if not x else f"{float(x):.3f}"
def f1pct(x):
    if not x:
        return "--"
    return f"{float(x) * 100:.1f}\\%"
def fnum(x, prec=1):
    if not x:
        return "--"
    return f"{float(x):.{prec}f}"
def fcost(x):
    if x is None or x == "":
        return "--"
    v = float(x)
    if v == 0.0:
        return "\\$0.0000"
    return f"\\${v:.4f}"


def fmt_n(row):
    total = int(row["total_sims"] or 0)
    infra = int(row["infra_errors"] or 0)
    n_eval = int(row["n_eval"] or 0)
    if n_eval == 0 and total == 0:
        return "not run"
    return f"{n_eval}/{total}"


def main():
    rows = load_rows()
    by_key = {(r["agent"], r["retrieval"], r["model"]): r for r in rows}

    def grab(agent, retrieval, model):
        return by_key.get((agent, retrieval, model))

    FL = "gemini-3.1-flash-lite"
    QW = "qwen3.5:9b (ollama)"

    fl_order = [
        ("Baseline",    "Golden"),
        ("Declarative", "Golden"),
        ("Imperative",  "Golden"),
        ("Baseline",    "Embedding"),
        ("Declarative", "Embedding"),
        ("Imperative",  "Embedding"),
    ]
    qw_order = [
        ("Baseline",    "Golden"),
        ("Declarative", "Golden"),
    ]

    def row_line(agent, retrieval, model):
        r = grab(agent, retrieval, model)
        if r is None:
            return (f"{agent} & {retrieval} & not run & -- & -- & -- & -- & -- \\\\")
        n_eval = int(r["n_eval"] or 0)
        if n_eval == 0:
            return (f"{agent} & {retrieval} & {fmt_n(r)} & -- & -- & -- & -- & -- \\\\")
        return (
            f"{agent} & {retrieval} & {fmt_n(r)} & "
            f"{f3(r['pass1'])} & {f1pct(r['db_match'])} & "
            f"{fnum(r['mean_turns'])} & {int(float(r['mean_tokens_in'])):,} & {fcost(r['mean_cost_usd'])} \\\\"
        )

    # Build a paired delta table: declarative vs baseline on golden retrieval
    def delta(agent_a, agent_b, retrieval, model):
        a = grab(agent_a, retrieval, model)
        b = grab(agent_b, retrieval, model)
        if not a or not b or not a["pass1"] or not b["pass1"]:
            return None
        return float(b["pass1"]) - float(a["pass1"])

    fl_pair_delta = delta("Baseline", "Declarative", "Golden", FL)
    qw_pair_delta = delta("Baseline", "Declarative", "Golden", QW)

    def delta_str(d):
        if d is None:
            return "n/a"
        sign = "+" if d >= 0 else ""
        return f"{sign}{d * 100:.1f} pp"

    today = dt.date.today().isoformat()
    flashlite_table = "\n        ".join(row_line(a, r, FL) for a, r in fl_order)
    qwen_table = "\n        ".join(row_line(a, r, QW) for a, r in qw_order)

    base_fl_g = grab("Baseline", "Golden", FL)
    dec_fl_g  = grab("Declarative", "Golden", FL)
    base_qw_g = grab("Baseline", "Golden", QW)
    dec_qw_g  = grab("Declarative", "Golden", QW)

    def pair_row(label, base, dec):
        if not base or not dec or not base["pass1"] or not dec["pass1"]:
            return f"{label} & -- & -- & -- \\\\"
        d = float(dec["pass1"]) - float(base["pass1"])
        sign = "+" if d >= 0 else ""
        return (
            f"{label} & {f3(base['pass1'])} & {f3(dec['pass1'])} & "
            f"{sign}{d * 100:.1f}\\,pp \\\\"
        )

    delta_table = "\n        ".join([
        pair_row(r"Gemini Flash-Lite", base_fl_g, dec_fl_g),
        pair_row(r"Qwen 3.5:9B (Ollama)", base_qw_g, dec_qw_g),
    ])

    # Qwen status note
    qw_dec = grab("Declarative", "Golden", QW)
    qw_dec_status = "complete" if (qw_dec and int(qw_dec["n_eval"] or 0) > 0) else "still running / not yet complete"
    qw_base = grab("Baseline", "Golden", QW)
    qw_base_n = int(qw_base["n_eval"] or 0) if qw_base else 0
    qw_dec_n = int(qw_dec["n_eval"] or 0) if qw_dec else 0

    body = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[margin=2.4cm]{{geometry}}
\usepackage{{amsmath}}
\usepackage{{booktabs}}
\usepackage{{tabularx}}
\usepackage{{caption}}
\usepackage{{parskip}}
\usepackage{{microtype}}
\usepackage{{float}}
\usepackage{{hyperref}}

\setlength{{\emergencystretch}}{{3em}}

\title{{Scaling Probe: Declarative Skills on Gemini Flash-Lite \\ and Local Qwen~3.5:9B}}
\author{{Auto-generated from \texttt{{scripts/compute\_scaling\_summary.py}}}}
\date{{{today}}}

\begin{{document}}
\maketitle

\section*{{Setup}}

We re-run the v2 declarative-vs-baseline comparison on two smaller agent LLMs
to test whether natural-language skill files transfer across the
capability-frontier when the underlying model is below the
procedural-competence threshold of DeepSeek~v4-pro:

\begin{{itemize}}
  \item \textbf{{Gemini Flash-Lite}} (\texttt{{gemini/gemini-3.1-flash-lite}}),
        paid tier, max\_tokens=4096, concurrency~20.
  \item \textbf{{Qwen~3.5:9B}} (\texttt{{ollama\_chat/qwen3.5:9b}}), local
        Ollama on macOS, max\_tokens=4096, concurrency~1.
\end{{itemize}}

The user simulator is held constant at \texttt{{deepseek/deepseek-v4-pro}} so
the evaluation oracle matches the existing $\tau$-Banking v2 rows.  Domain:
\texttt{{banking\_knowledge}}, 97 tasks, single trial, seed~300.  Conditions
were launched via \texttt{{scripts/run\_experiment.py}} using the configs
\texttt{{scaling-flash-lite.yaml}} and \texttt{{qwen-local.yaml}}.

Per the project decision recorded on 2026-05-07, BM25 retrieval is out of
scope; we report only \emph{{golden}} and \emph{{embedding}} (\texttt{{all-MiniLM-L6-v2}})
retrieval.

\section*{{Results: Gemini Flash-Lite (paid)}}

\begin{{table}}[H]
\centering
\caption{{Flash-Lite results on $\tau$-Banking v2 (97 tasks, 1~trial).
$N/\text{{Total}}$ counts evaluable simulations after removing
infrastructure errors (LiteLLM rate-limit / 429 prepayment-depleted).
Cost is computed from per-message token usage at published Gemini~2.5
Flash-Lite paid-tier rates (\$0.10/M input, \$0.40/M output, non-cached);
server-side implicit context caching makes this an upper bound.}}
\label{{tab:scaling-flashlite}}
\small
\begin{{tabular}}{{llrrrrrr}}
\toprule
Agent & Retrieval & $N/\text{{Total}}$ & Pass$^{{1}}$ & DB~match & Mean turns & Mean in-tok & Cost/conv \\
\midrule
        {flashlite_table}
\bottomrule
\end{{tabular}}
\end{{table}}

\section*{{Results: Local Qwen~3.5:9B (Ollama)}}

\begin{{table}}[H]
\centering
\caption{{Qwen~3.5:9B on $\tau$-Banking v2 (golden retrieval only;
embedding retrieval was not run because the local model would not finish
in the available compute window).  ``Cost/conv'' is \$0.00 because the
model is served locally; mean input tokens is reported as the comparable
cost-of-compute proxy.  Qwen baseline N=\textbf{{{qw_base_n}}},
declarative N=\textbf{{{qw_dec_n}}} -- declarative-v2 condition is {qw_dec_status}.}}
\label{{tab:scaling-qwen}}
\small
\begin{{tabular}}{{llrrrrrr}}
\toprule
Agent & Retrieval & $N/\text{{Total}}$ & Pass$^{{1}}$ & DB~match & Mean turns & Mean in-tok & Cost/conv \\
\midrule
        {qwen_table}
\bottomrule
\end{{tabular}}
\end{{table}}

\section*{{Paired declarative-vs-baseline delta (golden retrieval)}}

\begin{{table}}[H]
\centering
\caption{{Pass$^{{1}}$ for declarative-v2 against the matching baseline agent,
holding retrieval = golden constant.  ``pp'' = percentage points.}}
\label{{tab:scaling-delta}}
\small
\begin{{tabular}}{{lrrr}}
\toprule
Agent LLM & Baseline Pass$^{{1}}$ & Declarative Pass$^{{1}}$ & $\Delta$ \\
\midrule
        {delta_table}
\bottomrule
\end{{tabular}}
\end{{table}}

\section*{{Notes}}

\begin{{itemize}}

  \item \textbf{{Flash-Lite golden:}} declarative-v2 matches baseline on
        Pass$^{{1}}$ (both \textbf{{{f3(dec_fl_g['pass1']) if dec_fl_g else '--'}}}, $\Delta = {delta_str(fl_pair_delta)}$);
        the skill files do not lift task-completion accuracy on this model
        size despite costing roughly $1.3\times$ more input tokens.

  \item \textbf{{Flash-Lite embedding:}} declarative-v2 doubles Pass$^{{1}}$
        relative to the same-retrieval baseline (0.104 vs 0.052), and DB~match
        rises from 8.3\% to 13.5\%.  This is the only flashlite condition
        where the skills appear to compensate for noisy retrieval, though
        absolute scores remain low.

  \item \textbf{{Flash-Lite imperative-embedding:}} only \textbf{{6}} sims
        completed -- the run was throttled by the Gemini paid-tier TPM cap
        and then halted entirely by a billing-depleted retry.  The row is
        kept in the table so it is not silently dropped, but no claim is
        defensible at $N=6$.

  \item \textbf{{Qwen~3.5:9B:}} reported here without the API-cost column
        being misleading.  Local inference takes $\sim 70$--90~s/task at
        concurrency~1 on this hardware, so embedding retrieval was skipped
        in this report to keep wall-clock budget on the golden comparison.

  \item \textbf{{Reproducibility:}} numbers in the tables are written by
        \texttt{{scripts/compute\_scaling\_summary.py}}; the CSV is at
        \texttt{{results/scaling\_flashlite\_qwen.csv}}.  Re-run that
        script (then this generator) after any further sims complete.

\end{{itemize}}

\end{{document}}
"""

    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text(body)
    print(f"Wrote {TEX_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
