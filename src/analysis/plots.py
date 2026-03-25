#!/usr/bin/env python3
"""Generate all experiment figures from CSV results.

Reads sim_level.csv, condition_summary.csv, and task_pass_rates.csv
from --results-dir (default: PROJECT_ROOT/results) and writes six PNG
figures to results/figures/.
"""

import argparse
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# -- Style constants ----------------------------------------------------------

AGENT_PALETTE = {
    "baseline": "#888888",
    "declarative": "#4878CF",
    "imperative": "#E8792B",
}
AGENT_ORDER = ["baseline", "declarative", "imperative"]
AGENT_LABELS = {"baseline": "Baseline", "declarative": "Declarative", "imperative": "Imperative"}
RETRIEVAL_ORDER = ["golden_retrieval", "bm25"]
RETRIEVAL_LABELS = {"golden_retrieval": "Golden Retrieval", "bm25": "BM25"}

DPI = 300
FIG_WIDTH = 7  # inches, double-column
LABEL_SIZE = 11
TICK_SIZE = 9
TITLE_SIZE = 12


def _apply_style():
    """Set global matplotlib/seaborn style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.size": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": TICK_SIZE,
        "figure.dpi": DPI,
    })


# -- Bootstrap CI helper ------------------------------------------------------

def _pass_hat_k(n_successes: int, n_trials: int, k: int) -> float:
    """Compute Pass^k for a single task: C(n_successes, k) / C(n_trials, k)."""
    if n_trials < k or n_successes < 0:
        return 0.0
    return comb(n_successes, k) / comb(n_trials, k)


def _bootstrap_pass_k(
    sim_df: pd.DataFrame,
    agent_type: str,
    retrieval: str,
    k: int,
    n_iter: int = 10_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Cluster-bootstrap 95% CI for Pass^k over tasks.

    Returns (point_estimate, ci_low, ci_high).
    """
    cond = sim_df[(sim_df["agent_type"] == agent_type) & (sim_df["retrieval"] == retrieval)]
    task_stats = (
        cond.groupby("task_id")
        .agg(n_trials=("success", "size"), n_successes=("success", "sum"))
        .reset_index()
    )
    task_ids = task_stats["task_id"].values
    n_tasks = len(task_ids)
    if n_tasks == 0:
        return 0.0, 0.0, 0.0

    # Point estimate
    task_stats["pass_k"] = task_stats.apply(
        lambda r: _pass_hat_k(int(r["n_successes"]), int(r["n_trials"]), k), axis=1,
    )
    point = task_stats["pass_k"].mean()

    # Bootstrap
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_iter)
    pass_k_arr = task_stats["pass_k"].values
    for i in range(n_iter):
        idx = rng.integers(0, n_tasks, size=n_tasks)
        boot_means[i] = pass_k_arr[idx].mean()

    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return point, ci_low, ci_high


# -- Figure builders ----------------------------------------------------------

def _save(fig: plt.Figure, path: Path, name: str):
    """Save figure, print path, close."""
    out = path / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"  saved: {out}")
    plt.close(fig)


def plot_pass_k_bars(
    sim_df: pd.DataFrame,
    k: int,
    out_dir: Path,
    filename: str,
    title: str,
):
    """Grouped bar chart of Pass^k with bootstrap 95% CI."""
    rows = []
    for agent in AGENT_ORDER:
        for ret in RETRIEVAL_ORDER:
            point, ci_lo, ci_hi = _bootstrap_pass_k(sim_df, agent, ret, k)
            rows.append({
                "agent_type": AGENT_LABELS[agent],
                "retrieval": RETRIEVAL_LABELS[ret],
                "pass_k": point,
                "ci_lo": point - ci_lo,
                "ci_hi": ci_hi - point,
            })
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4))

    x = np.arange(len(AGENT_ORDER))
    width = 0.35
    for i, ret in enumerate(RETRIEVAL_ORDER):
        sub = plot_df[plot_df["retrieval"] == RETRIEVAL_LABELS[ret]]
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            sub["pass_k"],
            width,
            yerr=[sub["ci_lo"].values, sub["ci_hi"].values],
            label=RETRIEVAL_LABELS[ret],
            color=[AGENT_PALETTE[a] for a in AGENT_ORDER],
            edgecolor="white",
            linewidth=0.5,
            capsize=3,
            alpha=0.85 if i == 0 else 0.55,
        )
        # Annotate bar values
        for bar, val in zip(bars, sub["pass_k"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABELS[a] for a in AGENT_ORDER])
    ax.set_ylabel(f"Pass^{k}")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.set_title(title)
    ax.legend(title="Retrieval")
    sns.despine(left=True)

    _save(fig, out_dir, filename)


def plot_task_heatmap(task_df: pd.DataFrame, out_dir: Path):
    """Heatmap of per-task pass rates across 6 conditions."""
    col_order = [
        "baseline_golden", "declarative_golden", "imperative_golden",
        "baseline_bm25", "declarative_bm25", "imperative_bm25",
    ]
    col_labels = [
        "Baseline\nGolden", "Declarative\nGolden", "Imperative\nGolden",
        "Baseline\nBM25", "Declarative\nBM25", "Imperative\nBM25",
    ]

    # Sort by mean pass rate, highest at top
    df = task_df.sort_values("mean_pass_rate", ascending=True).copy()
    matrix = df[col_order].values
    task_labels = df["task_id"].values

    n_tasks = len(task_labels)
    fig_height = max(12, n_tasks * 0.25)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Discrete colormap: 0=red, 0.33=orange, 0.67=light green, 1.0=green
    colors = ["#d9534f", "#f0ad4e", "#a8d97f", "#5cb85c"]
    cmap = ListedColormap(colors)
    bounds = [0, 0.165, 0.5, 0.835, 1.01]
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    # Annotate cells
    frac_labels = {0.0: "0", 1 / 3: "\u2153", 2 / 3: "\u2154", 1.0: "1"}
    for i in range(n_tasks):
        for j in range(len(col_order)):
            val = matrix[i, j]
            # Find closest fraction label
            closest = min(frac_labels.keys(), key=lambda fv: abs(fv - val))
            label = frac_labels[closest]
            text_color = "white" if val < 0.5 else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=6, color=text_color)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(task_labels, fontsize=5)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Task ID")
    ax.set_title("Per-Task Pass Rate Across Conditions")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, ticks=[0, 0.33, 0.67, 1.0])
    cbar.ax.set_yticklabels(["0", "⅓", "⅔", "1"])
    cbar.set_label("Pass Rate")

    _save(fig, out_dir, "task_heatmap.png")


def plot_tool_calls_boxplot(sim_df: pd.DataFrame, out_dir: Path):
    """Box plot of tool calls per successful task."""
    success_df = sim_df[sim_df["success"] == True].copy()  # noqa: E712
    success_df["retrieval_label"] = success_df["retrieval"].map(RETRIEVAL_LABELS)
    success_df["agent_label"] = success_df["agent_type"].map(AGENT_LABELS)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4))

    # Check for low-count conditions
    counts = success_df.groupby(["agent_type", "retrieval"]).size()
    low_conditions = counts[counts < 10]
    subtitle = ""
    if len(low_conditions) > 0:
        names = [f"{AGENT_LABELS[a]}/{RETRIEVAL_LABELS[r]}" for a, r in low_conditions.index]
        subtitle = f"Note: <10 successes in {', '.join(names)}"

    sns.boxplot(
        data=success_df,
        x="retrieval_label",
        y="num_tool_calls",
        hue="agent_label",
        hue_order=[AGENT_LABELS[a] for a in AGENT_ORDER],
        palette=[AGENT_PALETTE[a] for a in AGENT_ORDER],
        order=[RETRIEVAL_LABELS[r] for r in RETRIEVAL_ORDER],
        ax=ax,
        fliersize=2,
        linewidth=0.8,
    )

    ax.set_xlabel("Retrieval Config")
    ax.set_ylabel("Tool Calls")
    ax.set_title("Tool Calls per Successful Task")
    if subtitle:
        ax.text(
            0.5, -0.12, subtitle, transform=ax.transAxes,
            ha="center", fontsize=8, style="italic", color="#666666",
        )
    ax.legend(title="Agent Type")
    sns.despine(left=True)

    _save(fig, out_dir, "tool_calls_boxplot.png")


def plot_turns_boxplot(sim_df: pd.DataFrame, out_dir: Path):
    """Box plot of conversation turns (all simulations)."""
    df = sim_df.copy()
    df["retrieval_label"] = df["retrieval"].map(RETRIEVAL_LABELS)
    df["agent_label"] = df["agent_type"].map(AGENT_LABELS)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4))

    sns.boxplot(
        data=df,
        x="retrieval_label",
        y="num_turns",
        hue="agent_label",
        hue_order=[AGENT_LABELS[a] for a in AGENT_ORDER],
        palette=[AGENT_PALETTE[a] for a in AGENT_ORDER],
        order=[RETRIEVAL_LABELS[r] for r in RETRIEVAL_ORDER],
        ax=ax,
        fliersize=2,
        linewidth=0.8,
    )

    ax.set_xlabel("Retrieval Config")
    ax.set_ylabel("Number of Turns")
    ax.set_title("Conversation Turns by Condition")
    ax.legend(title="Agent Type")
    sns.despine(left=True)

    _save(fig, out_dir, "turns_boxplot.png")


def plot_termination_breakdown(cond_df: pd.DataFrame, out_dir: Path):
    """Stacked bar chart of termination reason proportions."""
    # Build condition labels in consistent order
    condition_order = []
    for agent in AGENT_ORDER:
        for ret in RETRIEVAL_ORDER:
            condition_order.append(f"{agent}_{ret}")

    reason_cols = ["pct_user_stop", "pct_agent_stop", "pct_max_steps", "pct_error"]
    reason_labels = ["User Stop", "Agent Stop", "Max Steps", "Error"]
    reason_colors = ["#5cb85c", "#4878CF", "#f0ad4e", "#d9534f"]

    # Map conditions to display labels
    def _cond_label(cond_name: str) -> str:
        parts = cond_name.split("_", 1)
        agent = parts[0] if len(parts) > 0 else cond_name
        ret = parts[1] if len(parts) > 1 else ""
        return f"{AGENT_LABELS.get(agent, agent)}\n{RETRIEVAL_LABELS.get(ret, ret)}"

    # Build the plotting dataframe — match on agent_type + retrieval
    rows = []
    for cond in condition_order:
        agent, ret = cond.split("_", 1)
        match = cond_df[
            (cond_df["agent_type"] == agent) & (cond_df["retrieval"] == ret)
        ]
        if match.empty:
            continue
        row = match.iloc[0]
        pct_other = max(0, 1.0 - sum(row.get(c, 0) for c in reason_cols))
        rows.append({
            "condition": cond,
            "label": _cond_label(cond),
            "User Stop": row.get("pct_user_stop", 0),
            "Agent Stop": row.get("pct_agent_stop", 0),
            "Max Steps": row.get("pct_max_steps", 0),
            "Error": row.get("pct_error", 0),
            "Other": pct_other,
        })
    plot_df = pd.DataFrame(rows)

    all_reasons = reason_labels + ["Other"]
    all_colors = reason_colors + ["#cccccc"]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4.5))

    x = np.arange(len(plot_df))
    bottom = np.zeros(len(plot_df))
    for reason, color in zip(all_reasons, all_colors):
        vals = plot_df[reason].values.astype(float)
        ax.bar(x, vals, bottom=bottom, label=reason, color=color, edgecolor="white", linewidth=0.5)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], fontsize=8)
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.set_title("Termination Reason Breakdown")
    ax.legend(loc="upper right", fontsize=8)
    sns.despine(left=True)

    _save(fig, out_dir, "termination_breakdown.png")


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate experiment figures.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory containing CSV result files (default: PROJECT_ROOT/results)",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _apply_style()

    # Load data
    sim_df = pd.read_csv(results_dir / "sim_level.csv")
    cond_df = pd.read_csv(results_dir / "condition_summary.csv")
    task_df = pd.read_csv(results_dir / "task_pass_rates.csv")

    print(f"Loaded {len(sim_df)} simulations, {len(cond_df)} conditions, {len(task_df)} tasks")
    print(f"Saving figures to {fig_dir}/\n")

    # 1. Pass^1 bar chart
    plot_pass_k_bars(sim_df, k=1, out_dir=fig_dir, filename="pass1_by_condition.png",
                     title="Pass^1 by Agent Type and Retrieval Config")

    # 2. Pass^3 bar chart
    plot_pass_k_bars(sim_df, k=3, out_dir=fig_dir, filename="pass3_by_condition.png",
                     title="Pass^3 by Agent Type and Retrieval Config")

    # 3. Task heatmap
    plot_task_heatmap(task_df, fig_dir)

    # 4. Tool calls box plot (successes only)
    plot_tool_calls_boxplot(sim_df, fig_dir)

    # 5. Turns box plot (all sims)
    plot_turns_boxplot(sim_df, fig_dir)

    # 6. Termination breakdown
    plot_termination_breakdown(cond_df, fig_dir)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
