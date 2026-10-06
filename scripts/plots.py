import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (  # noqa: E402
    ALPHA, CONTEXT_TRANSITIONS, DATASET_LABELS, FIGURES_DIR, METRIC_LABELS, METRICS, MODEL_LABELS, TABLES_DIR,
    release_transitions,
)

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#52514e",
    "axes.linewidth": 0.6,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "lines.linewidth": 1.2,
})

INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"  # all 8 / all datasets
ORANGE = "#eb6834"  # strict 3 / ablation
GRAY = "#8a8984"  # context comparisons
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#e8e8e6", "#e34948"])
FULL_WIDTH = 7.0  # inches

METRIC_NOTE = {"sentiment_score": "higher = more positive",
               "toxicity_score": "lower = less toxic", "stereotype_score": "lower = fewer stereotype cues"}


def table(name):
    path = TABLES_DIR / name
    if not path.is_file():
        sys.exit(f"Missing {path}; run scripts/reproduce.py first.")
    return pd.read_csv(path)


def transition_label(old_model, new_model):
    return f"{MODEL_LABELS[old_model]} → {MODEL_LABELS[new_model]}"


def style_axis(ax, zero=True):
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    if zero:
        ax.axvline(0, color=MUTED, linewidth=0.7)


def save(fig, out_dir, stem):
    path = out_dir / f"{stem}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}")


def fig_aggregate(out_dir):
    agg = table("aggregate_drift.csv")
    is_all = agg["aggregate_set"] == "all_release_drift_8_transitions"
    agg = agg.assign(label=np.where(is_all, "All 8", "Strict 3"), color=np.where(is_all, BLUE, ORANGE),
                     marker=np.where(is_all, "o", "s"))
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 1.5))
    for ax, metric in zip(axes, METRICS):
        sub = agg[agg["metric"] == metric].reset_index(drop=True)
        for y, row in sub.iterrows():
            ax.errorbar(row["mean_drift"], y, xerr=[[row["mean_drift"] - row["ci_lower"]],
                                                    [row["ci_upper"] - row["mean_drift"]]],
                        fmt=row["marker"], color=row["color"], markersize=5, capsize=2, elinewidth=1.2)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["label"] if metric == METRICS[0] else [])
        ax.set_ylim(-0.6, len(sub) - 0.4)
        ax.invert_yaxis()
        ax.set_title(f"{METRIC_LABELS[metric]} ({METRIC_NOTE[metric]})")
        ax.set_xlabel("Mean paired drift (95% CI)")
        style_axis(ax)
    fig.tight_layout()
    save(fig, out_dir, "fig1_aggregate_drift")


def fig_transition_drift(out_dir):
    release = table("transition_drift.csv")
    context = table("context_drift.csv")
    order = [(old, new, True) for _, old, new in release_transitions()]
    order += [(old, new, False) for _, old, new in CONTEXT_TRANSITIONS]
    labels = [transition_label(old, new) for old, new, _ in order]

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 3.4), sharey=True)
    for ax, metric in zip(axes, METRICS):
        for y, (old, new, is_release) in enumerate(order):
            src = release if is_release else context
            row = src[(src["model_t"] == old) & (src["model_t1"] == new) & (src["metric"] == metric)].iloc[0]
            significant = is_release and row["q_value"] < ALPHA
            color = BLUE if is_release else GRAY
            ax.errorbar(row["drift"], y, xerr=[[row["drift"] - row["ci_lower"]], [row["ci_upper"] - row["drift"]]],
                        fmt="o", color=color, markersize=4.5, capsize=1.5, elinewidth=1.0,
                        markerfacecolor=color if significant else "white", markeredgewidth=1.0)
        ax.axhline(len(list(release_transitions())) - 0.5, color=MUTED, linewidth=0.5, linestyle=":")
        ax.set_title(f"{METRIC_LABELS[metric]} ({METRIC_NOTE[metric]})")
        ax.set_xlabel("Pooled paired drift (95% CI)")
        style_axis(ax)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(labels)
    axes[0].set_ylim(-0.6, len(order) - 0.4)
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([], [], marker="o", color=BLUE, linestyle="", markersize=4.5,
                   label="Release, BH q < 0.05 (24 tests)"),
        plt.Line2D([], [], marker="o", color=BLUE, markerfacecolor="white", linestyle="", markersize=4.5,
                   label="Release, not significant"),
        plt.Line2D([], [], marker="o", color=GRAY, markerfacecolor="white", linestyle="", markersize=4.5,
                   label="Cross-tier / scale (descriptive)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, out_dir, "fig2_transition_drift")


def fig_benchmark_effects(out_dir):
    cells = table("benchmark_drift.csv")
    transitions = [(old, new) for _, old, new in release_transitions()]
    datasets = sorted(cells["dataset"].unique())
    limit = float(np.ceil(cells["cohens_d"].abs().max() * 10) / 10)

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.9), sharey=True)
    for ax, metric in zip(axes, METRICS):
        grid = np.full((len(transitions), len(datasets)), np.nan)
        stars = np.zeros_like(grid, dtype=bool)
        sub = cells[cells["metric"] == metric]
        for _, row in sub.iterrows():
            i = transitions.index((row["model_t"], row["model_t1"]))
            j = datasets.index(row["dataset"])
            grid[i, j] = row["cohens_d"]
            stars[i, j] = row["q_value"] < ALPHA
        image = ax.imshow(grid, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
        for i in range(len(transitions)):
            for j in range(len(datasets)):
                if np.isnan(grid[i, j]):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=5.5, color=MUTED)
                elif stars[i, j]:
                    ax.text(j, i, "*", ha="center", va="center", fontsize=9, color=INK)
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels([DATASET_LABELS[d] for d in datasets], rotation=45, ha="right")
        ax.set_title(METRIC_LABELS[metric])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(transitions)))
    axes[0].set_yticklabels([transition_label(*t) for t in transitions])
    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Paired Cohen's d")
    cbar.outline.set_visible(False)
    save(fig, out_dir, "fig3_benchmark_effects")


def fig_correlations(out_dir):
    corr = table("correlations.csv")
    pairs = list(dict.fromkeys(zip(corr["metric_1"], corr["metric_2"])))
    columns = [("per_example", "all_7"), ("model_benchmark", "all_7"),
               ("per_example", "ablation_5"), ("model_benchmark", "ablation_5")]
    grid = np.array([[corr[(corr["metric_1"] == a) & (corr["metric_2"] == b) & (corr["analysis_level"] == lvl)
                           & (corr["benchmark_set"] == bset)]["pearson_r"].iloc[0] for lvl, bset in columns]
                     for a, b in pairs])
    n = {(lvl, bset): int(corr[(corr["analysis_level"] == lvl) & (corr["benchmark_set"] == bset)]["n"].iloc[0])
         for lvl, bset in columns}

    fig, ax = plt.subplots(figsize=(3.4, 1.7))
    ax.imshow(grid, cmap=DIVERGING, vmin=-0.5, vmax=0.5, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:+.2f}", ha="center", va="center", fontsize=7, color=INK)
    level_label = {"per_example": "Per example", "model_benchmark": "Model–bench."}
    set_label = {"all_7": "7 bench.", "ablation_5": "5 bench."}
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([f"{level_label[lvl]}\n{set_label[bset]}\nn={n[(lvl, bset)]:,}" for lvl, bset in columns])
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"{METRIC_LABELS[a]}–{METRIC_LABELS[b]}" for a, b in pairs])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    save(fig, out_dir, "fig4_metric_correlations")


def fig_ablation(out_dir):
    comp = table("ablation.csv")
    rows = [("aggregate", "all_8", None, None, "Aggregate, all 8"),
            ("aggregate", "strict_3", None, None, "Aggregate, strict 3")]
    rows += [("transition_pooled", "all_8", old, new, transition_label(old, new))
             for _, old, new in release_transitions()]

    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.8), sharey=True)
    for ax, metric in zip(axes, METRICS):
        sub = comp[comp["metric"] == metric]
        for y, (level, tset, old, new, _) in enumerate(rows):
            match = sub[(sub["analysis_level"] == level) & (sub["transition_set"] == tset)]
            if old is not None:
                match = match[(match["model_t"] == old) & (match["model_t1"] == new)]
            row = match.iloc[0]
            for prefix, color, marker, offset in [("full", BLUE, "o", -0.17), ("ablation", ORANGE, "s", 0.17)]:
                d, lo, hi = row[f"{prefix}_drift"], row[f"{prefix}_ci_lower"], row[f"{prefix}_ci_upper"]
                ax.errorbar(d, y + offset, xerr=[[d - lo], [hi - d]], fmt=marker, color=color,
                            markersize=3.5, capsize=1.2, elinewidth=0.9)
        ax.axhline(1.5, color=MUTED, linewidth=0.5, linestyle=":")
        ax.set_title(f"{METRIC_LABELS[metric]} ({METRIC_NOTE[metric]})")
        ax.set_xlabel("Paired drift (95% CI)")
        style_axis(ax)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[-1] for r in rows])
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], marker="o", color=BLUE, linestyle="", markersize=4, label="All 7 benchmarks"),
               plt.Line2D([], [], marker="s", color=ORANGE, linestyle="", markersize=4,
                          label="Ablation: without StereoSet and CrowS-Pairs")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save(fig, out_dir, "fig5_ablation_comparison")


def main(argv):
    out_dir = Path(argv[argv.index("--out") + 1]) if "--out" in argv else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for make in (fig_aggregate, fig_transition_drift, fig_benchmark_effects, fig_correlations, fig_ablation):
        make(out_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
