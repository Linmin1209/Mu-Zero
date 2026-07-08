#!/usr/bin/env python3
"""RoboCasa365 task language semantic analysis + visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(
    "/XYAIFS00/HDD_POOL/sysu_xdliang/sysu_xdliang_1/linmin/datasets/robocasa365-datasets"
)
DEFAULT_XLSX = Path(__file__).resolve().parents[3] / "Robocasa365.xlsx"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "analysis" / "task_semantics"


def load_corpus(dataset_root: Path) -> pd.DataFrame:
    records = []
    for tasks_path in dataset_root.rglob("meta/tasks.jsonl"):
        parts = tasks_path.parts
        split_idx = parts.index("robocasa365-datasets") + 1
        split, category, task_name = parts[split_idx : split_idx + 3]
        with open(tasks_path) as f:
            for line in f:
                row = json.loads(line)
                text = str(row.get("task", "")).strip()
                if text:
                    records.append(
                        {
                            "split": split,
                            "category": category,
                            "task_name": task_name,
                            "text": text,
                            "word_count": len(text.split()),
                        }
                    )
    return pd.DataFrame(records)


def build_per_task(df: pd.DataFrame) -> pd.DataFrame:
    per_task = (
        df.groupby("task_name")
        .agg(
            texts=("text", lambda s: sorted(set(s))),
            category=("category", "first"),
            split=("split", "first"),
        )
        .reset_index()
    )
    per_task["n_variants"] = per_task["texts"].apply(len)
    per_task["primary"] = per_task["texts"].apply(
        lambda xs: max(xs, key=lambda t: len(t.split()))
    )
    per_task["primary_word_count"] = per_task["primary"].str.split().str.len()
    per_task["concat"] = per_task["texts"].apply(lambda xs: " ".join(xs))
    per_task["family"] = per_task["task_name"].apply(_task_family)
    return per_task


def _task_family(name: str) -> str:
    if name.startswith("PickPlace"):
        return "PickPlace"
    if name.startswith("Open") or name.startswith("Close"):
        return "OpenClose"
    if name.startswith("Turn"):
        return "Turn"
    if name.startswith("Navigate"):
        return "Navigate"
    return "Other"


def load_eval_xlsx(xlsx_path: Path) -> pd.DataFrame:
    xlsx = pd.read_excel(xlsx_path, sheet_name="Robocasa365")
    xlsx = xlsx[xlsx["Task"].notna() & (xlsx["Task"] != "---")].copy()
    for col in [
        "b128_motionckpt-120k",
        "b128_gated_motionckpt-120k",
        "b128_vanillackpt-120k",
    ]:
        xlsx[col] = pd.to_numeric(xlsx[col], errors="coerce")
    xlsx["motion"] = xlsx["b128_motionckpt-120k"]
    xlsx["gated"] = xlsx["b128_gated_motionckpt-120k"]
    xlsx["vanilla"] = xlsx["b128_vanillackpt-120k"]
    xlsx["gate_delta"] = xlsx["gated"] - xlsx["motion"]
    return xlsx


def tfidf_similarity(texts: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=1)
    x = vec.fit_transform(texts)
    return cosine_similarity(x)


def pairwise_offdiag(sim: np.ndarray) -> np.ndarray:
    n = sim.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return sim[mask]


def intra_task_similarity(variants: list[str]) -> float:
    if len(variants) < 2:
        return 0.0
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    v = vec.fit_transform(variants)
    s = cosine_similarity(v)
    return float(pairwise_offdiag(s).mean())


def attach_eval_features(eval_df: pd.DataFrame, per_task: pd.DataFrame, sim: np.ndarray) -> pd.DataFrame:
    name2idx = {n: i for i, n in enumerate(per_task["task_name"])}
    pt = per_task.set_index("task_name")
    out = eval_df.merge(
        pt[["primary", "n_variants", "texts", "family", "primary_word_count"]],
        left_on="Task",
        right_index=True,
        how="inner",
    )
    eval_tasks = out["Task"].tolist()
    max_sim, mean_sim, intra = [], [], []
    for t in eval_tasks:
        i = name2idx[t]
        others = [name2idx[o] for o in eval_tasks if o != t and o in name2idx]
        if others:
            s = sim[i, others]
            max_sim.append(float(s.max()))
            mean_sim.append(float(s.mean()))
        else:
            max_sim.append(0.0)
            mean_sim.append(0.0)
        intra.append(intra_task_similarity(out.loc[out["Task"] == t, "texts"].iloc[0]))
    out["max_sim_eval"] = max_sim
    out["mean_sim_eval"] = mean_sim
    out["intra_task_sim"] = intra
    return out


def plot_similarity_histogram(all_sim: np.ndarray, eval_sim: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 41)
    ax.hist(
        all_sim,
        bins=bins,
        alpha=0.55,
        label=f"All tasks (n={len(all_sim):,})",
        color="#4C72B0",
        edgecolor="white",
    )
    ax.hist(
        eval_sim,
        bins=bins,
        alpha=0.75,
        label=f"Eval 50 tasks (n={len(eval_sim):,})",
        color="#DD8452",
        edgecolor="white",
    )
    ax.axvline(np.median(all_sim), color="#4C72B0", ls="--", lw=1.2, alpha=0.9)
    ax.axvline(np.median(eval_sim), color="#DD8452", ls="--", lw=1.2, alpha=0.9)
    ax.set_xlabel("TF-IDF cosine similarity")
    ax.set_ylabel("Pair count")
    ax.set_title("RoboCasa365 inter-task language similarity distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "01_similarity_histogram.png", dpi=160)
    plt.close(fig)


def plot_family_boxplot(per_task: pd.DataFrame, sim: np.ndarray, out_dir: Path) -> None:
    name2idx = {n: i for i, n in enumerate(per_task["task_name"])}
    rows = []
    for fam in per_task["family"].unique():
        idx = [name2idx[n] for n in per_task.loc[per_task["family"] == fam, "task_name"]]
        if len(idx) < 2:
            continue
        sub = sim[np.ix_(idx, idx)]
        for v in pairwise_offdiag(sub):
            rows.append({"family": fam, "similarity": v})
    fam_df = pd.DataFrame(rows)
    order = (
        fam_df.groupby("family")["similarity"].median().sort_values(ascending=False).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=fam_df, x="family", y="similarity", order=order, ax=ax, palette="Set2")
    sns.stripplot(
        data=fam_df,
        x="family",
        y="similarity",
        order=order,
        ax=ax,
        color="0.25",
        alpha=0.15,
        size=2,
        jitter=0.25,
    )
    ax.set_title("Intra-family vs inter-family language similarity")
    ax.set_xlabel("Task family (by name prefix)")
    ax.set_ylabel("Pairwise TF-IDF cosine similarity")
    fig.tight_layout()
    fig.savefig(out_dir / "02_family_similarity_boxplot.png", dpi=160)
    plt.close(fig)


def plot_word_count_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["word_count"], bins=30, ax=axes[0], color="#55A868")
    axes[0].set_title("All description variants")
    axes[0].set_xlabel("Word count")
    axes[0].set_ylabel("Count")
    axes[0].set_xlim(0, 40)

    vc = df["word_count"].value_counts().sort_index()
    top = vc.head(8)
    axes[1].bar(top.index.astype(str), top.values, color="#C44E52")
    axes[1].set_title("Most common word-count buckets")
    axes[1].set_xlabel("Word count")
    axes[1].set_ylabel("Count")
    fig.suptitle("RoboCasa365 language instruction length", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "03_word_count_distribution.png", dpi=160)
    plt.close(fig)


def plot_eval_heatmap(eval_df: pd.DataFrame, sim: np.ndarray, per_task: pd.DataFrame, out_dir: Path) -> None:
    tasks = eval_df["Task"].tolist()
    name2idx = {n: i for i, n in enumerate(per_task["task_name"])}
    idx = [name2idx[t] for t in tasks]
    sub = sim[np.ix_(idx, idx)]
    cat = eval_df.set_index("Task").loc[tasks, "Category"]

    # cluster order by category then name
    order = (
        eval_df.sort_values(["Category", "Task"])["Task"].tolist()
    )
    oidx = [tasks.index(t) for t in order]
    sub = sub[np.ix_(oidx, oidx)]

    fig_h = max(10, len(order) * 0.22)
    fig, ax = plt.subplots(figsize=(fig_h, fig_h))
    sns.heatmap(
        sub,
        xticklabels=order,
        yticklabels=order,
        cmap="YlOrRd",
        vmin=0,
        vmax=0.8,
        square=True,
        cbar_kws={"label": "cosine sim"},
        ax=ax,
    )
    ax.set_title("Eval 50 tasks — TF-IDF semantic similarity")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "04_eval_heatmap.png", dpi=160)
    plt.close(fig)

    # save category-annotated clustermap (smaller subset labels)
    cg = sns.clustermap(
        sub,
        cmap="YlOrRd",
        vmin=0,
        vmax=0.8,
        figsize=(14, 14),
        xticklabels=[t[:18] for t in order],
        yticklabels=[t[:18] for t in order],
        dendrogram_ratio=0.08,
        cbar_pos=(0.02, 0.8, 0.02, 0.15),
    )
    cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6)
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_yticklabels(), fontsize=6)
    cg.fig.suptitle("Eval tasks clustered by language similarity", y=1.01)
    cg.savefig(out_dir / "05_eval_clustermap.png", dpi=160)
    plt.close(cg.fig)


def plot_embedding_2d(per_task: pd.DataFrame, eval_df: pd.DataFrame, out_dir: Path) -> None:
    texts = per_task["primary"].tolist()
    names = per_task["task_name"].tolist()
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=1)
    x = vec.fit_transform(texts).toarray()

    # PCA for init, t-SNE for layout
    x_pca = PCA(n_components=min(50, x.shape[1]), random_state=0).fit_transform(x)
    perplexity = min(30, max(5, len(names) // 10))
    xy = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=0).fit_transform(
        x_pca
    )

    plot_df = per_task.copy()
    plot_df["x"] = xy[:, 0]
    plot_df["y"] = xy[:, 1]
    plot_df["is_eval"] = plot_df["task_name"].isin(set(eval_df["Task"]))

    fig, ax = plt.subplots(figsize=(11, 8))
    base = plot_df[~plot_df["is_eval"]]
    ev = plot_df[plot_df["is_eval"]]
    sns.scatterplot(
        data=base,
        x="x",
        y="y",
        hue="family",
        style="family",
        palette="tab10",
        alpha=0.45,
        s=35,
        ax=ax,
        legend="brief",
    )
    ax.scatter(
        ev["x"],
        ev["y"],
        s=90,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        label="eval task",
        zorder=5,
    )
    for _, row in ev.iterrows():
        ax.annotate(
            row["task_name"][:16],
            (row["x"], row["y"]),
            fontsize=5,
            alpha=0.85,
            xytext=(2, 2),
            textcoords="offset points",
        )
    ax.set_title("t-SNE of task primary descriptions (TF-IDF)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "06_tsne_all_tasks.png", dpi=160)
    plt.close(fig)


def plot_gate_vs_semantics(eval_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    sns.scatterplot(
        data=eval_df,
        x="max_sim_eval",
        y="gate_delta",
        hue="Category",
        style="family",
        ax=axes[0],
        palette="Set1",
        s=70,
    )
    axes[0].axhline(0, color="0.4", ls="--", lw=1)
    axes[0].set_xlabel("Max sim to other eval tasks")
    axes[0].set_ylabel("gate_delta (gated - motion)")
    axes[0].set_title("Gate effect vs cross-task similarity")

    sns.scatterplot(
        data=eval_df,
        x="primary_word_count",
        y="motion",
        hue="Category",
        ax=axes[1],
        palette="Set1",
        s=70,
    )
    axes[1].set_xlabel("Primary description word count")
    axes[1].set_ylabel("motion success rate")
    axes[1].set_title("Motion perf vs instruction length")

    sns.scatterplot(
        data=eval_df,
        x="intra_task_sim",
        y="n_variants",
        hue="gate_delta",
        size="motion",
        sizes=(40, 200),
        ax=axes[2],
        palette="coolwarm",
        hue_norm=plt.Normalize(eval_df["gate_delta"].min(), eval_df["gate_delta"].max()),
    )
    axes[2].set_xlabel("Intra-task variant similarity")
    axes[2].set_ylabel("# description variants")
    axes[2].set_title("Template diversity vs gate delta (color)")

    fig.tight_layout()
    fig.savefig(out_dir / "07_gate_vs_semantics.png", dpi=160)
    plt.close(fig)


def plot_top_pairs(per_task: pd.DataFrame, sim: np.ndarray, out_dir: Path, top_k: int = 15) -> None:
    names = per_task["task_name"].tolist()
    texts = per_task["primary"].tolist()
    pairs = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((sim[i, j], names[i], names[j], texts[i], texts[j]))
    pairs.sort(reverse=True)
    top = pairs[:top_k]

    fig, ax = plt.subplots(figsize=(10, 0.45 * top_k + 1.5))
    labels = [f"{a} ↔ {b}" for _, a, b, _, _ in top]
    vals = [p[0] for p in top]
    y = np.arange(len(top))
    ax.barh(y, vals, color=sns.color_palette("rocket", len(top)))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("TF-IDF cosine similarity")
    ax.set_title(f"Top-{top_k} most similar task pairs (primary text)")
    fig.tight_layout()
    fig.savefig(out_dir / "08_top_similar_pairs.png", dpi=160)
    plt.close(fig)


def write_summary(
    out_dir: Path,
    per_task: pd.DataFrame,
    df: pd.DataFrame,
    all_sim: np.ndarray,
    eval_sim: np.ndarray,
    eval_df: pd.DataFrame,
) -> None:
    lines = [
        "# RoboCasa365 Task Semantic Analysis",
        "",
        "## Corpus stats",
        f"- Unique tasks: {len(per_task)}",
        f"- Total description variants: {len(df)}",
        f"- Mean variants per task: {per_task['n_variants'].mean():.1f} (median {per_task['n_variants'].median():.0f})",
        f"- 1-word descriptions (task name only): {(df['word_count']==1).sum()} / {len(df)} ({100*(df['word_count']==1).mean():.1f}%)",
        "",
        "## Inter-task similarity (TF-IDF, primary text)",
        f"- All tasks: mean={all_sim.mean():.4f}, median={np.median(all_sim):.4f}, p90={np.percentile(all_sim,90):.4f}, max={all_sim.max():.4f}",
        f"- Eval 50:   mean={eval_sim.mean():.4f}, median={np.median(eval_sim):.4f}, p90={np.percentile(eval_sim,90):.4f}, max={eval_sim.max():.4f}",
        "",
        "## Gate vs semantics (eval)",
        f"- corr(gate_delta, max_sim_eval) = {eval_df['gate_delta'].corr(eval_df['max_sim_eval']):.3f}",
        f"- corr(motion, primary_word_count) = {eval_df['motion'].corr(eval_df['primary_word_count']):.3f}",
        "",
        "## Figures",
        "- `01_similarity_histogram.png` — global pairwise similarity distribution",
        "- `02_family_similarity_boxplot.png` — similarity by task family",
        "- `03_word_count_distribution.png` — instruction length",
        "- `04_eval_heatmap.png` — 50 eval tasks similarity matrix",
        "- `05_eval_clustermap.png` — hierarchical clustering",
        "- `06_tsne_all_tasks.png` — 2D embedding (eval tasks highlighted)",
        "- `07_gate_vs_semantics.png` — gate delta vs language features",
        "- `08_top_similar_pairs.png` — highest-similarity pairs",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    print("Loading corpus...")
    df = load_corpus(args.dataset_root)
    per_task = build_per_task(df)
    sim = tfidf_similarity(per_task["primary"].tolist())
    all_sim = pairwise_offdiag(sim)

    print("Loading eval xlsx...")
    eval_df = load_eval_xlsx(args.xlsx)
    eval_df = attach_eval_features(eval_df, per_task, sim)

    eval_names = eval_df["Task"].tolist()
    name2idx = {n: i for i, n in enumerate(per_task["task_name"])}
    eval_idx = [name2idx[t] for t in eval_names]
    eval_sim = pairwise_offdiag(sim[np.ix_(eval_idx, eval_idx)])

    print("Plotting...")
    plot_similarity_histogram(all_sim, eval_sim, args.out_dir)
    plot_family_boxplot(per_task, sim, args.out_dir)
    plot_word_count_distribution(df, args.out_dir)
    plot_eval_heatmap(eval_df, sim, per_task, args.out_dir)
    plot_embedding_2d(per_task, eval_df, args.out_dir)
    plot_gate_vs_semantics(eval_df, args.out_dir)
    plot_top_pairs(per_task, sim, args.out_dir)

    eval_df.to_csv(args.out_dir / "eval_semantics_metrics.csv", index=False)
    write_summary(args.out_dir, per_task, df, all_sim, eval_sim, eval_df)
    print(f"Done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
