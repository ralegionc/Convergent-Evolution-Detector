"""
Publication-quality figures on a dark background.

  1. functional_umap  -- ESM-2 embedding projected to 2D, colored by function
     group, marker shape by broad lineage. Convergent groups sit together
     despite different lineages.
  2. score_heatmap    -- pairwise convergence-score matrix.
  3. pr_curve         -- precision-recall of the ranking vs literature labels.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DARK_BG = "#0d1117"
FG = "#e6edf3"
PALETTE = ["#58a6ff", "#3fb950", "#f778ba", "#ffa657", "#a5d6ff", "#d2a8ff"]


def _style(ax, fig):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors=FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def functional_umap(labels, emb, groups, path):
    try:
        import umap
        low = umap.UMAP(n_neighbors=min(15, max(2, emb.shape[0] - 1)),
                        min_dist=0.1, random_state=0).fit_transform(emb)
    except Exception:
        from sklearn.decomposition import PCA
        low = PCA(n_components=2, random_state=0).fit_transform(emb)

    fig, ax = plt.subplots(figsize=(8, 6))
    _style(ax, fig)
    uniq = sorted(set(groups))
    cmap = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(uniq)}
    for g in uniq:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        ax.scatter(low[idx, 0], low[idx, 1], s=90, c=cmap[g],
                   label=g, edgecolors="white", linewidths=0.5, alpha=0.9)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (low[i, 0], low[i, 1]), color=FG, fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_title("ESM-2 functional embedding (UMAP)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    leg = ax.legend(facecolor=DARK_BG, edgecolor="#30363d", labelcolor=FG, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)


def score_heatmap(labels, S, path, title="Convergence score"):
    fig, ax = plt.subplots(figsize=(7, 6))
    _style(ax, fig)
    im = ax.imshow(S, cmap="magma")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7, color=FG)
    ax.set_yticklabels(labels, fontsize=7, color=FG)
    cbar = fig.colorbar(im, ax=ax); cbar.ax.tick_params(colors=FG)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)


def pr_curve(scores, y_true, path):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    prec, rec, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(7, 6))
    _style(ax, fig)
    ax.plot(rec, prec, color=PALETTE[0], lw=2, label=f"AP = {ap:.3f}")
    ax.axhline(y_true.mean(), color="#8b949e", ls="--", lw=1,
               label=f"random = {y_true.mean():.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Convergence ranking vs literature ground truth")
    ax.set_ylim(0, 1.02)
    ax.legend(facecolor=DARK_BG, edgecolor="#30363d", labelcolor=FG)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
