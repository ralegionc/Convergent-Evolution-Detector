"""
Evaluate the unsupervised convergence ranking against literature ground truth.

The model produces a ranked list of species pairs by convergence score. Ground
truth (config.build_ground_truth) marks which pairs are literature-confirmed
convergent (same function, independent origin). We report:

  - Average precision (area under precision-recall curve)
  - Precision@k for small k
  - ROC AUC
  - Spearman rank correlation between score and the binary label

These are standard retrieval metrics: a good detector puts the known convergent
pairs (bacteriorhodopsin vs bovine rhodopsin, bat vs dolphin prestin, monarch
vs toad ATPase, ...) near the top of the ranking.
"""
import numpy as np
from itertools import combinations


def pairs_and_labels(labels, S, ground_truth):
    """Align score matrix with ground-truth labels over evaluable pairs.

    Only pairs that exist in ground_truth (i.e. within-function-group pairs)
    are evaluated -- cross-group pairs are undefined for convergence and excluded.
    Returns (scores, y_true, pair_names).
    """
    scores, y, names = [], [], []
    idx = {lab: i for i, lab in enumerate(labels)}
    for (a, b), lab in ground_truth.items():
        if a in idx and b in idx:
            scores.append(float(S[idx[a], idx[b]]))
            y.append(int(lab))
            names.append((a, b))
    return np.array(scores), np.array(y), names


def evaluate(labels, S, ground_truth):
    from sklearn.metrics import average_precision_score, roc_auc_score
    from scipy.stats import spearmanr

    scores, y, names = pairs_and_labels(labels, S, ground_truth)
    out = {"n_pairs": len(y), "n_positive": int(y.sum())}

    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        out["note"] = "degenerate label set (all same class); metrics skipped"
        out["ranking"] = sorted(zip(names, scores, y),
                                key=lambda t: t[1], reverse=True)
        return out

    out["average_precision"] = float(average_precision_score(y, scores))
    try:
        out["roc_auc"] = float(roc_auc_score(y, scores))
    except ValueError:
        out["roc_auc"] = None
    rho, p = spearmanr(scores, y)
    out["spearman_rho"] = float(rho)
    out["spearman_p"] = float(p)

    order = scores.argsort()[::-1]
    for k in (1, 3, 5, 10):
        if k <= len(order):
            topk = y[order[:k]]
            out[f"precision@{k}"] = float(topk.mean())

    out["ranking"] = [(names[i], float(scores[i]), int(y[i])) for i in order]
    return out


def print_report(result, title="Convergence detection"):
    print(f"\n=== {title} ===")
    print(f"pairs evaluated : {result['n_pairs']}  (positives: {result['n_positive']})")
    for key in ("average_precision", "roc_auc", "spearman_rho"):
        if key in result and result[key] is not None:
            print(f"{key:18s}: {result[key]:.3f}")
    for k in (1, 3, 5, 10):
        pk = result.get(f"precision@{k}")
        if pk is not None:
            print(f"precision@{k:<9d}: {pk:.3f}")
    print("\nTop-ranked pairs (score, convergent?):")
    for (a, b), s, y in result.get("ranking", [])[:12]:
        tag = "CONVERGENT" if y == 1 else "."
        print(f"  {s:6.3f}  {a:22s} {b:22s} {tag}")
