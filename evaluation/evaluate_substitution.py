"""
Evaluate the substitution-convergence detector against echolocation ground truth.

The detector ranks species pairs by the number of convergent substitution sites
they share via independent origin. A good detector ranks echolocator x
echolocator pairs (bat x toothed-whale) at the top, because prestin is a
literature-confirmed case of convergent molecular evolution tied to laryngeal
echolocation (Liu et al. 2010; Li et al. 2010; Foote et al. 2015).

Echolocation labels are used ONLY here, never by the detector.
"""
import numpy as np
from itertools import combinations


def evaluate_pairs(labels, S, ground_truth):
    from sklearn.metrics import average_precision_score, roc_auc_score
    idx = {l: i for i, l in enumerate(labels)}
    scores, y, names = [], [], []
    for (a, b), lab in ground_truth.items():
        if a in idx and b in idx:
            scores.append(float(S[idx[a], idx[b]]))
            y.append(int(lab))
            names.append((a, b))
    scores, y = np.array(scores), np.array(y)
    out = {"n_pairs": len(y), "n_positive": int(y.sum())}
    if len(y) == 0 or y.sum() in (0, len(y)):
        out["note"] = "degenerate label set; metrics skipped"
        return out, list(zip(names, scores, y))
    out["average_precision"] = float(average_precision_score(y, scores))
    try:
        out["roc_auc"] = float(roc_auc_score(y, scores))
    except ValueError:
        out["roc_auc"] = None
    order = scores.argsort()[::-1]
    for k in (1, 3, 5):
        if k <= len(order):
            out[f"precision@{k}"] = float(y[order[:k]].mean())
    ranking = [(names[i], float(scores[i]), int(y[i])) for i in order]
    return out, ranking


def print_report(out, ranking, title="Substitution convergence"):
    print(f"\n=== {title} ===")
    print(f"species pairs : {out['n_pairs']}  (echolocator-echolocator: {out['n_positive']})")
    for k in ("average_precision", "roc_auc"):
        if out.get(k) is not None:
            print(f"{k:18s}: {out[k]:.3f}")
    for k in (1, 3, 5):
        if out.get(f"precision@{k}") is not None:
            print(f"precision@{k:<9d}: {out[f'precision@{k}']:.3f}")
    print("\nTop species pairs by convergent-site count:")
    for (a, b), s, y in ranking[:12]:
        tag = "ECHO-ECHO" if y == 1 else "."
        print(f"  {s:5.0f}  {a:20s} {b:20s} {tag}")
