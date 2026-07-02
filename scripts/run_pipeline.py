"""
End-to-end convergent-evolution detection on REAL data.

Run on a machine with network access (UniProt, NCBI, HuggingFace reachable):

    pip install -r requirements.txt
    export NCBI_EMAIL="you@example.com"
    python scripts/run_pipeline.py

Stages:
  1. Fetch real protein sequences from UniProt for the curated function panel.
  2. Embed them with ESM-2 (CPU).
  3. Compute the three unsupervised convergence signals + ensemble
     (relatedness proxy = raw k-mer sequence identity; no phylogeny input).
  4. (optional) Cross-check with NCBI lineage distance.
  5. Evaluate the ranking against literature ground truth.
  6. Write figures to results/.

Every artifact is cached, so re-runs skip network + embedding.
"""
import os, sys, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from config import (flatten_members, build_ground_truth, ESM_MODEL,
                    MAX_SEQ_LEN, DEFAULT_EMAIL, HOMOLOGY_IDENTITY_THRESHOLD)
from data import fetch_sequences as fs
from embeddings import esm_embed as ee
from detection import scores as sc
from detection import homology_gate as hg
from evaluation import evaluate as ev
from visualization import plots

DATA_DIR = os.path.join(ROOT, "data")
EMB_DIR = os.path.join(ROOT, "embeddings")
RES_DIR = os.path.join(ROOT, "results")
SEQ_PATH = os.path.join(DATA_DIR, "sequences.json")
EMB_PATH = os.path.join(EMB_DIR, "esm_embeddings.npz")


def stage_sequences():
    if os.path.exists(SEQ_PATH):
        print("[cache] loading sequences.json")
        return fs.load_records(SEQ_PATH)
    print("[fetch] querying UniProt for real sequences ...")
    recs = fs.fetch_panel(flatten_members())
    fs.save_records(recs, SEQ_PATH)
    return recs


def stage_embeddings(records):
    if os.path.exists(EMB_PATH):
        print("[cache] loading esm_embeddings.npz")
        labels, emb = ee.load_embeddings(EMB_PATH)
        return labels, emb
    print(f"[embed] running {ESM_MODEL} on CPU ...")
    labels, emb = ee.embed_sequences(records, ESM_MODEL, max_len=MAX_SEQ_LEN)
    ee.save_embeddings(labels, emb, EMB_PATH)
    return labels, emb


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    records = stage_sequences()
    if len(records) < 4:
        print("Not enough sequences resolved; aborting.")
        return

    labels, emb = stage_embeddings(records)
    # keep records aligned to embedding label order
    records = sorted(records, key=lambda r: labels.index(r["label"]))
    sequences = [r["sequence"] for r in records]
    rec_by_label = {r["label"]: r for r in records}

    gt = build_ground_truth(flatten_members())

    # ------------------------------------------------------------------
    # PRIMARY detector: homology-gated functional similarity, evaluated on the
    # ANALOGOUS-convergence group only (the case this method is designed for).
    # ------------------------------------------------------------------
    print("\n[detect] homology-gated analogous-convergence detector ...")
    ana_labels = [r["label"] for r in records if r["conv_type"] == "analogous"]
    ana_idx = [labels.index(l) for l in ana_labels]
    ana_emb = emb[ana_idx]
    ana_seqs = [rec_by_label[l]["sequence"] for l in ana_labels]

    S_gate, pid, mask = hg.homology_gated_score(
        ana_emb, ana_seqs, identity_threshold=HOMOLOGY_IDENTITY_THRESHOLD)

    # report the homology families the gate discovered (unsupervised)
    comp, _ = hg.homology_components(ana_seqs, HOMOLOGY_IDENTITY_THRESHOLD, pid)
    print("[gate] homology families discovered (transitive closure):")
    for c in sorted(set(comp)):
        fam = [ana_labels[i] for i in range(len(ana_labels)) if comp[i] == c]
        print(f"       family {c}: {', '.join(fam)}")
    print(f"[gate] {int(mask.sum() // 2)} of "
          f"{len(ana_labels) * (len(ana_labels) - 1) // 2} pairs are cross-family "
          f"(convergence candidates)")

    result = ev.evaluate(ana_labels, S_gate, gt)
    ev.print_report(result, title="Homology-gated analogous convergence")

    # Show which pairs were excluded as same-family homologs (transparency)
    from itertools import combinations
    print("\nExcluded as same-family homologs:")
    for i, j in combinations(range(len(ana_labels)), 2):
        if not mask[i, j]:
            print(f"  id={pid[i, j]:.2f}  {ana_labels[i]:22s} {ana_labels[j]:22s}")

    # ------------------------------------------------------------------
    # BASELINE for contrast: the naive residual ensemble on the same group.
    # Demonstrates why gating matters (residual surfaces divergent homologs).
    # ------------------------------------------------------------------
    print("\n[baseline] naive residual ensemble (for contrast) ...")
    res = sc.ensemble(ana_emb, ana_seqs)
    base = ev.evaluate(ana_labels, res["ensemble"], gt)
    ap_gate = result.get("average_precision")
    ap_base = base.get("average_precision")
    print(f"  residual ensemble AP : {ap_base:.3f}"
          if ap_base is not None else "  residual: n/a")
    print(f"  homology-gated AP    : {ap_gate:.3f}"
          if ap_gate is not None else "  gated: n/a")

    # ------------------------------------------------------------------
    # Substitution groups: out of scope for this detector. Report, don't score.
    # ------------------------------------------------------------------
    sub_labels = [r["label"] for r in records if r["conv_type"] == "substitution"]
    if sub_labels:
        print(f"\n[note] {len(sub_labels)} substitution-convergence proteins "
              f"({', '.join(sorted(set(l.split('_')[0] for l in sub_labels)))}) "
              "are NOT scored here -- whole-protein embeddings cannot detect\n"
              "       site-level convergent substitutions. Use the substitution "
              "detector (see README) for those.")

    print("\n[viz] writing figures to results/ ...")
    groups = [rec_by_label[l]["group"] for l in ana_labels]
    plots.functional_umap(ana_labels, ana_emb, groups,
                          os.path.join(RES_DIR, "functional_umap.png"))
    plots.score_heatmap(ana_labels, S_gate,
                        os.path.join(RES_DIR, "score_heatmap_gated.png"),
                        title="Homology-gated convergence score")
    scores, y, _ = ev.pairs_and_labels(ana_labels, S_gate, gt)
    if y.sum() not in (0, len(y)):
        plots.pr_curve(scores, y, os.path.join(RES_DIR, "pr_curve.png"))

    with open(os.path.join(RES_DIR, "metrics.json"), "w") as f:
        json.dump({"gated": {k: v for k, v in result.items() if k != "ranking"},
                   "residual_baseline": {k: v for k, v in base.items()
                                         if k != "ranking"}}, f, indent=2)
    print("\nDone. See results/.")


if __name__ == "__main__":
    if os.environ.get("NCBI_EMAIL"):
        import config
        config.DEFAULT_EMAIL = os.environ["NCBI_EMAIL"]
    main()
