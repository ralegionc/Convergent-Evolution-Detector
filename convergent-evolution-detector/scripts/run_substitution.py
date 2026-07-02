"""
Substitution-convergence pipeline on REAL prestin (SLC26A5) orthologs.

Detects convergent amino-acid substitutions tied to echolocation, unsupervised:
no phylogeny and no echolocation labels reach the detector's scoring. The
relatedness backbone is an INDEPENDENT species tree (accepted mammal topology,
not derived from prestin), which is the correct control -- a prestin gene tree
would be distorted by the very convergence we test for (Liu 2010).

Two nulls address chance homoplasy (the central pitfall; Zou & Zhang 2015):
  - EXCESS: convergent-site count minus what the pair's divergence predicts
    (primary; convergence rises with divergence, so control for it).
  - Z: per-pair z-score vs a simulation null that redraws substitutions from
    each column's empirical amino-acid frequencies (supplementary).

Run:
    export NCBI_EMAIL="you@example.com"
    python scripts/run_substitution.py
"""
import os, sys, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from data import fetch_sequences as fs
from data.prestin_panel import (build_species_queries, echo_ground_truth,
                                 load_resolved_tree, PRESTIN_SPECIES)
from detection import substitution_convergence as scv
from evaluation import evaluate_substitution as evs

DATA_DIR = os.path.join(ROOT, "data")
RES_DIR = os.path.join(ROOT, "results")
SEQ_PATH = os.path.join(DATA_DIR, "prestin_sequences.json")


def stage_fetch():
    if os.path.exists(SEQ_PATH):
        print("[cache] loading prestin_sequences.json")
        return fs.load_records(SEQ_PATH)
    print("[fetch] querying UniProt for real prestin orthologs ...")
    recs = fs.fetch_panel(build_species_queries())
    fs.save_records(recs, SEQ_PATH)
    return recs


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    email = os.environ.get("NCBI_EMAIL", "your.email@example.com")

    records = stage_fetch()
    if len(records) < 5:
        print("Not enough orthologs resolved; aborting.")
        return

    echo = {s["label"]: s["echo"] for s in PRESTIN_SPECIES}
    labels_present = [r["label"] for r in records]

    # INDEPENDENT backbone: accepted mammal topology pruned to resolved species.
    print("[tree] loading resolved mammal species-tree backbone ...")
    backbone = load_resolved_tree(labels_present)
    if backbone is None:
        print("[warn] resolved topology missing some species; trying taxonomy ...")
        try:
            backbone = scv.taxonomy_tree(records, email)
        except Exception as e:
            print(f"[warn] taxonomy failed ({e}); will use NJ-iterative fallback")
            backbone = None

    print("[detect] MSA + ancestral reconstruction + null models ...")
    res = scv.run(records, min_independent=2, provided_tree=backbone,
                  score="combined", run_simulation=True, n_sim=200)
    print(f"[detect] backbone={res['backbone']}  "
          f"convergent+parallel sites: {len(res['sites'])}")

    members = [{"label": r["label"], "echo": echo.get(r["label"], False)}
               for r in records]
    gt = echo_ground_truth(members, clade_aware=True)
    n_pos = sum(gt.values())
    print(f"[eval] clade-aware convergent (bat x whale) pairs in panel: {n_pos}")
    if n_pos == 0:
        print("[eval] no bat x whale pair resolved; cannot evaluate convergence. "
              "Need at least one echolocating bat AND one toothed whale.")

    for name, key in [("RAW count", "raw_score"),
                      ("EXCESS (divergence-controlled)", "excess_score"),
                      ("Z (simulation null)", "zscore")]:
        out, ranking = evs.evaluate_pairs(res["labels"], res[key], gt)
        evs.print_report(out, ranking, title=f"Prestin convergence — {name}")

    # top convergent sites, annotated with how many echolocators carry the residue
    print("\nTop convergent+parallel sites (independent origins):")
    def n_indep(s): return len(s["origins"])
    for s in sorted(res["sites"], key=n_indep, reverse=True)[:15]:
        tips = sorted({t for o in s["origins"] for t in o["tips"]})
        echos = [t for t in tips if echo.get(t)]
        print(f"  col {s['col']:4d} ->{s['target_aa']} origins={n_indep(s)} "
              f"tips={','.join(tips)}  ({len(echos)} echolocators)")

    with open(os.path.join(RES_DIR, "substitution_metrics.json"), "w") as f:
        summary = {}
        for key in ("raw_score", "excess_score", "zscore"):
            out, _ = evs.evaluate_pairs(res["labels"], res[key], gt)
            summary[key] = {k: v for k, v in out.items()}
        json.dump({"metrics": summary, "n_sites": len(res["sites"]),
                   "backbone": res["backbone"]}, f, indent=2)
    print("\nDone. See results/substitution_metrics.json")


if __name__ == "__main__":
    main()
