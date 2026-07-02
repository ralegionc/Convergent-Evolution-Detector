"""
Offline validation of the substitution-convergence detector WITH null model.

Scenario reproduces the real challenge:
  - TRUE convergence: a bat clade and a whale clade each independently acquire
    the same residues at a few sites (cross-origin -> real signal).
  - INHERITANCE decoy: within-whale shared residues (must NOT count as
    convergence between the two whales).
  - CHANCE HOMOPLASY noise: many variable sites where random lineages collide
    on the same residue by chance (must be down-weighted by the null model).
"""
import os, sys, io
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Bio import Phylo
from detection import substitution_convergence as scv
from evaluation import evaluate_substitution as evs
from data.prestin_panel import echo_ground_truth

rng = np.random.default_rng(0)
AAS = list("ACDEFGHIKLMNPQRSTVWY")

species = ["dolphin", "sperm_whale", "myotis_bat", "horseshoe_bat",
           "cow", "human", "mouse", "dog"]
echo = {"dolphin": True, "sperm_whale": True, "myotis_bat": True,
        "horseshoe_bat": True, "cow": False, "human": False,
        "mouse": False, "dog": False}
newick = "((((dolphin,sperm_whale),cow),((myotis_bat,horseshoe_bat),dog)),(human,mouse));"
backbone = Phylo.read(io.StringIO(newick), "newick")

L = 400
base = "".join(rng.choice(AAS, size=L))
seqs = {s: list(base) for s in species}
for s, pos in [("cow", 5), ("dog", 6), ("human", 7), ("mouse", 8)]:
    seqs[s][pos] = "P"
# TRUE convergence: independent bat & whale origins of same residue
for col, aa in [(50, "W"), (100, "C"), (150, "Y")]:
    for s in ["dolphin", "sperm_whale"]:
        seqs[s][col] = aa
    for s in ["myotis_bat", "horseshoe_bat"]:
        seqs[s][col] = aa
# INHERITANCE decoy: sister whales share extra residues by descent
for col, aa in [(200, "K"), (210, "K")]:
    seqs["dolphin"][col] = aa
    seqs["sperm_whale"][col] = aa
# CHANCE HOMOPLASY noise
for col in range(250, 340):
    for s in rng.choice(species, size=rng.integers(2, 5), replace=False):
        seqs[s][col] = rng.choice(AAS)

records = [{"label": s, "organism": s, "sequence": "".join(seqs[s])} for s in species]

res = scv.run(records, min_independent=2, provided_tree=backbone,
              score="combined", run_simulation=True, n_sim=150)
labels = res["labels"]
print(f"convergent sites detected: {len(res['sites'])}  backbone={res['backbone']}")

members = [{"label": s, "echo": echo[s]} for s in species]
gt = echo_ground_truth(members, clade_aware=True)
pos = [k for k, v in gt.items() if v == 1]
print(f"clade-aware convergent pairs (bat x whale): {len(pos)}")

i, j = labels.index("dolphin"), labels.index("sperm_whale")
bi, wj = labels.index("myotis_bat"), labels.index("dolphin")
print(f"\ndolphin~sperm_whale RAW convergent count (should be ~0): {res['raw_score'][i,j]:.0f}")
print(f"myotis_bat~dolphin RAW convergent count (should be high): {res['raw_score'][bi,wj]:.0f}")

for name, key in [("RAW", "raw_score"), ("EXCESS(div-controlled)", "excess_score"),
                  ("Z(simulation null)", "zscore")]:
    out, ranking = evs.evaluate_pairs(labels, res[key], gt)
    print(f"  {name:24s} AP={out.get('average_precision'):.3f}  precision@1={out.get('precision@1')}")

out, ranking = evs.evaluate_pairs(labels, res["excess_score"], gt)
assert res["raw_score"][i, j] < 1, "sister-whale inheritance leaked as convergence"
assert ranking[0][2] == 1, f"top pair not convergent: {ranking[0]}"
assert out["average_precision"] >= 0.8, f"excess-score AP too low: {out['average_precision']:.2f}"
print("\nPASS: inheritance excluded; divergence-controlled excess score ranks bat x whale "
      "convergence on top despite chance-homoplasy noise.")
