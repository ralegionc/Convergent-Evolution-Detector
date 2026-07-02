"""
Offline validation of the CORRECTED (homology-gated) pipeline.

Reproduces the real analogous-convergence structure without network/ESM:
  - 4 'animal opsins'  : homologous to each other (shared family-A motif)
  - 4 'microbial rhod.': homologous to each other (shared family-B motif)
  - cross-family pairs : non-homologous (different alphabets) == the true
                         convergent pairs in build_ground_truth
  - ESM stand-in       : all 8 share a 'photoreceptor' direction (so functional
                         similarity is high across families), but homologs sit
                         slightly tighter -- exactly the property that made the
                         naive residual method fail.

Asserts: the homology-gated detector achieves high AP against the REAL
config.build_ground_truth, while the naive residual ensemble does not.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from detection import homology_gate as hg
from detection import scores as sc
from evaluation import evaluate as ev

# ---- real sequences so the alignment gate is real -------------------------
# Family A alphabet vs family B alphabet chosen to be disjoint -> low cross ID.
CORE_A = "ACDEFGHIKLMNPQRSTVWY" * 3          # shared animal-opsin core
CORE_B = "LIVMFWYCLIVMFWYCLIVM" * 3          # shared microbial core (diff residues)

def variant(core, tail_char, n_tail=40, swaps=0, seed=0):
    rng = np.random.default_rng(seed)
    s = list(core)
    for _ in range(swaps):                    # introduce divergence within family
        p = rng.integers(0, len(s))
        s[p] = rng.choice(list("ACDEFGHIKLMNPQRSTVWY"))
    return "".join(s) + tail_char * n_tail

labels = ["human_rhodopsin", "bovine_rhodopsin", "squid_opsin", "drosophila_rh1",
          "bacteriorhodopsin", "proteorhodopsin", "halorhodopsin", "sensory_rhodopsin"]

sequences = [
    variant(CORE_A, "A", swaps=5,  seed=1),   # animal opsins (family A)
    variant(CORE_A, "A", swaps=10, seed=2),
    variant(CORE_A, "C", swaps=25, seed=3),   # more divergent animal opsin
    variant(CORE_A, "C", swaps=30, seed=4),   # even more divergent -> homolog trap
    variant(CORE_B, "K", swaps=5,  seed=5),   # microbial (family B)
    variant(CORE_B, "K", swaps=10, seed=6),
    variant(CORE_B, "R", swaps=25, seed=7),
    variant(CORE_B, "R", swaps=30, seed=8),
]

# ---- ESM stand-in embeddings ----------------------------------------------
D = 64
rng = np.random.default_rng(0)
g = rng.normal(0, 1, D); g /= np.linalg.norm(g)          # shared photoreceptor axis
oa = rng.normal(0, 1, D); oa -= oa @ g * g; oa /= np.linalg.norm(oa)
ob = rng.normal(0, 1, D); ob -= ob @ g * g; ob /= np.linalg.norm(ob)

def emb_for(fam_off, seed):
    r = np.random.default_rng(seed)
    v = 1.0 * g + 0.55 * fam_off + r.normal(0, 0.04, D)   # g dominant -> cross-fam sim high
    return v
emb = np.vstack([emb_for(oa, s) for s in (11, 12, 13, 14)] +
                [emb_for(ob, s) for s in (15, 16, 17, 18)]).astype("float32")

# ---- real ground truth from config ----------------------------------------
from config import build_ground_truth, flatten_members
gt = build_ground_truth(flatten_members())

# ---- homology-gated detector ----------------------------------------------
S_gate, pid, mask = hg.homology_gated_score(emb, sequences, identity_threshold=0.25)
print("within-family identity (should be HIGH, gated out):")
print(f"  human~bovine opsin : {pid[0,1]:.2f}")
print(f"  bacterio~proteo    : {pid[4,5]:.2f}")
print("cross-family identity (should be LOW, kept as candidate):")
print(f"  human_opsin~bacterio: {pid[0,4]:.2f}")
print(f"  squid_opsin~sensory : {pid[2,7]:.2f}")

res_gate = ev.evaluate(labels, S_gate, gt)
ev.print_report(res_gate, title="CORRECTED homology-gated detector")

# ---- naive residual ensemble (the old, failing approach) ------------------
res = sc.ensemble(emb, sequences)
res_base = ev.evaluate(labels, res["ensemble"], gt)

ap_gate = res_gate.get("average_precision", 0.0)
ap_base = res_base.get("average_precision", 0.0)
print(f"\nnaive residual ensemble AP : {ap_base:.3f}")
print(f"homology-gated AP          : {ap_gate:.3f}")

assert ap_gate >= 0.9, f"gated detector underperformed (AP={ap_gate:.2f})"
assert ap_gate > ap_base, "gate should beat the naive residual baseline"
print("\nPASS: gating recovers the analogous-convergence signal the naive method missed.")
