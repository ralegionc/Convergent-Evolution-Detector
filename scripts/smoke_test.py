"""
Smoke test: verify the detection + evaluation math on a controlled example
that reproduces the STRUCTURE of the real problem, without needing network,
ESM weights, or UniProt. This is NOT synthetic training data -- it is a unit
test of the scoring logic. The real pipeline (run_pipeline.py) uses only real
sequences and real ESM-2 embeddings.

Construction:
  - 6 "proteins" in 2 function groups.
  - Group 1 (light): 3 members. Two are functionally identical but from very
    different raw sequences (the convergent pair). One is a homolog (similar
    function AND similar sequence -> NOT convergent).
  - Group 2 (motor): 3 members, one convergent pair likewise.
A correct detector must rank the two convergent pairs above the homolog pairs.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from detection import scores as sc
from evaluation import evaluate as ev

rng = np.random.default_rng(0)

# Build embeddings so that within-group function is similar. Group centroid
# defines function; individual noise is small (function conserved).
def make(centroid, noise=0.05):
    v = centroid + rng.normal(0, noise, size=centroid.shape)
    return v.astype(np.float32)

D = 32
c_light = rng.normal(0, 1, D)
c_motor = rng.normal(0, 1, D)

labels = ["light_A", "light_B_conv", "light_C_homolog",
          "motor_A", "motor_B_conv", "motor_C_homolog"]
emb = np.vstack([
    make(c_light), make(c_light), make(c_light),   # all 3 functionally close
    make(c_motor), make(c_motor), make(c_motor),
])

# Raw sequences: convergent members share FUNCTION but NOT sequence.
# We encode "raw sequence relatedness" directly via distinct k-mer content.
# light_A & light_B_conv: very different letters (distant lineages).
# light_A & light_C_homolog: overlapping letters (related lineage).
sequences = [
    "AAAAAAAAAACCCCCCCCCC" * 5,   # light_A
    "GGGGGGGGGGTTTTTTTTTT" * 5,   # light_B_conv  (no shared 3-mers with A)
    "AAAAAAAAAACCCCCCCCCG" * 5,   # light_C_homolog (almost identical to A)
    "MMMMMMMMMMWWWWWWWWWW" * 5,   # motor_A
    "FFFFFFFFFFYYYYYYYYYY" * 5,   # motor_B_conv
    "MMMMMMMMMMWWWWWWWWWG" * 5,   # motor_C_homolog (almost identical to motor_A)
]

# Ground truth: within each group, the (A, B_conv) pair is convergent (1);
# pairs involving the homolog are same-lineage similarity (0).
ground_truth = {
    tuple(sorted(("light_A", "light_B_conv"))): 1,
    tuple(sorted(("light_A", "light_C_homolog"))): 0,
    tuple(sorted(("light_B_conv", "light_C_homolog"))): 1,   # B is also distant from C
    tuple(sorted(("motor_A", "motor_B_conv"))): 1,
    tuple(sorted(("motor_A", "motor_C_homolog"))): 0,
    tuple(sorted(("motor_B_conv", "motor_C_homolog"))): 1,
}

res = sc.ensemble(emb, sequences)
print("residual linear fit (a, b):", np.round(res["residual_fit"], 3))
print("cluster labels           :", res["cluster_labels"])

result = ev.evaluate(labels, res["ensemble"], ground_truth)
ev.print_report(result, title="SMOKE TEST (controlled)")

ap = result.get("average_precision", 0)
assert ap >= 0.8, f"detector failed to rank convergent pairs (AP={ap:.2f})"
print("\nPASS: convergent pairs ranked above homolog pairs (AP >= 0.8)")
