"""
Homology-gated convergence detection (corrected approach).

Analogous convergence is, by definition, *same function, no common ancestry*.
The residual approach in scores.py conflates this with divergent homology,
because ESM-2 similarity is itself largely a homology signal. This module fixes
that with a two-step rule that matches the biological definition:

  1. GATE: group sequences into homology FAMILIES. Two sequences are in the
     same family if they can be linked by a chain of significant local
     alignments (transitive closure of pairwise identity >= threshold). This
     correctly groups deeply-divergent homologs that a flat pairwise threshold
     would miss (e.g. halorhodopsin vs proteorhodopsin at ~24% identity are
     linked through bacteriorhodopsin). Only CROSS-family pairs survive as
     convergence candidates.

  2. RANK: among cross-family pairs, rank by ESM-2 functional similarity. High
     functional similarity between DIFFERENT homology families is the signature
     of analogous convergence (e.g. microbial type-1 rhodopsin vs animal type-2
     opsin: same 7-TM retinal photoreceptor job, zero common ancestry).

No phylogeny is used. Homology is estimated from the sequences themselves.
"""
import numpy as np
from itertools import combinations

_ALIGNER = None


def _get_aligner():
    global _ALIGNER
    if _ALIGNER is None:
        from Bio import Align
        from Bio.Align import substitution_matrices
        a = Align.PairwiseAligner()
        a.mode = "local"
        a.substitution_matrix = substitution_matrices.load("BLOSUM62")
        a.open_gap_score = -11
        a.extend_gap_score = -1
        _ALIGNER = a
    return _ALIGNER


def percent_identity(seq_a, seq_b):
    """Local-alignment percent identity, normalized by the shorter sequence.

    Normalizing by min length (not aligned length) prevents a short high-identity
    local hit from masquerading as whole-protein homology.
    """
    aligner = _get_aligner()
    try:
        aln = aligner.align(seq_a, seq_b)[0]
    except Exception:
        return 0.0
    c = aln.counts()
    identities = c.identities
    denom = min(len(seq_a), len(seq_b))
    return identities / denom if denom else 0.0


def identity_matrix(sequences, verbose=False):
    n = len(sequences)
    M = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        pid = percent_identity(sequences[i], sequences[j])
        M[i, j] = M[j, i] = pid
        if verbose:
            print(f"  identity[{i},{j}] = {pid:.3f}")
    np.fill_diagonal(M, 1.0)
    return M


def homology_components(sequences, identity_threshold=0.25, identity_mat=None):
    """Group sequences into homology families by transitive closure.

    Homology is an equivalence relation: if A aligns to C and B aligns to C, then
    A and B are homologs even if their DIRECT identity fell below threshold (the
    twilight zone). We build a graph with an edge wherever pairwise identity >=
    threshold, then take connected components as protein families. This fixes the
    failure where two deeply-divergent homologs (e.g. halorhodopsin vs
    proteorhodopsin, ~24% identity) slip through a flat pairwise gate.

    Returns (component_labels, identity_mat).
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    if identity_mat is None:
        identity_mat = identity_matrix(sequences)
    A = (identity_mat >= identity_threshold).astype(int)
    np.fill_diagonal(A, 0)
    _, comp = connected_components(csr_matrix(A), directed=False)
    return comp, identity_mat


def homology_gated_score(emb, sequences, identity_threshold=0.25, verbose=False):
    """Convergence score = functional similarity, gated to CROSS-FAMILY pairs.

    Families are homology connected components (transitive closure), not raw
    pairwise identity, so deeply-divergent homologs are correctly grouped.

    Returns (S, identity_mat, gate_mask):
      S[i,j]        = ESM cosine similarity if i,j are in different families, else 0
      identity_mat  = pairwise percent identity
      gate_mask     = True where pair passed the gate (different families)
    """
    from detection.scores import cosine_similarity_matrix
    func_sim = cosine_similarity_matrix(emb)
    pid = identity_matrix(sequences)
    comp, _ = homology_components(sequences, identity_threshold, pid)
    if verbose:
        for c in sorted(set(comp)):
            print(f"  family {c}: idx {[i for i in range(len(comp)) if comp[i]==c]}")
    n = emb.shape[0]
    S = np.zeros((n, n))
    mask = np.zeros((n, n), dtype=bool)
    for i, j in combinations(range(n), 2):
        cross_family = comp[i] != comp[j]
        mask[i, j] = mask[j, i] = cross_family
        if cross_family:
            score = 0.5 * (func_sim[i, j] + 1.0)  # cosine -> [0,1]
            S[i, j] = S[j, i] = score
    return S, pid, mask
