"""
Convergence detection scores. No phylogeny is given to the model.

The operational definition of convergence used here:

    Two lineages are convergent for a trait if they are FUNCTIONALLY SIMILAR
    (close in ESM-2 embedding space) yet EVOLUTIONARILY DISTANT (low raw
    sequence relatedness).

Because "convergence" is meaningless without some notion of relatedness, we
must estimate relatedness -- but we do it WITHOUT an external tree:

  Method A (primary, phylogeny-free): relatedness proxy = raw sequence identity
      estimated by k-mer cosine similarity. Needs nothing but the sequences.
  Method B (cross-check): coarse NCBI lineage distance (data/fetch_taxonomy.py).

Signals combined into an ensemble:
  1. residual_score  -- functional sim minus what raw-sequence relatedness predicts
  2. cluster_score   -- co-clustering in functional space despite low seq identity
  3. module_score    -- shared sparse ICA component despite genome-wide divergence

All three are unsupervised. Ground-truth labels are used only in evaluation.
"""
import numpy as np
from itertools import combinations
from collections import Counter


# ---------------------------------------------------------------------------
# Relatedness proxy (Method A): k-mer cosine similarity of raw sequences.
# ---------------------------------------------------------------------------

def kmer_vector(seq, k=3):
    counts = Counter(seq[i:i + k] for i in range(len(seq) - k + 1))
    return counts


def kmer_similarity_matrix(sequences, k=3):
    """NxN cosine similarity of k-mer frequency vectors (raw sequence proxy)."""
    vecs = [kmer_vector(s, k) for s in sequences]
    vocab = sorted({kmer for v in vecs for kmer in v})
    idx = {kmer: i for i, kmer in enumerate(vocab)}
    M = np.zeros((len(sequences), len(vocab)), dtype=np.float32)
    for r, v in enumerate(vecs):
        for kmer, c in v.items():
            M[r, idx[kmer]] = c
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Mn = M / norms
    return Mn @ Mn.T


# ---------------------------------------------------------------------------
# Functional similarity: cosine similarity of ESM-2 embeddings.
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    E = emb / norms
    return E @ E.T


# ---------------------------------------------------------------------------
# Signal 1: residual score.
# Regress functional similarity on relatedness across all pairs; the positive
# residual (more functionally similar than relatedness predicts) IS the
# convergence signal.
# ---------------------------------------------------------------------------

def residual_score(func_sim, related_sim):
    n = func_sim.shape[0]
    iu = np.triu_indices(n, k=1)
    x = related_sim[iu]           # relatedness proxy (raw sequence)
    y = func_sim[iu]              # functional similarity (ESM)
    # robust linear fit y ~ a*x + b
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    resid = y - pred
    S = np.zeros_like(func_sim)
    S[iu] = resid
    S = S + S.T
    return S, coef


# ---------------------------------------------------------------------------
# Signal 2: cluster score.
# Cluster in functional space; a pair that co-clusters but has LOW raw sequence
# identity is a convergence candidate. Falls back to a threshold rule if
# hdbscan/umap are unavailable.
# ---------------------------------------------------------------------------

def cluster_score(emb, related_sim, min_cluster_size=2):
    n = emb.shape[0]
    labels = _cluster_functional(emb, min_cluster_size)
    S = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        co = 1.0 if (labels[i] == labels[j] and labels[i] != -1) else 0.0
        # weight by evolutionary distance (1 - relatedness): distant + co-clustered = strong
        S[i, j] = S[j, i] = co * (1.0 - related_sim[i, j])
    return S, labels


def _cluster_functional(emb, min_cluster_size):
    try:
        import umap
        import hdbscan
        reducer = umap.UMAP(n_neighbors=min(15, max(2, emb.shape[0] - 1)),
                            min_dist=0.1, n_components=2, random_state=0)
        low = reducer.fit_transform(emb)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        return clusterer.fit_predict(low)
    except Exception:
        # Fallback: agglomerative on cosine distance
        from sklearn.cluster import AgglomerativeClustering
        sim = cosine_similarity_matrix(emb)
        dist = 1.0 - sim
        np.fill_diagonal(dist, 0.0)
        k = max(2, emb.shape[0] // 3)
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                        linkage="average")
        return model.fit_predict(dist)


# ---------------------------------------------------------------------------
# Signal 3: module score.
# Decompose functional space with ICA into sparse "modules". Two proteins that
# load strongly on the SAME module while being globally divergent share a
# convergent functional module (e.g. an eye/echolocation-like axis).
# ---------------------------------------------------------------------------

def module_score(emb, related_sim, n_components=None):
    from sklearn.decomposition import FastICA
    n = emb.shape[0]
    k = n_components or min(8, max(2, n - 1))
    ica = FastICA(n_components=k, random_state=0, max_iter=1000, whiten="unit-variance")
    try:
        comp = ica.fit_transform(emb)          # (n, k) loadings
    except Exception:
        return np.zeros((n, n))
    # normalize loadings per component; shared dominant module -> high product
    comp = np.abs(comp)
    comp /= (comp.max(axis=0, keepdims=True) + 1e-9)
    dominant = comp.argmax(axis=1)
    S = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        shared = 1.0 if dominant[i] == dominant[j] else 0.0
        strength = min(comp[i, dominant[i]], comp[j, dominant[j]])
        S[i, j] = S[j, i] = shared * strength * (1.0 - related_sim[i, j])
    return S


# ---------------------------------------------------------------------------
# Ensemble.
# ---------------------------------------------------------------------------

def _rank_normalize(S):
    """Map upper-triangle values to [0,1] by rank (robust to scale)."""
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = S[iu]
    order = vals.argsort().argsort().astype(np.float64)
    if len(order) > 1:
        order /= (len(order) - 1)
    out = np.zeros_like(S)
    out[iu] = order
    return out + out.T


def ensemble(emb, sequences, related_sim=None, weights=(0.5, 0.3, 0.2)):
    """Compute all three signals and a rank-normalized weighted ensemble.

    Returns dict with each signal matrix, the ensemble matrix, and metadata.
    """
    if related_sim is None:
        related_sim = kmer_similarity_matrix(sequences)
    func_sim = cosine_similarity_matrix(emb)

    S_res, coef = residual_score(func_sim, related_sim)
    S_clu, clabels = cluster_score(emb, related_sim)
    S_mod = module_score(emb, related_sim)

    Rr, Rc, Rm = (_rank_normalize(S) for S in (S_res, S_clu, S_mod))
    w = np.array(weights, dtype=float)
    w /= w.sum()
    E = w[0] * Rr + w[1] * Rc + w[2] * Rm

    return {
        "func_sim": func_sim,
        "related_sim": related_sim,
        "residual": S_res,
        "cluster": S_clu,
        "module": S_mod,
        "cluster_labels": clabels,
        "ensemble": E,
        "residual_fit": coef,
    }


def ranked_pairs(labels, S):
    """Flatten an NxN score matrix to a sorted list of (labelA,labelB,score)."""
    n = len(labels)
    out = []
    for i, j in combinations(range(n), 2):
        out.append((labels[i], labels[j], float(S[i, j])))
    out.sort(key=lambda t: t[2], reverse=True)
    return out
