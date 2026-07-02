"""
Substitution-convergence detector.

Finds sites where the same amino acid arose INDEPENDENTLY in different lineages,
controlling for shared ancestry via ancestral-state reconstruction. No phylogeny
and no phenotype labels are given; the tree is estimated from the sequences.

Pipeline:
  1. MSA of orthologs (pyfamsa / FAMSA, pip-installable, no system binary).
  2. Distance matrix (p-distance over aligned columns) -> neighbor-joining tree.
     This is the unsupervised relatedness estimate; no external phylogeny.
  3. Fitch parsimony ancestral reconstruction per alignment column.
  4. Convergent site = a residue reached by >= 2 INDEPENDENT substitution branches
     (independent per the reconstructed tree, so NOT explained by inheritance).
  5. Species-pair score = number of convergent sites the two tips SHARE via
     independent origins. Ranked and evaluated against echolocation labels.

The ancestral control in step 4 is the scientific core: two species sharing a
residue by descent from a common ancestor is NOT convergence; only independent
derivation counts.
"""
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np


# ---------------------------------------------------------------------------
# 1. Multiple sequence alignment
# ---------------------------------------------------------------------------

def align(records):
    """MSA via pyfamsa. records: list of {label, sequence}. Returns dict
    label -> aligned sequence (equal length, with '-' gaps)."""
    from pyfamsa import Aligner, Sequence
    seqs = [Sequence(r["label"].encode(), r["sequence"].encode()) for r in records]
    msa = Aligner(threads=1).align(seqs)
    return {s.id.decode(): s.sequence.decode() for s in msa}


# ---------------------------------------------------------------------------
# 2. Distance matrix + neighbor-joining tree (unsupervised relatedness)
# ---------------------------------------------------------------------------

def nj_tree(aln, exclude_cols=None):
    """Neighbor-joining tree from p-distances over aligned columns.

    exclude_cols: optional set of column indices to ignore when computing
    distances. Used to rebuild the tree AFTER removing candidate convergent
    sites, which otherwise inflate similarity between convergent taxa and
    distort the topology (the classic prestin gene-tree artifact, Liu 2010).
    """
    from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
    names = list(aln.keys())
    exclude = exclude_cols or set()
    length = len(next(iter(aln.values())))
    keep = [c for c in range(length) if c not in exclude]

    def pdist(a, b):
        n = d = 0
        for c in keep:
            x, y = aln[a][c], aln[b][c]
            if x == "-" or y == "-":
                continue
            n += 1
            d += (x != y)
        return d / n if n else 1.0

    lower = []
    for i in range(len(names)):
        lower.append([pdist(names[i], names[j]) for j in range(i + 1)])
    dm = DistanceMatrix(names=names, matrix=lower)
    tree = DistanceTreeConstructor().nj(dm)
    tree.root_at_midpoint()
    return tree


def taxonomy_tree(records, email):
    """Build an INDEPENDENT species-tree topology from NCBI taxonomy lineages.

    This is the recommended relatedness backbone for substitution-convergence
    detection: it is estimated from data OTHER than the gene under test, so it
    cannot be distorted by the very convergence we are testing for (unlike a
    gene tree; see Liu 2010). Branch lengths are irrelevant for Fitch parsimony;
    only topology matters.

    records need an 'organism' field (scientific name). Returns a Bio.Phylo tree
    whose tips are the record labels, or None if taxonomy could not be built.
    """
    import io
    from Bio import Phylo
    from data.fetch_taxonomy import get_lineage, RANK_ORDER

    paths = {}
    for r in records:
        org = r.get("organism")
        if not org:
            return None
        lin = get_lineage(org, email)
        path = [lin[rank] for rank in RANK_ORDER if lin.get(rank)]
        paths[r["label"]] = path + [r["label"]]  # unique leaf per species

    root = {}
    for label, path in paths.items():
        node = root
        for taxon in path:
            node = node.setdefault(taxon, {})

    def to_newick(node):
        parts = []
        for name, child in node.items():
            parts.append(to_newick(child) if child else name)
        return "(" + ",".join(parts) + ")" if len(parts) > 1 else parts[0]

    newick = to_newick(root) + ";"
    try:
        return Phylo.read(io.StringIO(newick), "newick")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 3-4. Fitch ancestral reconstruction + convergent-site detection
# ---------------------------------------------------------------------------

def _fitch_column(tree, tip_states):
    """Fitch parsimony. Returns (state_by_clade, substitution_branches).

    Gaps/unknowns ('-','X') are treated as missing (wildcard) so they neither
    create nor block substitutions spuriously.
    """
    ALL = None  # wildcard marker via None
    setmap = {}

    def up(clade):
        if clade.is_terminal():
            s = tip_states.get(clade.name, "-")
            setmap[clade] = None if s in ("-", "X") else {s}
            return setmap[clade]
        childsets = [up(c) for c in clade.clades]
        concrete = [cs for cs in childsets if cs is not None]
        if not concrete:
            setmap[clade] = None
            return None
        inter = set.intersection(*concrete)
        setmap[clade] = inter if inter else set.union(*concrete)
        return setmap[clade]

    up(tree.root)
    statemap = {}

    def resolve_root():
        s = setmap[tree.root]
        return sorted(s)[0] if s else None

    def down(clade, parent_state):
        s = setmap[clade]
        if s is None:
            statemap[clade] = parent_state
        elif parent_state in s:
            statemap[clade] = parent_state
        else:
            statemap[clade] = sorted(s)[0]
        for c in clade.clades:
            down(c, statemap[clade])

    root_state = resolve_root()
    down(tree.root, root_state)

    subs = []
    def walk(clade):
        for c in clade.clades:
            ps, cs = statemap[clade], statemap[c]
            if ps is not None and cs is not None and ps != cs:
                subs.append((ps, cs, c))
            walk(c)
    walk(tree.root)
    return statemap, subs


def detect_convergent_sites(aln, tree, min_independent=2):
    """For each column, find residues reached by >= min_independent INDEPENDENT
    substitution branches, recording each independent origin separately.

    Returns list of site dicts:
      {col, target_aa, origins: [{parent_state, tips:[...]}, ...]}
    where each origin is one independent branch that produced target_aa, and
    `tips` are the terminals under that branch actually carrying target_aa.
    Recording origins separately lets us (a) count only CROSS-origin pairs as
    convergent (two tips under the same origin share by inheritance), and
    (b) classify parallel (same parent_state) vs convergent (different).
    """
    names = list(aln.keys())
    length = len(next(iter(aln.values())))
    sites = []
    for col in range(length):
        tip_states = {n: aln[n][col] for n in names}
        statemap, subs = _fitch_column(tree, tip_states)
        by_target = defaultdict(list)
        for ps, cs, clade in subs:
            by_target[cs].append((ps, clade))
        for target, branches in by_target.items():
            origins = []
            for parent_state, clade in branches:
                tips = [t.name for t in clade.get_terminals()
                        if aln[t.name][col] == target]
                if tips:
                    origins.append({"parent_state": parent_state, "tips": sorted(tips)})
            if len(origins) >= min_independent:
                sites.append({"col": col, "target_aa": target, "origins": origins})
    return sites


# ---------------------------------------------------------------------------
# 5. Species-pair scores: convergent + parallel, counting only CROSS-origin pairs
# ---------------------------------------------------------------------------

def pair_scores(labels, sites):
    """Return dict of NxN matrices:
       'convergent' : cross-origin pairs with DIFFERENT ancestral state
       'parallel'   : cross-origin pairs with SAME ancestral state
       'combined'   : convergent + parallel
    Only tips under DIFFERENT independent origins are counted (same-origin tips
    share the residue by inheritance, not convergence)."""
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    conv = np.zeros((n, n)); par = np.zeros((n, n))
    for site in sites:
        origins = site["origins"]
        for a in range(len(origins)):
            for b in range(a + 1, len(origins)):
                oa, ob = origins[a], origins[b]
                same_anc = (oa["parent_state"] == ob["parent_state"])
                for ti in oa["tips"]:
                    for tj in ob["tips"]:
                        if ti in idx and tj in idx:
                            i, j = idx[ti], idx[tj]
                            M = par if same_anc else conv
                            M[i, j] += 1; M[j, i] += 1
    return {"convergent": conv, "parallel": par, "combined": conv + par}


# ---------------------------------------------------------------------------
# 6. Null model: control for divergence (Zou & Zhang 2015 insight).
# Convergent-site counts rise with divergence by chance; the meaningful signal
# is EXCESS convergence over what a pair's divergence predicts.
# ---------------------------------------------------------------------------

def pdistance_matrix(aln):
    names = list(aln.keys())
    n = len(names)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = aln[names[i]], aln[names[j]]
            num = den = 0
            for x, y in zip(a, b):
                if x == "-" or y == "-":
                    continue
                den += 1; num += (x != y)
            D[i, j] = D[j, i] = num / den if den else 1.0
    return D


def divergence_controlled(count_mat, divergence, degree=1):
    """Score = residual of count_mat regressed on divergence across all pairs.

    Positive residual = more convergence than the pair's divergence predicts.
    A polynomial fit (degree>=1) captures the roughly monotonic count-vs-
    divergence trend; residuals are the excess-convergence signal.
    Returns (residual_matrix, fit_coeffs)."""
    n = count_mat.shape[0]
    iu = np.triu_indices(n, k=1)
    x = divergence[iu]; y = count_mat[iu]
    coeffs = np.polyfit(x, y, deg=degree)
    pred = np.polyval(coeffs, x)
    resid = y - pred
    S = np.zeros_like(count_mat)
    S[iu] = resid
    return S + S.T, coeffs


def simulate_null(aln, tree, n_sim=200, seed=0, freq_mode="global"):
    """Simulation null for chance homoplasy.

    Redraws each observed substitution branch's child state from a null
    amino-acid distribution, then recounts cross-origin convergent pairs, to
    build a per-pair null distribution.

    freq_mode:
      'global' (default) -- protein-wide amino-acid frequencies. Non-circular:
          a residue rare overall but shared by several lineages looks
          surprising, as it should.
      'column' -- each column's own frequencies. CIRCULAR and over-conservative:
          a convergent residue is already frequent within its column, so the
          null reproduces the collision and hides the signal. Kept only for
          comparison / demonstration of the pitfall.

    Returns (mean_null, std_null) NxN matrices.
    """
    rng = np.random.default_rng(seed)
    names = list(aln.keys())
    n = len(names)
    length = len(aln[names[0]])
    idx = {nm: i for i, nm in enumerate(names)}

    # global (protein-wide) frequencies
    gcount = Counter(c for nm in names for c in aln[nm] if c not in ("-", "X"))
    gaas = list(gcount)
    gp = np.array([gcount[a] for a in gaas], float)
    gp = gp / gp.sum()

    col_freqs = []
    for col in range(length):
        if freq_mode == "column":
            counts = Counter(aln[nm][col] for nm in names if aln[nm][col] not in ("-", "X"))
            if counts:
                aas = list(counts); w = np.array([counts[a] for a in aas], float)
                col_freqs.append((aas, w / w.sum()))
            else:
                col_freqs.append((gaas, gp))
        else:
            col_freqs.append((gaas, gp))

    col_subs = []
    for col in range(length):
        tip_states = {nm: aln[nm][col] for nm in names}
        _, subs = _fitch_column(tree, tip_states)
        col_subs.append(subs)
    col_tips = [{clade: [t.name for t in clade.get_terminals()]
                 for _, _, clade in subs} for subs in col_subs]

    null_stack = np.zeros((n_sim, n, n))
    for s in range(n_sim):
        M = np.zeros((n, n))
        for col in range(length):
            if not col_subs[col]:
                continue
            aas, p = col_freqs[col]
            by_t = defaultdict(list)
            for (_, _, clade) in col_subs[col]:
                by_t[rng.choice(aas, p=p)].append(clade)
            for target, branches in by_t.items():
                for a in range(len(branches)):
                    for b in range(a + 1, len(branches)):
                        for x in col_tips[col][branches[a]]:
                            for y in col_tips[col][branches[b]]:
                                M[idx[x], idx[y]] += 1; M[idx[y], idx[x]] += 1
        null_stack[s] = M

    return null_stack.mean(axis=0), null_stack.std(axis=0)


def run(records, min_independent=2, provided_tree=None, score="combined",
        run_simulation=True, n_sim=200):
    """Full detector. score in {'convergent','parallel','combined'} selects
    which substitution class drives the pair ranking (convergent = different
    ancestral state = the stricter, more meaningful signal).

    Returns dict with alignment, tree, sites, labels, raw pair-score matrices,
    the divergence-controlled (excess) score, p-distance, and metadata."""
    aln = align(records)
    labels = list(aln.keys())

    if provided_tree is not None:
        tree = provided_tree
        backbone = "provided"
        iters = 0
        excluded = []
    else:
        exclude = set()
        tree = nj_tree(aln)
        sites = detect_convergent_sites(aln, tree, min_independent)
        it = 0
        for it in range(5):
            new_exclude = {s["col"] for s in sites}
            if new_exclude == exclude:
                break
            exclude = new_exclude
            tree = nj_tree(aln, exclude_cols=exclude)
            sites = detect_convergent_sites(aln, tree, min_independent)
        backbone = "nj_iterative"
        iters = it + 1
        excluded = sorted(exclude)

    sites = detect_convergent_sites(aln, tree, min_independent)
    mats = pair_scores(labels, sites)
    div = pdistance_matrix(aln)
    excess, coeffs = divergence_controlled(mats[score], div)

    result = {"alignment": aln, "tree": tree, "sites": sites, "labels": labels,
              "pair_scores": mats, "score_used": score,
              "raw_score": mats[score], "excess_score": excess,
              "divergence": div, "fit_coeffs": coeffs,
              "backbone": backbone, "iterations": iters, "excluded_cols": excluded}

    if run_simulation:
        mean_null, std_null = simulate_null(aln, tree, n_sim=n_sim)
        # floor std: integer counts with tiny variance otherwise inflate z
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (mats[score] - mean_null) / np.maximum(std_null, 1.0)
        z[~np.isfinite(z)] = 0.0
        result["null_mean"] = mean_null
        result["null_std"] = std_null
        result["zscore"] = z

    return result
