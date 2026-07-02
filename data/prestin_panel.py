"""
Prestin (SLC26A5) ortholog panel for substitution-convergence detection.

This is the OTHER kind of molecular convergence: the same orthologous protein
independently acquiring the same amino-acid states in different lineages. The
textbook case is Prestin, the outer-hair-cell motor protein, which shows
convergent substitutions in echolocating bats and toothed whales (Liu et al.
2010, Curr Biol; Li et al. 2010, PNAS; Foote et al. 2015, Nat Genet).

Detection is unsupervised: no phylogeny and no echolocation labels are given to
the detector. Relatedness is estimated from the sequences (an NJ tree), ancestral
states are reconstructed, and sites where the same residue arose on independent
branches are flagged. Echolocation labels below are used ONLY for evaluation.

The panel deliberately includes CONTROLS:
  - Non-echolocating bats (e.g. flying foxes, Pteropus): bat lineage without the
    trait, so "bat-ness" alone cannot explain a convergence signal.
  - A broad set of non-echolocating mammals as background.
A correct detector should rank echolocator-echolocator species pairs (bat x whale)
above pairs involving non-echolocators, using convergent SITES that survive the
ancestral-state control (i.e. not explained by shared ancestry).
"""

# label: display name; taxon: NCBI taxid for the UniProt query; echo: echolocates?
# query resolves SLC26A5 (prestin) for that species, live, at run time.
PRESTIN_SPECIES = [
    # --- echolocators: toothed whales ---
    {"label": "dolphin",      "taxon": 9739,   "echo": True,  "clade": "cetacea"},
    {"label": "sperm_whale",  "taxon": 9755,   "echo": True,  "clade": "cetacea"},
    {"label": "porpoise",     "taxon": 9740,   "echo": True,  "clade": "cetacea"},
    # --- echolocators: laryngeal-echolocating bats ---
    {"label": "myotis_bat",   "taxon": 59463,  "echo": True,  "clade": "chiroptera"},
    {"label": "horseshoe_bat", "taxon": 59479, "echo": True,  "clade": "chiroptera"},
    {"label": "microbat_eptesicus", "taxon": 29078, "echo": True, "clade": "chiroptera"},
    # --- CONTROL: non-echolocating bats (Old World fruit bats) ---
    {"label": "flying_fox",   "taxon": 9402,   "echo": False, "clade": "chiroptera"},
    {"label": "rousettus_fruitbat", "taxon": 9407, "echo": False, "clade": "chiroptera"},
    # --- CONTROL: non-echolocating whale relatives / other mammals ---
    {"label": "cow",          "taxon": 9913,   "echo": False, "clade": "artiodactyla"},
    {"label": "pig",          "taxon": 9823,   "echo": False, "clade": "artiodactyla"},
    {"label": "horse",        "taxon": 9796,   "echo": False, "clade": "perissodactyla"},
    {"label": "dog",          "taxon": 9615,   "echo": False, "clade": "carnivora"},
    {"label": "human",        "taxon": 9606,   "echo": False, "clade": "primates"},
    {"label": "mouse",        "taxon": 10090,  "echo": False, "clade": "rodentia"},
    {"label": "elephant",     "taxon": 9785,   "echo": False, "clade": "afrotheria"},
]


def build_species_queries():
    """Return member dicts with a UniProt query per species (resolved live)."""
    out = []
    for s in PRESTIN_SPECIES:
        out.append({
            "label": s["label"],
            "echo": s["echo"],
            "clade": s["clade"],
            # prefer reviewed; fall back handled by the fetcher's resolve logic
            "query": f'gene:SLC26A5 AND organism_id:{s["taxon"]}',
        })
    return out


def echo_ground_truth(members, clade_aware=True):
    """dict[(labelA,labelB)] -> 1 if a CONVERGENT echolocation pair, else 0.

    The two independent origins of laryngeal echolocation are in bats (Chiroptera)
    and toothed whales (Cetacea). So true convergence is a bat-echolocator paired
    with a whale-echolocator (different origin). Two whales sharing echo residues,
    or two bats, are shared ANCESTRY within one origin, not convergence.

    clade_aware=True  -> positive iff one member is a bat echolocator and the
                         other a whale echolocator (the defensible convergence set).
    clade_aware=False -> positive iff both echolocate (legacy; conflates
                         inheritance with convergence).
    Used ONLY for evaluation.
    """
    from itertools import combinations
    info = {s["label"]: s for s in PRESTIN_SPECIES}
    gt = {}
    for a, b in combinations(members, 2):
        la, lb = a["label"], b["label"]
        key = tuple(sorted((la, lb)))
        if not clade_aware:
            gt[key] = 1 if (a["echo"] and b["echo"]) else 0
            continue
        ca = info.get(la, {}).get("clade")
        cb = info.get(lb, {}).get("clade")
        both_echo = a["echo"] and b["echo"]
        cross_origin = ca != cb and {ca, cb} == {"chiroptera", "cetacea"}
        gt[key] = 1 if (both_echo and cross_origin) else 0
    return gt


# Accepted mammalian species-tree topology for the panel (no branch lengths;
# only topology matters for parsimony reconstruction). This is an INDEPENDENT
# relatedness backbone: the consensus mammal phylogeny, not derived from prestin.
# Prune to whichever species actually resolve at run time.
#   - Cetacea (dolphin, sperm_whale, porpoise) form a clade  -> shared echo
#     residues are inherited, correctly NOT counted as convergence between them.
#   - Bats split Yangochiroptera (Myotis, Eptesicus) vs Yinpterochiroptera
#     (Rhinolophus=horseshoe with the fruit bats), the accepted topology.
RESOLVED_MAMMAL_NEWICK = (
    "(elephant,((human,mouse),(((((dolphin,sperm_whale,porpoise),cow),pig),"
    "(horse,dog)),((myotis_bat,microbat_eptesicus),"
    "(horseshoe_bat,(flying_fox,rousettus_fruitbat))))));"
)


def load_resolved_tree(labels):
    """Return the accepted mammal topology pruned to `labels`, or None."""
    import io, tempfile, os
    from Bio import Phylo
    # robust across Biopython versions (older ones want a path, not StringIO)
    try:
        tree = Phylo.read(io.StringIO(RESOLVED_MAMMAL_NEWICK), "newick")
    except TypeError:
        with tempfile.NamedTemporaryFile("w", suffix=".nwk", delete=False) as fh:
            fh.write(RESOLVED_MAMMAL_NEWICK)
            path = fh.name
        tree = Phylo.read(path, "newick")
        os.unlink(path)
    present = set(labels)
    for tip in [t.name for t in tree.get_terminals()]:
        if tip not in present:
            try:
                tree.prune(target=tip)
            except Exception:
                pass
    got = {t.name for t in tree.get_terminals()}
    if not present.issubset(got):
        return None
    return tree
