"""
Curated convergence panel and literature-derived ground truth.

The detector never receives a phylogeny as input. Relatedness is estimated at
run time from the sequences themselves (local-alignment identity). Ground truth
used ONLY for evaluation comes from the comparative genomics literature.

Two DISTINCT kinds of molecular convergence appear here, and they require
different detection methods:

  * ANALOGOUS convergence  (conv_type="analogous")
      Non-homologous protein families that independently evolved the SAME
      function (e.g. microbial type-1 rhodopsins vs animal type-2 opsins:
      same 7-TM retinal photoreceptor job, zero sequence homology).
      -> Detectable by the homology-gated detector (detection/homology_gate.py):
         same function + no homology.

  * SUBSTITUTION convergence  (conv_type="substitution")
      The SAME orthologous protein independently acquiring the same derived
      amino-acid states in different lineages (e.g. Prestin in bats and whales;
      Na+/K+-ATPase in milkweed insects and toads). The proteins are homologs;
      convergence lives at a few sites.
      -> NOT detectable by whole-protein embeddings. Needs a site-level
         convergent-substitution detector (a separate pipeline; see README).
      Included here so the evaluation can correctly EXCLUDE them from the
      analogous detector rather than silently scoring them wrong.

References:
  - Type-1 vs type-2 rhodopsins: Ernst et al. 2014, Chem Rev; Shen et al. 2013.
  - Prestin (SLC26A5) echolocation convergence, bats & toothed whales:
    Liu et al. 2010; Li et al. 2010; Foote et al. 2015, Nat Genet.
  - Na+/K+-ATPase (ATP1A) cardiac-glycoside resistance, insects & toads:
    Dobler et al. 2012, PNAS; Ujvari et al. 2015.
"""

FUNCTION_GROUPS = [
    {
        "group": "phototransduction",
        "conv_type": "analogous",
        "description": "Retinal-binding photoreceptors; type-1 (microbial) vs type-2 (animal)",
        "members": [
            # Animal type-2 opsins  (origin: opsin_type2)
            {"query": 'gene:RHO AND organism_id:9606 AND reviewed:true',
             "label": "human_rhodopsin", "origin": "opsin_type2", "anchor": "P08100"},
            {"query": 'gene:RHO AND organism_id:9913 AND reviewed:true',
             "label": "bovine_rhodopsin", "origin": "opsin_type2", "anchor": "P02699"},
            {"query": 'opsin AND organism_id:6577 AND reviewed:true',
             "label": "squid_opsin", "origin": "opsin_type2", "anchor": "P31356"},
            {"query": 'gene:ninaE AND organism_id:7227 AND reviewed:true',
             "label": "drosophila_rh1", "origin": "opsin_type2", "anchor": "P06002"},
            # Microbial type-1 rhodopsins  (origin: rhodopsin_type1) — NO homology to above
            {"query": 'bacteriorhodopsin AND organism_id:2242 AND reviewed:true',
             "label": "bacteriorhodopsin", "origin": "rhodopsin_type1", "anchor": "P02945"},
            {"query": 'proteorhodopsin AND reviewed:true',
             "label": "proteorhodopsin", "origin": "rhodopsin_type1", "anchor": "Q9F7P4"},
            # halorhodopsin: resolve live (previous fixed anchor returned no usable seq)
            {"query": 'halorhodopsin AND reviewed:true',
             "label": "halorhodopsin", "origin": "rhodopsin_type1", "anchor": None},
            # one more type-1 to strengthen the microbial side; resolve live
            {"query": 'sensory rhodopsin II AND reviewed:true',
             "label": "sensory_rhodopsin", "origin": "rhodopsin_type1", "anchor": None},
        ],
    },
    {
        "group": "outer_hair_cell_motility",
        "conv_type": "substitution",
        "description": "Prestin / SLC26A5; convergent substitutions in echolocators",
        # `derived=True` marks the independently-derived echolocation phenotype.
        "members": [
            {"query": 'gene:SLC26A5 AND organism_id:9606 AND reviewed:true',
             "label": "prestin_human", "derived": False, "anchor": "P58743"},
            {"query": 'gene:SLC26A5 AND organism_id:9785',
             "label": "prestin_elephant", "derived": False, "anchor": None},
            {"query": 'gene:SLC26A5 AND organism_id:59463',
             "label": "prestin_bat", "derived": True, "anchor": None},
            {"query": 'gene:SLC26A5 AND organism_id:9739',
             "label": "prestin_dolphin", "derived": True, "anchor": None},
        ],
    },
    {
        "group": "cardiac_glycoside_resistance",
        "conv_type": "substitution",
        "description": "Na+/K+-ATPase alpha (ATP1A); convergent toxin resistance",
        "members": [
            {"query": 'gene:ATP1A1 AND organism_id:9606 AND reviewed:true',
             "label": "atp1a1_human", "derived": False, "anchor": "P05023"},
            {"query": 'ATP1A AND organism_id:13037',                # monarch (Danaus plexippus)
             "label": "atp1a_monarch", "derived": True, "anchor": None},
            {"query": 'ATP1A AND organism_id:8404',                 # toad (Bufo/Rhinella)
             "label": "atp1a_toad", "derived": True, "anchor": None},
        ],
    },
]

# Origins denoting an independently-derived analogous family (for analogous GT).
DERIVED_ORIGINS = {"opsin_type2", "rhodopsin_type1"}


def build_ground_truth(members):
    """Return dict[(labelA,labelB)] -> 1 if a literature-confirmed convergent
    pair, else 0. Type-aware:

      analogous group   : positive iff the two members are from DIFFERENT
                          independently-derived families (different `origin`).
                          Same-family pairs (homologs) are negatives.
      substitution group: positive iff BOTH members carry the derived phenotype
                          (`derived=True`), i.e. independent acquisitions of the
                          same derived state.

    Only within-group pairs are defined; cross-group pairs are excluded.
    """
    gt = {}
    by_group = {}
    ctype = {}
    for m in members:
        by_group.setdefault(m["group"], []).append(m)
        ctype[m["group"]] = m.get("conv_type", "analogous")
    for group, ms in by_group.items():
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = ms[i], ms[j]
                key = tuple(sorted((a["label"], b["label"])))
                if ctype[group] == "analogous":
                    both_derived = (a.get("origin") in DERIVED_ORIGINS
                                    and b.get("origin") in DERIVED_ORIGINS)
                    independent = a.get("origin") != b.get("origin")
                    gt[key] = 1 if (both_derived and independent) else 0
                else:  # substitution
                    gt[key] = 1 if (a.get("derived") and b.get("derived")) else 0
    return gt


def flatten_members(conv_types=None):
    """Flatten FUNCTION_GROUPS into member dicts carrying group + conv_type.

    Pass conv_types={"analogous"} to restrict to groups this detector handles.
    """
    out = []
    for g in FUNCTION_GROUPS:
        if conv_types and g["conv_type"] not in conv_types:
            continue
        for m in g["members"]:
            out.append({**m, "group": g["group"], "conv_type": g["conv_type"]})
    return out


DEFAULT_EMAIL = "your.email@example.com"
ESM_MODEL = "facebook/esm2_t30_150M_UR50D"
MAX_SEQ_LEN = 1022
HOMOLOGY_IDENTITY_THRESHOLD = 0.25  # below this = non-homologous (twilight zone)
