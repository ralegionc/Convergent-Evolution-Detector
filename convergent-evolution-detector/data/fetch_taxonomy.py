"""
Fetch NCBI taxonomic lineages (Method B, optional cross-check only).

IMPORTANT framing: the detector's primary relatedness estimate is the sequence
proxy in detection/scores.py, which needs no tree and no taxonomy. This module
provides an INDEPENDENT coarse relatedness axis (shared-lineage depth) purely to
sanity-check the sequence proxy. It is never used as a supervised label, and the
main pipeline runs end-to-end without it.

Uses Biopython Entrez. Requires an email (NCBI policy).
"""
import time
from Bio import Entrez


# Standard Linnaean ranks, root -> tip. Deeper shared rank == more related.
RANK_ORDER = [
    "superkingdom", "kingdom", "phylum", "class",
    "order", "family", "genus", "species",
]


def get_lineage(organism_name, email, retries=3):
    """Return ordered lineage dict {rank: taxon_name} for a scientific name."""
    Entrez.email = email
    for attempt in range(retries):
        try:
            h = Entrez.esearch(db="taxonomy", term=organism_name)
            rec = Entrez.read(h)
            h.close()
            ids = rec.get("IdList", [])
            if not ids:
                return {}
            h = Entrez.efetch(db="taxonomy", id=ids[0], retmode="xml")
            data = Entrez.read(h)
            h.close()
            if not data:
                return {}
            lineage = {}
            for node in data[0].get("LineageEx", []):
                rank = node.get("Rank")
                if rank in RANK_ORDER:
                    lineage[rank] = node.get("ScientificName")
            lineage[data[0].get("Rank", "species")] = data[0].get("ScientificName")
            return lineage
        except Exception as e:
            if attempt == retries - 1:
                print(f"[warn] taxonomy fetch failed for {organism_name}: {e}")
                return {}
            time.sleep(2.0 * (attempt + 1))
    return {}


def lineage_distance(lin_a, lin_b):
    """Coarse phylogenetic distance in [0,1] from shared-rank depth.

    0.0 == identical down to species; 1.0 == share nothing (different kingdoms).
    Distance = 1 - (deepest shared rank index + 1) / len(RANK_ORDER).
    """
    deepest_shared = -1
    for i, rank in enumerate(RANK_ORDER):
        a, b = lin_a.get(rank), lin_b.get(rank)
        if a and b and a == b:
            deepest_shared = i
        else:
            break
    return 1.0 - (deepest_shared + 1) / len(RANK_ORDER)


def build_distance_matrix(records, email):
    """Return (labels, NxN numpy distance matrix) from live NCBI lineages."""
    import numpy as np
    lineages = {}
    for rec in records:
        org = rec.get("organism")
        if org and org not in lineages:
            lineages[org] = get_lineage(org, email)
            time.sleep(0.34)
    labels = [r["label"] for r in records]
    n = len(records)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            oi, oj = records[i].get("organism"), records[j].get("organism")
            d = lineage_distance(lineages.get(oi, {}), lineages.get(oj, {}))
            D[i, j] = D[j, i] = d
    return labels, D
