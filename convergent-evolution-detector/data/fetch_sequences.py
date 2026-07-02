"""
Fetch real protein sequences from the UniProt REST API.

No synthetic data. Each function-group member is resolved from its UniProt
query string to a real accession + amino-acid sequence at run time. If a query
returns multiple hits we take the first reviewed entry (or the first hit if
none are reviewed), which keeps the panel reproducible without hard-coding IDs
that may become stale.

UniProt REST docs: https://www.uniprot.org/help/api_queries
"""
import io
import time
import json
import urllib.parse
import requests

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"
HEADERS = {"User-Agent": "convergent-evolution-detector/1.0 (research)"}


def _get(url, params=None, retries=3, backoff=2.0):
    """GET with polite retry/backoff for transient UniProt 5xx/429."""
    for attempt in range(retries):
        r = requests.get(url, params=params, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff * (attempt + 1))
            continue
        r.raise_for_status()
    r.raise_for_status()


def resolve_accession(query, anchor=None):
    """Resolve a UniProt query string to a single accession.

    Prefers the provided anchor accession if given (highest confidence),
    otherwise the first reviewed hit, otherwise the first hit.
    Returns (accession, protein_name, organism) or (None, None, None).
    """
    if anchor:
        return anchor, None, None
    params = {
        "query": query,
        "fields": "accession,protein_name,organism_name,reviewed",
        "format": "json",
        "size": 5,
    }
    r = _get(UNIPROT_SEARCH, params=params)
    results = r.json().get("results", [])
    if not results:
        return None, None, None
    reviewed = [x for x in results
                if x.get("entryType", "").lower().startswith("uniprotkb reviewed")]
    pick = (reviewed or results)[0]
    acc = pick["primaryAccession"]
    pname = (pick.get("proteinDescription", {})
                 .get("recommendedName", {})
                 .get("fullName", {})
                 .get("value"))
    org = pick.get("organism", {}).get("scientificName")
    return acc, pname, org


def fetch_fasta(accession):
    """Fetch the amino-acid sequence for one accession. Returns (header, seq)."""
    r = _get(UNIPROT_ENTRY.format(acc=accession))
    text = r.text.strip()
    if not text.startswith(">"):
        return None, None
    lines = text.splitlines()
    header = lines[0][1:]
    seq = "".join(lines[1:])
    return header, seq


def fetch_panel(members, sleep=0.34, verbose=True):
    """Resolve + download the full panel.

    Returns list of records:
      {label, group, origin, accession, organism, sequence, header}
    Members that cannot be resolved are skipped with a warning so a single
    missing entry never breaks the whole run.
    """
    records = []
    for m in members:
        acc, pname, org = resolve_accession(m["query"], m.get("anchor"))
        if acc is None:
            if verbose:
                print(f"[warn] could not resolve: {m['label']}  ({m['query']})")
            continue
        header, seq = fetch_fasta(acc)
        if seq is None or len(seq) < 30:
            if verbose:
                print(f"[warn] no usable sequence for {m['label']} ({acc})")
            continue
        records.append({
            "label": m["label"],
            "group": m.get("group"),
            "conv_type": m.get("conv_type"),
            "origin": m.get("origin"),
            "derived": m.get("derived"),
            "echo": m.get("echo"),
            "clade": m.get("clade"),
            "accession": acc,
            "organism": org,
            "sequence": seq,
            "header": header,
        })
        if verbose:
            print(f"[ok]  {m['label']:22s} {acc:10s} len={len(seq):4d}  {header[:48]}")
        time.sleep(sleep)  # be polite to UniProt
    return records


def save_records(records, path):
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def load_records(path):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import os
    from config import flatten_members
    recs = fetch_panel(flatten_members())
    out = os.path.join(os.path.dirname(__file__), "sequences.json")
    save_records(recs, out)
    print(f"\nSaved {len(recs)} real sequences -> {out}")
