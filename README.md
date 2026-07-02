# Convergent Evolution Detector

An unsupervised model that identifies molecular convergent evolution — cases
where distantly related lineages independently evolved the same functional
solution — **without being given a phylogeny**. Ground truth for evaluation
comes only from the comparative-genomics literature.

## The idea in one sentence

Analogous convergence is *same function, no common ancestry*. The detector finds
protein pairs that are **functionally similar (close in ESM-2 space) yet
non-homologous (no detectable sequence alignment)** — and ranks those. The
canonical case is microbial type-1 rhodopsins vs animal type-2 opsins: same
7-TM retinal-photoreceptor job, zero sequence homology, independent origins.

## A negative result that shaped the design

The first version scored functional similarity minus raw-sequence similarity (a
residual). It failed on real data — ROC AUC below 0.5, ranking convergent pairs
*below* homologs. The reason is instructive: **ESM-2 similarity is itself
dominated by homology**, so the residual surfaced *divergent homologs* (e.g.
squid vs Drosophila opsin — related but sequence-diverged) rather than analogs.
The true type-1 vs type-2 rhodopsin pairs sat mid-pack.

The fix is the homology **gate**: first exclude any pair with detectable
sequence homology (local-alignment identity above a twilight-zone threshold),
then rank the surviving non-homologous pairs by ESM-2 similarity. On the
validation structure this moves average precision from ~0.38 to ~1.0. Run
`python scripts/validate_gate.py` to see both side by side. The failed first
attempt is kept in `detection/scores.py` as an explicit baseline for contrast —
negative results are part of the record.

## Two kinds of convergence (only one is in scope)

- **Analogous** — non-homologous families, same function (type-1 vs type-2
  rhodopsins). **This detector handles it**, via the gate.
- **Substitution** — the *same* orthologous gene independently acquiring the
  same amino-acid states in different lineages (prestin in bats & whales;
  Na⁺/K⁺-ATPase in insects & toads). Whole-protein embeddings cannot see this —
  it lives at a few sites. Those groups are tagged `conv_type="substitution"` in
  `config.py` so evaluation **excludes** them rather than scoring them wrong;
  detecting them needs a site-level pipeline (see "Next steps").

## How relatedness is estimated without a tree

Convergence is undefined without some notion of relatedness, so we estimate it
— but never from an external phylogeny:

- **Method A (primary, phylogeny-free):** relatedness proxy = raw amino-acid
  k-mer cosine similarity. Needs nothing but the sequences themselves.
- **Method B (cross-check, optional):** coarse NCBI lineage distance
  (shared-rank depth). Independent of Method A; used only to sanity-check.
  The pipeline runs fully without it.

Neither is used as a supervised label. Labels enter **only** at evaluation.

## Pipeline

```
UniProt REST ──► ESM-2 embeddings ──► 3 unsupervised signals ──► ensemble ──► ranked pairs
 (real seqs)        (CPU, 150M)          │                                          │
                                         ├─ residual  (func sim vs seq relatedness) │
                                         ├─ cluster   (co-cluster despite distance) ├─► evaluate vs
                                         └─ module    (shared ICA module)           │   literature GT
```

1. **`data/fetch_sequences.py`** — pulls real sequences from the UniProt REST
   API for a curated panel of function groups. Accessions are resolved live
   from query strings (not hard-coded) so the panel stays reproducible.
2. **`embeddings/esm_embed.py`** — mean-pooled ESM-2 embeddings, CPU-only.
3. **`detection/scores.py`** — the three signals and the rank-normalized
   ensemble. This is the scientific core.
4. **`data/fetch_taxonomy.py`** — optional NCBI lineage cross-check (Method B).
5. **`evaluation/evaluate.py`** — average precision, ROC AUC, precision@k, and
   Spearman correlation against literature-confirmed convergent pairs.
6. **`visualization/plots.py`** — UMAP of functional space, score heatmap,
   precision-recall curve (dark theme).

## The three signals

| Signal | Intuition | Convergence = |
|---|---|---|
| **Residual** | Fit `func_sim ≈ a·seq_relatedness + b` over all pairs. | Large positive residual: more functionally similar than relatedness predicts. |
| **Cluster** | UMAP + HDBSCAN on ESM space (HDBSCAN/UMAP optional; agglomerative fallback). | Co-clustered in function but low raw sequence identity. |
| **Module** | FastICA decomposes ESM space into sparse functional modules. | Share a dominant module despite genome-wide divergence (catches *partial* convergence). |

Scores are rank-normalized to [0,1] and combined (default weights 0.5 / 0.3 / 0.2).

## Ground truth (evaluation only)

Curated from comparative genomics, with a strict definition: a pair is
convergent iff **both** members carry an *independently derived* specialization
of the same molecular function.

- Type-1 vs type-2 rhodopsins — Ernst et al. 2014; Shen et al. 2013
- Prestin (SLC26A5) echolocation convergence, bats vs toothed whales —
  Liu et al. 2010; Li et al. 2010; Foote et al. 2015
- Na⁺/K⁺-ATPase (ATP1A) cardiac-glycoside resistance, insects vs toads —
  Dobler et al. 2012; Ujvari et al. 2015

Homologous pairs (e.g. two animal opsins) are negatives — the hard case the
detector must get right.

## Second detector: substitution convergence (prestin / echolocation)

The analogous detector above handles non-homologous families. The *other* kind of
molecular convergence -- the same orthologous gene independently acquiring the
same residues in different lineages -- needs a site-level method. That lives in
`detection/substitution_convergence.py` and runs on real prestin (SLC26A5)
orthologs across echolocating and non-echolocating mammals.

Method: MSA (FAMSA via `pyfamsa`, no system binary) -> ancestral-state
reconstruction (Fitch parsimony) -> flag sites where the same residue arose on
independent branches -> rank species pairs by shared convergent sites, evaluated
against echolocation labels (used only in evaluation).

The critical subtlety, and a genuine lesson baked into the code: a gene tree is
**distorted by the very convergence being tested** (prestin convergence is strong
enough to group echolocators together in a naive tree; Liu et al. 2010). So the
relatedness backbone must be estimated *independently of the test gene*. The
detector uses an NCBI-taxonomy species tree as that backbone by default; a
self-contained NJ path with iterative convergent-site removal is available but is
known to be fragile when convergence dominates the signal. The ancestral-state
control is what separates true independent derivation from shared inheritance --
without it, bat+dolphin sharing a residue by common descent would be miscounted
as convergence.

Run it (needs UniProt + NCBI reachable):

```bash
export NCBI_EMAIL="you@example.com"
python scripts/validate_substitution.py   # offline: proves the logic
python scripts/run_substitution.py        # real prestin orthologs
```

Expected: echolocator x echolocator species pairs (bat x toothed-whale) rank at
the top by convergent-site count, while the non-echolocating fruit bat control
does not -- so "bat-ness" alone cannot explain the signal.

**Chance homoplasy and the null model.** With 20 amino acids, distant lineages
collide on the same residue constantly by chance; raw convergent-site counts are
dominated by this noise (the Thomas & Hahn 2015 critique of genome-wide
bat/dolphin convergence claims). The detector therefore reports three scores:

- `raw_score`     — convergent+parallel site count per pair (chance-dominated).
- `excess_score`  — **primary**: count minus what the pair's divergence predicts
  (convergence rises with divergence, so regress it out; Zou & Zhang 2015).
- `zscore`        — supplementary: per-pair z vs a simulation null that redraws
  each substitution from the column's empirical amino-acid frequencies.

**Two substitution classes.** Parallel (same ancestral residue -> same derived,
independently) and convergent (different ancestral -> same derived) are both
counted; the default `combined` score sums them, matching the literature.

**Resolved backbone and clade-aware truth.** The relatedness backbone is the
accepted mammal topology (pruned to resolved species), so that sister toothed
whales sharing echo residues are scored as INHERITANCE, not convergence — only
cross-origin (bat x whale) pairs count. Correspondingly, the evaluation ground
truth marks only bat-echolocator x whale-echolocator pairs as convergent.

**On real prestin, expect a contested result, not a clean one.** prestin
echolocation convergence is real but sits close to the chance-homoplasy boundary
even in the literature. A tool that returned AP 1.0 here would be hiding that;
the honest outcome is that the top raw pair is a whale pair, and after divergence
control the bat x whale signal is what stands out — if it does at all in your
resolved species set.

## Running it

Requires network access to UniProt, NCBI, and HuggingFace.

```bash
pip install -r requirements.txt
export NCBI_EMAIL="you@example.com"      # NCBI Entrez policy
python scripts/run_pipeline.py
```

Artifacts are cached (`data/sequences.json`, `embeddings/esm_embeddings.npz`),
so re-runs skip the network and the embedding pass. Figures and `metrics.json`
land in `results/`.

To sanity-check the scoring logic offline (no network, no ESM weights):

```bash
python scripts/smoke_test.py
```

The smoke test builds a controlled 6-protein example with known convergent
pairs and asserts the ensemble ranks them above the homolog pairs (AP ≥ 0.8).
It exists to test the *math*; the real pipeline uses only real data.

## Hardware

Designed for a CPU-only laptop (developed against a Ryzen 7735HS / 32 GB, no
discrete GPU). The 150M ESM-2 model embeds a few hundred sequences in minutes.
Swap `ESM_MODEL` in `config.py` to `esm2_t33_650M_UR50D` for higher fidelity if
you have the RAM and patience.

## Extending the panel

Add function groups in `config.py::FUNCTION_GROUPS`. Each member needs a UniProt
`query`, a `label`, and an `origin` tag. Tag independently-derived
specializations with a name in `DERIVED_ORIGINS`; tag ancestral references as a
baseline. Good candidates to add: lysozyme c in foregut fermenters, antifreeze
glycoproteins in polar fish, C4 PEPC across plant lineages, venom
phospholipases, electric-organ Na channels in mormyrids vs gymnotiforms.

## Honest limitations

- **Sequence-proxy relatedness is coarse.** k-mer similarity is a stand-in for
  true evolutionary distance; deep divergences all saturate near "very
  distant." Method B (taxonomy) helps triangulate but is also coarse.
- **ESM encodes function *and* residual homology.** For proteins with weak but
  real homology, the residual signal shrinks. The type-1/type-2 rhodopsin case
  works precisely because homology is absent.
- **Small curated panel.** The ground-truth set is modest by design (high
  confidence over coverage). Metrics on ~30 pairs are indicative, not final.
- **No structure.** Adding structural embeddings (e.g. from ESMFold or
  Foldseek) would sharpen the functional axis, especially for the module signal.
- **Homoplasy vs convergence.** The model does not distinguish adaptive
  convergence from horizontal gene transfer or retained ancestral state; those
  need explicit modeling (a natural next module).

## Next steps

- **Substitution-convergence detector** (for prestin / ATPase): align orthologs,
  reconstruct ancestral states from sequence-derived relatedness, and score
  sites where distant lineages independently reached the same residue. This is
  the Zou & Zhang (2015) style analysis, phylogeny estimated from the data.
- **Structural embeddings**: swap or augment ESM-2 with structure (ESMFold +
  Foldseek). Structure captures analogous fold/function even better than
  sequence, and would sharpen the gate's ranking step.
- **Expand the analogous panel**: carbonic anhydrase alpha/beta/gamma families
  (a textbook non-homologous convergent enzyme) and antifreeze protein families
  are strong additions that would broaden the evaluation beyond rhodopsins.

## Repository layout

```
convergent-evolution-detector/
├── config.py                 # curated panel + literature ground truth
├── data/
│   ├── fetch_sequences.py     # UniProt REST (real sequences)
│   ├── fetch_taxonomy.py      # NCBI lineage (also the substitution backbone)
│   └── prestin_panel.py       # echolocation species panel + ground truth
├── embeddings/
│   └── esm_embed.py           # ESM-2, CPU
├── detection/
│   ├── scores.py              # naive residual/cluster/module ensemble (baseline)
│   ├── homology_gate.py       # analogous convergence: gate on non-homology
│   └── substitution_convergence.py  # substitution convergence: MSA+ancestral+sites
├── evaluation/
│   ├── evaluate.py            # analogous: AP, ROC AUC, precision@k, Spearman
│   └── evaluate_substitution.py     # substitution: species-pair ranking vs echo
├── visualization/
│   └── plots.py               # UMAP, heatmap, PR curve
├── scripts/
│   ├── run_pipeline.py        # analogous detector, end-to-end on real data
│   ├── run_substitution.py    # substitution detector, real prestin orthologs
│   ├── validate_gate.py       # offline: gated vs naive (analogous)
│   ├── validate_substitution.py     # offline: substitution logic + ancestral control
│   └── smoke_test.py          # offline unit test of the scoring math
├── requirements.txt
└── README.md
```
