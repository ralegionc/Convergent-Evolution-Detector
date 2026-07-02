"""
Embed protein sequences with ESM-2 (Meta) via HuggingFace transformers.

ESM-2 is a self-supervised protein language model trained by masked-token
prediction on UniRef. It never saw phylogenetic labels; its embeddings encode
learned biochemistry/structure. Two proteins with the same FUNCTION but
different ancestry land close in this space -- that closeness is the raw
material for the convergence signal.

CPU-only by default (no discrete GPU needed). The 150M model embeds a few
hundred sequences in minutes on a modern laptop CPU. Embeddings are the
mean over residue positions (per-protein 640-dim vector for t30_150M).
"""
import numpy as np


def load_model(model_name):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    torch.set_grad_enabled(False)
    return tok, model


def embed_sequences(records, model_name, max_len=1022, batch_size=4, verbose=True):
    """Return (labels, embeddings NxD) as mean-pooled ESM-2 representations.

    Mean pooling excludes special tokens and padding via the attention mask.
    """
    import torch
    tok, model = load_model(model_name)
    labels, seqs = [], []
    for r in records:
        labels.append(r["label"])
        seqs.append(r["sequence"][:max_len])

    vecs = []
    for start in range(0, len(seqs), batch_size):
        batch = seqs[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len + 2)
        out = model(**enc)
        hidden = out.last_hidden_state                      # (B, L, D)
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
        # zero out special tokens (CLS at 0, EOS at last real position)
        mask[:, 0, :] = 0.0
        lengths = enc["attention_mask"].sum(dim=1)
        for b in range(hidden.size(0)):
            mask[b, int(lengths[b]) - 1, :] = 0.0
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        mean = (summed / counts).cpu().numpy()
        vecs.append(mean)
        if verbose:
            print(f"[embed] {start + len(batch):3d}/{len(seqs)} sequences")
    emb = np.vstack(vecs).astype(np.float32)
    return labels, emb


def save_embeddings(labels, emb, path):
    np.savez_compressed(path, labels=np.array(labels), emb=emb)


def load_embeddings(path):
    d = np.load(path, allow_pickle=True)
    return list(d["labels"]), d["emb"]
