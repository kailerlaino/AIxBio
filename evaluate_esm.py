"""
Step 1.3 — Semantic Screener: ESM-2 Embedding + Cosine Similarity
Translates DNA sequences to amino acids, embeds them with ESM-2,
and computes cosine similarity against the wild-type embedding.
"""

import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from Bio import SeqIO
from Bio.Seq import Seq
from transformers import AutoTokenizer, AutoModel


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME     = "facebook/esm2_t33_650M_UR50D"
INPUT_CSV      = "variants.csv"
INPUT_FASTA    = "variants.fasta"
SIMILARITY_THRESHOLD = 0.95


# ── Device setup ──────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── Loaders (mirrors run_blast.py) ────────────────────────────────────────────

def load_from_csv(path: str) -> tuple[str, list[tuple[str, str]]]:
    wildtype = None
    variants = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            seq_type = row["type"].strip().lower()
            seq      = row["sequence"].strip().upper()
            sid      = row["id"].strip()
            if seq_type == "wildtype":
                wildtype = seq
            else:
                variants.append((sid, seq))
    if wildtype is None:
        raise ValueError("No row with type='wildtype' found in CSV.")
    return wildtype, variants


def load_from_fasta(path: str) -> tuple[str, list[tuple[str, str]]]:
    records = list(SeqIO.parse(path, "fasta"))
    if not records:
        raise ValueError("FASTA file is empty.")
    wildtype = str(records[0].seq).upper()
    variants = [(r.id, str(r.seq).upper()) for r in records[1:]]
    return wildtype, variants


def auto_load() -> tuple[str, list[tuple[str, str]]]:
    if Path(INPUT_CSV).exists():
        print(f"Loading from {INPUT_CSV}")
        return load_from_csv(INPUT_CSV)
    if Path(INPUT_FASTA).exists():
        print(f"Loading from {INPUT_FASTA}")
        return load_from_fasta(INPUT_FASTA)
    raise FileNotFoundError(
        f"No input file found. Expected '{INPUT_CSV}' or '{INPUT_FASTA}'."
    )


# ── DNA → Amino Acid translation ──────────────────────────────────────────────

def dna_to_protein(dna: str) -> str:
    """
    Translates a DNA sequence to an amino acid string.
    Strips the stop codon (*) if present.
    Pads to a multiple of 3 if needed.
    """
    dna = dna.upper().replace(" ", "")
    remainder = len(dna) % 3
    if remainder:
        dna = dna[:-remainder]  # trim incomplete codon at end
    protein = str(Seq(dna).translate())
    # Remove stop codon and anything after it
    if "*" in protein:
        protein = protein[:protein.index("*")]
    return protein


# ── ESM-2 embedding ───────────────────────────────────────────────────────────

def load_model(model_name: str):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)
    model.eval().to(DEVICE)
    print("Model loaded.\n")
    return tokenizer, model


@torch.no_grad()
def get_embedding(protein_seq: str, tokenizer, model) -> torch.Tensor:
    """
    Returns the mean-pooled last hidden state for a protein sequence.
    Shape: (hidden_dim,)
    """
    inputs = tokenizer(
        protein_seq,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(DEVICE)

    outputs = model(**inputs)
    # outputs.last_hidden_state: (1, seq_len, hidden_dim)
    # Mask out special tokens (BOS/EOS) before pooling
    hidden = outputs.last_hidden_state[0]   # (seq_len, hidden_dim)
    # tokens 0 and -1 are <cls> and <eos> — exclude them
    token_embeddings = hidden[1:-1]
    mean_embedding   = token_embeddings.mean(dim=0)
    return mean_embedding


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    wildtype_dna, variants = auto_load()
    print(f"Wild-type length : {len(wildtype_dna)} bp")
    print(f"Variants loaded  : {len(variants)}\n")

    tokenizer, model = load_model(MODEL_NAME)

    # Wild-type embedding
    wt_protein   = dna_to_protein(wildtype_dna)
    print(f"Wild-type protein length: {len(wt_protein)} aa")
    wt_embedding = get_embedding(wt_protein, tokenizer, model)

    # Variant embeddings + cosine similarities
    similarities = []
    below_threshold = []

    print("Computing embeddings for all variants...")
    for i, (vid, vdna) in enumerate(variants):
        vprotein   = dna_to_protein(vdna)
        vembedding = get_embedding(vprotein, tokenizer, model)
        sim        = F.cosine_similarity(wt_embedding.unsqueeze(0),
                                         vembedding.unsqueeze(0)).item()
        similarities.append((vid, sim))
        if sim < SIMILARITY_THRESHOLD:
            below_threshold.append((vid, sim))

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(variants)} variants...")

    # ── Summary ───────────────────────────────────────────────────────────────
    scores      = [s for _, s in similarities]
    avg_sim     = sum(scores) / len(scores)
    min_sim     = min(scores)
    max_sim     = max(scores)
    above_count = len(scores) - len(below_threshold)

    print(f"\n{'RESULTS':=^55}")
    print(f"  Model            : {MODEL_NAME}")
    print(f"  Threshold        : {SIMILARITY_THRESHOLD}")
    print(f"  Average cosine similarity : {avg_sim:.4f}")
    print(f"  Min similarity            : {min_sim:.4f}")
    print(f"  Max similarity            : {max_sim:.4f}")
    print(f"  Above {SIMILARITY_THRESHOLD} threshold : {above_count:>4} / {len(scores)}")
    print(f"  Below {SIMILARITY_THRESHOLD} threshold : {len(below_threshold):>4} / {len(scores)}")
    print("=" * 55)

    detection_rate = above_count / len(scores) * 100
    print(f"\nConclusion: ESM-2 flagged {detection_rate:.1f}% of obfuscated sequences "
          f"as functionally similar to the wild-type (sim ≥ {SIMILARITY_THRESHOLD}).")

    if below_threshold:
        print(f"\nVariants below threshold (first 5):")
        for vid, sim in below_threshold[:5]:
            print(f"  {vid:<20}  similarity = {sim:.4f}")

    print()


if __name__ == "__main__":
    main()
