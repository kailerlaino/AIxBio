"""
Step 1.2 — Legacy Baseline: Pairwise DNA Homology Screening
Loads wild-type + obfuscated DNA variants and calculates sequence identity
against a 70% regulatory flag threshold using Biopython pairwise alignment.
"""

import csv
import sys
from pathlib import Path
from Bio import SeqIO, pairwise2
from Bio.pairwise2 import format_alignment


# ── Configuration ─────────────────────────────────────────────────────────────

FLAG_THRESHOLD = 0.70          # 70% identity → regulatory flag
INPUT_CSV      = "variants.csv"  # expected columns: id, sequence, type
INPUT_FASTA    = "variants.fasta" # alternative FASTA input


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_from_csv(path: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Returns (wildtype_seq, [(variant_id, variant_seq), ...]).
    Expects columns: id, sequence, type   (type = 'wildtype' or 'variant')
    """
    wildtype = None
    variants = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_type = row["type"].strip().lower()
            seq_id   = row["id"].strip()
            seq      = row["sequence"].strip().upper()
            if seq_type == "wildtype":
                wildtype = seq
            else:
                variants.append((seq_id, seq))
    if wildtype is None:
        raise ValueError("No row with type='wildtype' found in CSV.")
    return wildtype, variants


def load_from_fasta(path: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Returns (wildtype_seq, [(variant_id, variant_seq), ...]).
    The first record is treated as the wild-type.
    """
    records = list(SeqIO.parse(path, "fasta"))
    if not records:
        raise ValueError("FASTA file is empty.")
    wildtype = str(records[0].seq).upper()
    variants = [(r.id, str(r.seq).upper()) for r in records[1:]]
    return wildtype, variants


def auto_load() -> tuple[str, list[tuple[str, str]]]:
    """Tries CSV first, then FASTA."""
    if Path(INPUT_CSV).exists():
        print(f"Loading from {INPUT_CSV}")
        return load_from_csv(INPUT_CSV)
    if Path(INPUT_FASTA).exists():
        print(f"Loading from {INPUT_FASTA}")
        return load_from_fasta(INPUT_FASTA)
    raise FileNotFoundError(
        f"No input file found. Expected '{INPUT_CSV}' or '{INPUT_FASTA}'."
    )


# ── Homology calculation ───────────────────────────────────────────────────────

def sequence_identity(seq_a: str, seq_b: str) -> float:
    """
    Global pairwise alignment (match=1, mismatch=0, no gap penalties).
    Identity = matched positions / alignment length.
    """
    alignments = pairwise2.align.globalxx(seq_a, seq_b)
    if not alignments:
        return 0.0
    best = alignments[0]
    aligned_a, aligned_b = best.seqA, best.seqB
    matches = sum(a == b for a, b in zip(aligned_a, aligned_b) if a != "-" or b != "-")
    alignment_length = len(aligned_a)
    return matches / alignment_length if alignment_length > 0 else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    wildtype, variants = auto_load()

    print(f"\nWild-type length : {len(wildtype)} bp")
    print(f"Variants loaded  : {len(variants)}")
    print(f"Flag threshold   : {FLAG_THRESHOLD:.0%} sequence identity")
    print("-" * 55)

    bypassed     = []
    flagged      = []
    identities   = []

    for vid, vseq in variants:
        identity = sequence_identity(wildtype, vseq)
        identities.append(identity)
        if identity < FLAG_THRESHOLD:
            bypassed.append((vid, identity))
        else:
            flagged.append((vid, identity))

    # ── Summary ───────────────────────────────────────────────────────────────
    avg_identity = sum(identities) / len(identities) if identities else 0.0
    min_identity = min(identities) if identities else 0.0
    max_identity = max(identities) if identities else 0.0

    print(f"\n{'RESULTS':=^55}")
    print(f"  Variants flagged (≥ {FLAG_THRESHOLD:.0%})  : {len(flagged):>4}")
    print(f"  Variants bypassed (< {FLAG_THRESHOLD:.0%}) : {len(bypassed):>4}  ← evaded legacy screening")
    print(f"\n  Average identity : {avg_identity:.2%}")
    print(f"  Min identity     : {min_identity:.2%}")
    print(f"  Max identity     : {max_identity:.2%}")
    print("=" * 55)

    if bypassed:
        print(f"\nSample bypassing variants (first 5):")
        for vid, identity in bypassed[:5]:
            print(f"  {vid:<20}  identity = {identity:.2%}  [BYPASS]")

    bypass_rate = len(bypassed) / len(variants) * 100 if variants else 0
    print(f"\nConclusion: {bypass_rate:.1f}% of obfuscated sequences evaded the "
          f"{FLAG_THRESHOLD:.0%} identity threshold.\n")


if __name__ == "__main__":
    main()
