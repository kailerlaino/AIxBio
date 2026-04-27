"""
Structural Validation — ESMFold PDB Generator
Reads gfp_protein_variants.fasta, folds the wild-type and first 5 protein
variants via the public ESMFold API, and saves the results as .pdb files.
"""

import os
import time
import requests
from Bio import SeqIO


FASTA_FILE   = "gfp_protein_variants.fasta"
OUTPUT_DIR   = "pdb_files"
ESMFOLD_URL  = "https://api.esmatlas.com/foldSequence/v1/pdb/"
REQUEST_TIMEOUT = 120   # seconds — ESMFold can be slow for long sequences
RATE_LIMIT_SLEEP = 5    # seconds between requests


def fold_sequence(sequence: str) -> str:
    """
    Sends an amino acid sequence to the ESMFold API and returns the PDB text.
    Raises requests.exceptions.RequestException on network/timeout errors.
    """
    response = requests.post(
        ESMFOLD_URL,
        data=sequence,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.text


def load_sequences(fasta_path: str) -> tuple[tuple[str, str], list[tuple[str, str]]]:
    """
    Parses the FASTA file and returns (wildtype, variants[:5]).

    The wild-type record is identified by an id or description containing
    'original' or 'wild' (case-insensitive). Falls back to the first record.
    Returns each sequence as (label, amino_acid_sequence).
    """
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        raise ValueError(f"No sequences found in {fasta_path}")

    wildtype_record = None
    variant_records = []

    for record in records:
        header = (record.id + " " + record.description).lower()
        if wildtype_record is None and ("original" in header or "wild" in header):
            wildtype_record = record
        else:
            variant_records.append(record)

    # Fallback: treat first record as wild-type if no explicit marker found
    if wildtype_record is None:
        wildtype_record = records[0]
        variant_records = records[1:]

    wildtype = (wildtype_record.id, str(wildtype_record.seq).upper())
    variants = [(r.id, str(r.seq).upper()) for r in variant_records[:5]]
    return wildtype, variants


def save_pdb(pdb_text: str, filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        f.write(pdb_text)
    print(f"  Saved: {path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading sequences from {FASTA_FILE}...")
    wildtype, variants = load_sequences(FASTA_FILE)
    wt_id, wt_seq = wildtype
    print(f"  Wild-type : {wt_id} ({len(wt_seq)} aa)")
    print(f"  Variants  : {len(variants)} (capped at 5)")

    jobs = [("wildtype", wt_seq)] + [(f"variant_{i+1:03d}", seq) for i, (_, seq) in enumerate(variants)]

    for i, (label, sequence) in enumerate(jobs):
        out_path = os.path.join(OUTPUT_DIR, f"{label}.pdb")
        if os.path.isfile(out_path):
            print(f"\n[{i+1}/{len(jobs)}] Skipping {label} — already exists.")
            continue

        print(f"\n[{i+1}/{len(jobs)}] Folding {label} ({len(sequence)} aa)...")
        try:
            pdb_text = fold_sequence(sequence)
            save_pdb(pdb_text, f"{label}.pdb")
        except requests.exceptions.Timeout:
            print(f"  ERROR: Request timed out after {REQUEST_TIMEOUT}s — skipping {label}.")
        except requests.exceptions.HTTPError as e:
            print(f"  ERROR: HTTP {e.response.status_code} — skipping {label}.")
        except requests.exceptions.RequestException as e:
            print(f"  ERROR: Network error — {e} — skipping {label}.")

        if i < len(jobs) - 1 and not os.path.isfile(out_path):
            print(f"  Waiting {RATE_LIMIT_SLEEP}s before next request...")
            time.sleep(RATE_LIMIT_SLEEP)

    print(f"\nDone. PDB files written to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
