"""
Step 1.4 — Demo Dashboard
Streamlit app: paste an obfuscated DNA sequence, get BLAST homology
and ESM-2 cosine similarity scores side-by-side.
"""

import csv
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from Bio import SeqIO, pairwise2
from Bio.Seq import Seq
from transformers import AutoTokenizer, AutoModel


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_NAME        = "facebook/esm2_t33_650M_UR50D"
FLAG_THRESHOLD    = 0.70
SEMANTIC_THRESHOLD = 0.95
INPUT_CSV         = "variants.csv"
INPUT_FASTA       = "variants.fasta"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_wildtype() -> str:
    if Path(INPUT_CSV).exists():
        with open(INPUT_CSV, newline="") as f:
            for row in csv.DictReader(f):
                if row["type"].strip().lower() == "wildtype":
                    return row["sequence"].strip().upper()
        raise ValueError("No wildtype row found in CSV.")
    if Path(INPUT_FASTA).exists():
        records = list(SeqIO.parse(INPUT_FASTA, "fasta"))
        if records:
            return str(records[0].seq).upper()
    raise FileNotFoundError(
        f"No input file found. Place '{INPUT_CSV}' or '{INPUT_FASTA}' "
        "in the same directory as app.py."
    )


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading wild-type sequence...")
def get_wildtype_dna() -> str:
    return load_wildtype()


@st.cache_resource(show_spinner="Loading ESM-2 model (one-time, ~2 min on T4)...")
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)
    model.eval().to(DEVICE)
    return tokenizer, model


@st.cache_resource(show_spinner="Computing wild-type embedding...")
def get_wildtype_embedding():
    wt_dna            = get_wildtype_dna()
    wt_protein        = dna_to_protein(wt_dna)
    tokenizer, model  = get_model()
    return embed_protein(wt_protein, tokenizer, model)


# ── Biology helpers ───────────────────────────────────────────────────────────

def dna_to_protein(dna: str) -> str:
    dna = dna.upper().replace(" ", "").replace("\n", "")
    remainder = len(dna) % 3
    if remainder:
        dna = dna[:-remainder]
    protein = str(Seq(dna).translate())
    if "*" in protein:
        protein = protein[:protein.index("*")]
    return protein


@torch.no_grad()
def embed_protein(protein: str, tokenizer, model) -> torch.Tensor:
    inputs  = tokenizer(protein, return_tensors="pt",
                        truncation=True, max_length=1024).to(DEVICE)
    hidden  = model(**inputs).last_hidden_state[0]
    return hidden[1:-1].mean(dim=0)   # exclude <cls> and <eos>


def blast_identity(seq_a: str, seq_b: str) -> float:
    alignments = pairwise2.align.globalxx(seq_a, seq_b)
    if not alignments:
        return 0.0
    a, b = alignments[0].seqA, alignments[0].seqB
    matches = sum(x == y for x, y in zip(a, b))
    return matches / len(a) if a else 0.0


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="BioScreen AI", layout="centered")

st.title("BioScreen AI")
st.markdown(
    "Paste an obfuscated DNA sequence below. The pipeline runs **traditional "
    "homology screening** (BLAST-style) and **semantic AI screening** (ESM-2) "
    "in parallel, exposing the gap between them."
)
st.divider()

query_dna = st.text_area(
    label="Obfuscated DNA Sequence",
    placeholder="Paste a raw DNA sequence here (e.g. ATGCGT...)",
    height=160,
)

run_button = st.button("Screen Sequence", type="primary", use_container_width=True)

if run_button:
    query_dna = query_dna.strip().upper()

    # ── Input validation ──────────────────────────────────────────────────────
    if not query_dna:
        st.warning("Please paste a DNA sequence before screening.")
        st.stop()

    invalid = set(query_dna) - set("ATGCN")
    if invalid:
        st.error(f"Sequence contains invalid characters: {invalid}")
        st.stop()

    if len(query_dna) < 9:
        st.error("Sequence is too short (minimum 9 bp).")
        st.stop()

    with st.spinner("Running pipeline..."):
        try:
            wt_dna       = get_wildtype_dna()
            tokenizer, model = get_model()
            wt_embedding = get_wildtype_embedding()

            # BLAST score
            blast_score = blast_identity(wt_dna, query_dna)

            # ESM-2 score
            query_protein   = dna_to_protein(query_dna)
            query_embedding = embed_protein(query_protein, tokenizer, model)
            cosine_sim      = F.cosine_similarity(
                wt_embedding.unsqueeze(0),
                query_embedding.unsqueeze(0)
            ).item()

        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    # ── Results ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Screening Results")

    blast_flagged    = blast_score >= FLAG_THRESHOLD
    semantic_flagged = cosine_sim  >= SEMANTIC_THRESHOLD

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Traditional Screening (BLAST)")
        st.metric(
            label="DNA Sequence Identity",
            value=f"{blast_score:.1%}",
            delta=f"{'FLAGGED' if blast_flagged else 'BYPASSED threshold'}",
            delta_color="inverse",
        )
        if blast_flagged:
            st.error(f"Flagged — identity ≥ {FLAG_THRESHOLD:.0%} threshold.")
        else:
            st.success(
                f"Evades legacy screening — identity below {FLAG_THRESHOLD:.0%} threshold."
            )

    with col2:
        st.markdown("#### AI Semantic Screening (ESM-2)")
        st.metric(
            label="Cosine Similarity to Wild-Type",
            value=f"{cosine_sim:.4f}",
            delta=f"{'FLAGGED' if semantic_flagged else 'Below threshold'}",
            delta_color="inverse",
        )
        if semantic_flagged:
            st.error(
                f"Flagged — functionally identical to reference "
                f"(sim ≥ {SEMANTIC_THRESHOLD})."
            )
        else:
            st.warning("Similarity below semantic threshold — verify manually.")

    # ── Interpretation ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Interpretation")

    if not blast_flagged and semantic_flagged:
        st.error(
            "Critical gap detected: this sequence **evades legacy screening** "
            "yet the AI model recognises it as functionally equivalent to the "
            "reference toxin. Traditional screening alone would have missed it."
        )
    elif blast_flagged and semantic_flagged:
        st.warning("Both methods flagged this sequence.")
    elif not blast_flagged and not semantic_flagged:
        st.success("Neither method flagged this sequence.")
    else:
        st.info("Results are mixed — review manually.")

    with st.expander("Translated protein sequence"):
        st.code(query_protein, language=None)
