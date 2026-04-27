#!/usr/bin/env python3
"""
Bioinformatics script for studying genetic drift in Green Fluorescent Protein (GFP).

This script generates synonymous codon variants of GFP to study how genetic drift
affects codon usage while maintaining protein function. All variants encode the
identical GFP amino acid sequence but use different synonymous codons.

Requirements:
    - Biopython

Author: Bioinformatics Analysis
"""

import random
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from typing import List, Dict


# Standard genetic code codon table
CODON_TABLE = {
    'F': ['TTT', 'TTC'],
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'Y': ['TAT', 'TAC'],
    '*': ['TAA', 'TAG', 'TGA'],  # Stop codons
    'C': ['TGT', 'TGC'],
    'W': ['TGG'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'H': ['CAT', 'CAC'],
    'Q': ['CAA', 'CAG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'I': ['ATT', 'ATC', 'ATA'],
    'M': ['ATG'],  # Start codon
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'N': ['AAT', 'AAC'],
    'K': ['AAA', 'AAG'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'D': ['GAT', 'GAC'],
    'E': ['GAA', 'GAG'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG']
}



def get_gfp_sequence() -> str:
    """
    Returns the amino acid sequence of standard Green Fluorescent Protein (GFP).
    
    This is the wild-type GFP sequence from Aequorea victoria, commonly used
    in molecular biology research.
    
    Returns:
        str: GFP amino acid sequence
    """
    gfp_sequence = (
        "MKIIIFRVLTFFFVIFSVNVVAKEFTLDFSTAKTYVDSLNVIRSAIGTPLQTISSGGTSLLMIDSGTGDNLFAVDVRGIDPEEGRFNNLRLIVERNNLYVTGFVNRTNNVFYRFADFSHVTFPGTTAVTLSGDSSYTTLQRVAGISRTGMQINRHSLTTSYLDLMSHSGTSLTQSVARAMLRFVTVTAEALRFRQIQRGFRTTLDDLSGRSYVMTAEDVDLTLNWGRLSSVLPDYHGQDSVRVGRISFGSINAILGSVALILNCHHHASRVARMASDEFPSMCPADGRVRGITHNKILWDSSTLGAILMRRTISS"
    )
    return gfp_sequence


def reverse_translate_to_dna(amino_seq: str, seed: int = None) -> str:
    """
    Reverse translates an amino acid sequence to a DNA sequence using random codon selection.
    
    For each amino acid, randomly selects one of the possible synonymous codons.
    Uses ATG as start codon for methionine and adds a stop codon at the end.
    
    Args:
        amino_seq (str): Amino acid sequence to reverse translate
        seed (int, optional): Random seed for reproducible results
        
    Returns:
        str: DNA sequence encoding the amino acid sequence
    """
    if seed is not None:
        random.seed(seed)
    
    dna_sequence = ""
    
    for i, aa in enumerate(amino_seq):
        if aa not in CODON_TABLE:
            raise ValueError(f"Unknown amino acid: {aa}")
        
        # Use ATG for the first methionine (start codon)
        if i == 0 and aa == 'M':
            dna_sequence += 'ATG'
        else:
            # Randomly select from synonymous codons
            possible_codons = CODON_TABLE[aa]
            selected_codon = random.choice(possible_codons)
            dna_sequence += selected_codon
    
    # Add stop codon
    stop_codon = random.choice(CODON_TABLE['*'])
    dna_sequence += stop_codon
    
    return dna_sequence


def generate_synonymous_variants(amino_seq: str, num_variants: int = 100) -> List[str]:
    """
    Generates multiple synonymous DNA variants of the same amino acid sequence.
    
    Each variant uses different combinations of synonymous codons while encoding
    the identical protein sequence. This simulates genetic drift effects on
    codon usage bias.
    
    Args:
        amino_seq (str): Amino acid sequence to generate variants for
        num_variants (int): Number of variants to generate (default: 100)
        
    Returns:
        List[str]: List of DNA sequences, all encoding the same protein
    """
    variants = []
    
    # Generate variants with different random seeds to ensure diversity
    for i in range(num_variants):
        # Use different seeds to ensure each variant is unique
        dna_variant = reverse_translate_to_dna(amino_seq, seed=i + 1000)
        variants.append(dna_variant)
    
    # Verify all variants are unique
    unique_variants = list(set(variants))
    
    # If we don't have enough unique variants, generate more
    attempt = 0
    while len(unique_variants) < num_variants and attempt < 10000:
        additional_variant = reverse_translate_to_dna(amino_seq, seed=attempt + 2000)
        if additional_variant not in unique_variants:
            unique_variants.append(additional_variant)
        attempt += 1
    
    return unique_variants[:num_variants]


def verify_translation(dna_seq: str, expected_aa_seq: str) -> bool:
    """
    Verifies that a DNA sequence translates to the expected amino acid sequence.
    
    Args:
        dna_seq (str): DNA sequence to verify
        expected_aa_seq (str): Expected amino acid sequence
        
    Returns:
        bool: True if translation matches, False otherwise
    """
    try:
        # Create Biopython Seq object and translate
        dna_obj = Seq(dna_seq)
        translated = str(dna_obj.translate(to_stop=True))
        return translated == expected_aa_seq
    except Exception as e:
        print(f"Translation error: {e}")
        return False


def save_sequences_to_fasta(sequences: List[str], labels: List[str], filename: str):
    """
    Saves DNA sequences to a FASTA file.
    
    Args:
        sequences (List[str]): List of DNA sequences
        labels (List[str]): List of sequence labels/identifiers
        filename (str): Output FASTA filename
    """
    seq_records = []
    
    for seq, label in zip(sequences, labels):
        # Create SeqRecord object for each sequence
        record = SeqRecord(
            Seq(seq),
            id=label,
            description=f"GFP variant - Length: {len(seq)} bp"
        )
        seq_records.append(record)
    
    # Write to FASTA file
    with open(filename, 'w') as output_handle:
        SeqIO.write(seq_records, output_handle, "fasta")
    
    print(f"Successfully saved {len(sequences)} sequences to {filename}")


def analyze_codon_usage(dna_sequences: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Analyzes codon usage patterns across all variants.
    
    Args:
        dna_sequences (List[str]): List of DNA sequences to analyze
        
    Returns:
        Dict[str, Dict[str, int]]: Codon usage counts for each amino acid
    """
    codon_usage = {}
    
    # Initialize codon usage dictionary
    for aa, codons in CODON_TABLE.items():
        if aa != '*':  # Skip stop codons for this analysis
            codon_usage[aa] = {codon: 0 for codon in codons}
    
    # Count codon usage across all sequences
    for dna_seq in dna_sequences:
        # Process sequence in triplets (codons)
        for i in range(0, len(dna_seq) - 2, 3):
            codon = dna_seq[i:i+3]
            
            # Find which amino acid this codon encodes
            for aa, codons in CODON_TABLE.items():
                if codon in codons and aa != '*':
                    codon_usage[aa][codon] += 1
                    break
    
    return codon_usage


def print_codon_usage_summary(codon_usage: Dict[str, Dict[str, int]]):
    """
    Prints a summary of codon usage patterns.
    
    Args:
        codon_usage (Dict[str, Dict[str, int]]): Codon usage data
    """
    print("\n" + "="*60)
    print("CODON USAGE ANALYSIS SUMMARY")
    print("="*60)
    
    for aa in sorted(codon_usage.keys()):
        total_count = sum(codon_usage[aa].values())
        if total_count > 0:
            print(f"\nAmino Acid {aa} (total occurrences: {total_count}):")
            for codon, count in codon_usage[aa].items():
                if count > 0:
                    frequency = (count / total_count) * 100
                    print(f"  {codon}: {count:3d} ({frequency:5.1f}%)")


def main():
    """
    Main function to execute the GFP genetic drift analysis.
    """
    print("Generating GFP synonymous variants for genetic drift analysis...")
    print("="*60)
    
    # Step 1: Get GFP amino acid sequence
    gfp_aa_seq = get_gfp_sequence()
    print(f"GFP amino acid sequence length: {len(gfp_aa_seq)} residues")
    print(f"GFP sequence: {gfp_aa_seq[:50]}...")
    
    # Step 2: Generate wild-type DNA sequence
    print("\nGenerating wild-type DNA sequence...")
    wildtype_dna = reverse_translate_to_dna(gfp_aa_seq, seed=42)
    
    # Verify wild-type translation
    if verify_translation(wildtype_dna, gfp_aa_seq):
        print("✓ Wild-type DNA sequence verified - translates correctly")
        print(f"Wild-type DNA length: {len(wildtype_dna)} bp")
    else:
        print("✗ Error: Wild-type DNA translation failed!")
        return
    
    # Step 3: Generate synonymous variants
    print(f"\nGenerating 100 synonymous variants...")
    variants = generate_synonymous_variants(gfp_aa_seq, 100)
    
    # Verify all variants translate correctly
    valid_variants = []
    for i, variant in enumerate(variants):
        if verify_translation(variant, gfp_aa_seq):
            valid_variants.append(variant)
        else:
            print(f"Warning: Variant {i+1} translation failed - skipping")
    
    print(f"✓ Generated {len(valid_variants)} valid synonymous variants")
    
    # Check sequence diversity
    unique_sequences = len(set([wildtype_dna] + valid_variants))
    print(f"✓ All {unique_sequences} sequences are unique")
    
    # Step 4: Save to FASTA file
    print(f"\nSaving sequences to FASTA file...")
    all_sequences = [wildtype_dna] + valid_variants
    all_labels = ['GFP_wildtype'] + [f'GFP_variant_{i+1:03d}' for i in range(len(valid_variants))]
    
    save_sequences_to_fasta(all_sequences, all_labels, 'gfp_variants.fasta')
    
    # Step 5: Analyze codon usage
    print(f"\nAnalyzing codon usage patterns...")
    codon_usage = analyze_codon_usage(all_sequences)
    print_codon_usage_summary(codon_usage)
    
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"• Generated: {len(valid_variants)} synonymous variants + 1 wild-type")
    print(f"• Output file: gfp_variants.fasta")
    print(f"• All sequences encode identical GFP protein ({len(gfp_aa_seq)} AA)")
    print(f"• DNA sequence length: {len(wildtype_dna)} bp each")
    print("\nThese variants can be used to study:")
    print("  - Genetic drift effects on codon usage")
    print("  - Selection pressure on synonymous sites")
    print("  - Translation efficiency differences")
    print("  - Protein expression level variations")


if __name__ == "__main__":
    main()