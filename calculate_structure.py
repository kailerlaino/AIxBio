"""
Structural comparison of GFP variant PDB files against the wild-type reference.

Uses alpha-carbon (CA) superimposition and RMSD to quantify structural deviation.

── Colab / fresh-environment install ────────────────────────────────────────────
    pip install biotite
─────────────────────────────────────────────────────────────────────────────────
"""

import os
import sys

import biotite.structure as struc
import biotite.structure.io.pdb as pdb


PDB_DIR      = "pdb_files"
WILDTYPE_PDB = os.path.join(PDB_DIR, "wildtype.pdb")
NUM_VARIANTS = 5


def load_ca(pdb_path: str) -> struc.AtomArray:
    """Return the CA-only AtomArray from model 1 of a PDB file."""
    f = pdb.PDBFile.read(pdb_path)
    structure = f.get_structure(model=1)
    ca = structure[structure.atom_name == "CA"]
    if len(ca) == 0:
        raise ValueError(f"No CA atoms found in {pdb_path}")
    return ca


def compare(wt_ca: struc.AtomArray, variant_path: str) -> float:
    """
    Superimpose variant CA atoms onto the wild-type and return the RMSD in Ångströms.
    Raises ValueError if the atom counts don't match (different sequence lengths).
    """
    var_ca = load_ca(variant_path)

    if len(var_ca) != len(wt_ca):
        raise ValueError(
            f"CA atom count mismatch: wild-type has {len(wt_ca)}, "
            f"{os.path.basename(variant_path)} has {len(var_ca)}"
        )

    fitted, _ = struc.superimpose(wt_ca, var_ca)
    return float(struc.rmsd(wt_ca, fitted))


def main():
    if not os.path.isfile(WILDTYPE_PDB):
        sys.exit(f"Error: wild-type PDB not found at '{WILDTYPE_PDB}'. "
                 "Run generate_pdb.py first.")

    wt_ca = load_ca(WILDTYPE_PDB)
    print(f"Wild-type loaded: {len(wt_ca)} CA atoms\n")

    header   = f"{'Variant':<20}  {'RMSD (Å)':>10}"
    divider  = "-" * len(header)
    print(header)
    print(divider)

    results = []
    for i in range(1, NUM_VARIANTS + 1):
        name = f"variant_{i:03d}"
        path = os.path.join(PDB_DIR, f"{name}.pdb")

        if not os.path.isfile(path):
            print(f"{name:<20}  {'FILE NOT FOUND':>10}")
            continue

        try:
            rmsd_val = compare(wt_ca, path)
            results.append((name, rmsd_val))
            print(f"{name:<20}  {rmsd_val:>10.4f}")
        except ValueError as e:
            print(f"{name:<20}  ERROR: {e}")

    if results:
        rmsds = [r for _, r in results]
        print(divider)
        print(f"{'Mean RMSD':<20}  {sum(rmsds)/len(rmsds):>10.4f}")
        print(f"{'Min  RMSD':<20}  {min(rmsds):>10.4f}  ({results[rmsds.index(min(rmsds))][0]})")
        print(f"{'Max  RMSD':<20}  {max(rmsds):>10.4f}  ({results[rmsds.index(max(rmsds))][0]})")


if __name__ == "__main__":
    main()
