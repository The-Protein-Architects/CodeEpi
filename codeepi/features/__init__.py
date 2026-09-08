"""Feature extraction for prediction.

These modules are only used by the prediction CLI
(``codeepi.predict.student_antigen_ensemble``). Training and reproduction
consume pre-computed features from ``codeepi_pyg_release_v1.0`` and do not
import anything here.

Public helpers:

    pdb_parser.parse_pdb              -- Bio.PDB scan of a PDB file
    fasta.read_fasta                  -- FASTA reader (SEQRES authority for ESM-C)
    seqmap_align.map_structure_to_seqres -- anchor atm->SEQRES reconciliation
    surface.compute_surface           -- surface residues (DSSP/RSA-based)
    rsa.compute_rsa                   -- RSA per residue (DSSP)
    esmif_antigen.encode_antigen      -- ESM-IF antigen-only encoding
    esmc.load_precomputed             -- load a precomputed ESM-C .pt
    encode_esmc                       -- standalone ESM-C API encoder
                                         (run in the separate codeepi-esmc env)
"""
import importlib

__all__ = [
    "esmc",
    "encode_esmc",
    "esmif_antigen",
    "fasta",
    "pdb_parser",
    "rsa",
    "seqmap_align",
    "surface",
]


def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
