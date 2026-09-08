"""CodeEpi teacher-derived 4-bin codebook.

The codebook is a set of ``K=4`` row-normalized prototypes in 256-d space,
one per teacher-probability bin, initialized from the teacher's post-QKV 
antibody-context antigen residue representations. 
It is the knowledge the CodeEpi student inherits from the teacher, 
and is the *only* teacher-side artifact needed by student inference.
"""
from .build_codebook import (
    bin_score_from_cut_edges,
    build_codebook_from_residue_table,
)
from .utils import default_bin_edges, load_codebook

__all__ = [
    "bin_score_from_cut_edges",
    "build_codebook_from_residue_table",
    "default_bin_edges",
    "load_codebook",
]
