"""CodeEpi: teacher-codebook-student framework for B-cell epitope prediction.

Top-level package. Submodules:
    codeepi.data     -- CodeEpi PyG Release v1.0 dataset adapters
    codeepi.teacher  -- teacher model + trainer
    codeepi.codebook -- teacher-derived 4-bin codebook
    codeepi.student  -- student model + trainer
    codeepi.predict  -- prediction CLI (student_antigen antigen-only)
    codeepi.features -- PDB / ESM-IF / ESM-C feature extraction (for prediction)
    codeepi.utils    -- seed / io / graph / metrics helpers
"""
__version__ = "1.0.0"
