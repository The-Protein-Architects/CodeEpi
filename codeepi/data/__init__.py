"""CodeEpi dataset adapters over the **CodeEpi PyG Release v1.0** dataset.

The release ships two independent PyG datasets:

    Complex1723 : 1,723 antibody-antigen complex-level graphs
    Antigen1321 : 1,321 representative antigen-level graphs

The antigen path is wrapped here into :class:`AntigenReleaseDataset` whose
``Data`` objects match the field names the student trainer / evaluator /
predictor expect. The complex path is consumed directly through the
release's own ``Complex1723Dataset`` in
:mod:`codeepi.teacher.group_dataset`.
"""
from .release import (
    AntigenReleaseDataset,
    detect_release_root,
)

__all__ = [
    "AntigenReleaseDataset",
    "detect_release_root",
]
