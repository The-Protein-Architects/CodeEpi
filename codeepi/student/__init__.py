"""``codeepi.student`` — antigen-only epitope predictor.

Public API
----------

``CodeEpiStudent``
    The full student architecture. Consumes antigen-only graphs
    (``x_esmif``, ``x_esmc``, ``pos``, ``rsa``, ``edge_index``, ``edge_attr``)
    and returns residue-level backbone features.

``ProtoHead``
    Codebook-consuming head that projects the backbone features to a
    prototype space, computes a temperature-softmaxed similarity to the
    ``K``-way teacher-derived codebook, and returns the expected bin score
    as the per-residue epitope probability.

``afl_p``
    Asymmetric focal loss operating directly on probabilities (with label
    smoothing), used by the full-student trainer.

``EMA``
    Parameter-space exponential moving average wrapper.

``cosine_tau``
    Cosine schedule from ``tau_init`` to ``tau_final`` over training.
"""
from .model import CodeEpiStudent, ProtoHead
from .losses import EMA, afl_p, cosine_tau

__all__ = ["CodeEpiStudent", "ProtoHead", "afl_p", "EMA", "cosine_tau"]
