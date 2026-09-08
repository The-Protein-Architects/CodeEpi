"""CodeEpi prediction CLIs.

One entry point:

* ``codeepi.predict.student_antigen_ensemble``  -- antigen-only 5-seed ensemble prediction.

Readme on `` /CodeEpi/docs/prediction.md ``

It writes ``residue_predictions.csv``, ``residue_predictions.json``,
``antigen_colored_by_probability.pdb`` (b-factor swap), and
``config_used.yaml`` into the requested output directory, plus
``per_seed_probabilities.csv`` for auditing.
"""
from . import postprocess

__all__ = ["postprocess"]
