"""Deterministic-seed helpers used by every trainer / builder / predictor."""
from __future__ import annotations
import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
