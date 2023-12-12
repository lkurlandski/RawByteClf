"""
Called when the package is imported.
"""

print(f"Entered {__file__=}")

import random

import numpy as np
import torch


random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)
