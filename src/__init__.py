"""
Called when the package is imported.
"""

# pylint: disable=wrong-import-position
print(f"Entered {__file__=}")
# pylint: enable=wrong-import-position

import random

import numpy as np
import torch


random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)
