"""
Tüm random seed'leri ayarlar.
Notebook Cell 1'den taşındı.
"""

import random
import numpy as np
import torch
from config.settings import SEED


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
