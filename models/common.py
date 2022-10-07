# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

# import logging
# import math
# import warnings
# from copy import copy
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import requests
import torch
import torch.nn as nn
# from PIL import Image
# from torch.cuda import amp

# from utils.datasets import exif_transpose, letterbox
# from utils.general import colorstr, increment_path, make_divisible, non_max_suppression, save_one_box, \
#     scale_coords, xyxy2xywh
# from utils.plots import Annotator, colors
# from utils.torch_utils import time_sync

# LOGGER = logging.getLogger(__name__)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
