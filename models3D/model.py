"""
Model definition script for 3D YOLO.  Defines the modules and the overall model.
Modified with extra detection layer
Mostly contains 3D versions of code from models/yolo.py and models/common.py
"""

# standard library imports
import sys
import torch
from pathlib import Path
import os
import logging
import warnings
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math

# 2D YOLO imports
from models.common import autopad, Concat
from utils.general import make_divisible

# 3D YOLO imports
from utils3D.anchors import check_anchor_order

