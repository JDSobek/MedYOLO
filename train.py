"""
Training script for 3D YOLO.
"""

# standard library imports
import argparse
from copy import deepcopy
import logging
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 2D YOLO imports
from utils.general import methods, colorstr, labels_to_class_weights, increment_path, set_logging, print_args, \
    check_yaml, check_file, get_latest_run, one_cycle, print_mutation, strip_optimizer
from utils.callbacks import Callbacks
from utils.torch_utils import select_device, de_parallel, EarlyStopping, ModelEMA, torch_distributed_zero_first
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.loggers import Loggers

# 3D YOLO imports
from models.model import Model, attempt_load
from utils3D.datasets import nifti_dataloader, normalize_CT
from utils3D.lossandmetrics import ComputeLossVF
from utils3D.anchors import nifti_check_anchors
from utils3D.general import check_dataset

import val
