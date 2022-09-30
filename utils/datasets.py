# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

# import glob
import hashlib
# import json
# import logging
import os
# import random
# import shutil
# import time
# from itertools import repeat
# from multiprocessing.pool import ThreadPool, Pool
# from pathlib import Path
# from threading import Thread
# from zipfile import ZipFile

# import cv2
# import numpy as np
import torch
# import torch.nn.functional as F
# import yaml
# from PIL import Image, ImageOps, ExifTags
# from torch.utils.data import Dataset
# from tqdm import tqdm

# from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
# from utils.general import check_dataset, check_requirements, check_yaml, clean_str, segments2boxes, \
#     xywh2xyxy, xywhn2xyxy, xyxy2xywhn, xyn2xy
# from utils.torch_utils import torch_distributed_zero_first


# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
            
            
class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
