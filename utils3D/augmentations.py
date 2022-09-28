# standard library imports
from configparser import Interpolation
import random
import numpy as np
import math
import torchvision

# 2D YOLO imports
from utils.general import colorstr, segment2box, resample_segments, check_version

# 3D YOLO imports
from utils3D.lossandmetrics import bbox_iov


def nifti_cutout(im, labels, p=0.5):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    if random.random() < p:
        d, h, w = im.shape[1:]
        
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_d = random.randint(1, int(d * s))  # create random masks
            mask_h = random.randint(1, int(h * s))
            mask_w = random.randint(1, int(w * s))
            
            # box
            zmin = max(0, random.randint(0, d) - mask_d // 2)
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            zmax = min(d, zmin + mask_d)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)
            
            # apply random greyscale mask
            # images scaled between 0 and 1 after being returned by the dataset
            im[zmin:zmax, ymin:ymax, xmin:xmax] = random.randint(-1024, 1024)
            
            if len(labels) and s > 0.03:
                box = np.array([zmin, xmin, ymin, zmax, xmax, ymax], dtype=np.float32)
                iov = bbox_iov(box, labels[:, 1:7])  # intersection over area
                labels = labels[iov < 0.60]  # remove >60% obscured labels
                
    return labels
