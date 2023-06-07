"""
Augmentations used by 3D YOLO    
"""


# standard library imports
import random
import numpy as np
import torch
from typing import List

# 2D YOLO imports

# 3D YOLO imports
from utils3D.lossandmetrics import bbox_iov


def tensor_cutout(im: torch.Tensor, labels, cutout_params: List[List[float]], p=0.5):
    """Applies image cutout augmentation https://arxiv.org/abs/1708.04552

    Args:
        im (torch.Tensor): 3-D tensor to be augmented.
        labels (List[float]): Med YOLO labels corresponding to im. class z1 x1 y1 z2 x2 y2
        cutout_params (List[List[float]]): a list of ordered pairs that set sizes and numbers of cutout boxes.
                                           the maximum extent of the boxes is set by the first element of the ordered pair
                                           the number of boxes with that maximum extent is set by the second element of the ordered pair
        p (float, optional): probability of performing cutout augmentation. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    if random.random() < p:
        d, h, w = im.shape[1:]
        scales = []
        for param_pair in cutout_params:
            scales = scales + [param_pair[0]]*param_pair[1]
       
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
            im[:,zmin:zmax, ymin:ymax, xmin:xmax] = random.uniform(torch.min(im), torch.max(im))
            
            # remove obscured labels
            if len(labels) and s > 0.03:
                box = np.array([zmin, xmin, ymin, zmax, xmax, ymax], dtype=np.float32)
                iov = bbox_iov(box, labels[:, 1:7])  # intersection over volume
                labels = labels[iov < 0.60]  # remove >60% obscured labels
                
    return im, labels


def random_zoom(im: torch.Tensor, labels, max_zoom=1.5, min_zoom=0.7, p=0.5):
    """Randomly zooms in or out of a random part of the input image.

    Args:
        im (torch.Tensor): 3-D tensor to be augmented.
        labels (List[float]): Med YOLO labels corresponding to im. class z1 x1 y1 z2 x2 y2
        max_zoom (float, optional): maximum edge length multiplier. Defaults to 1.5.
        min_zoom (_type_, optional): minimum edge length multiplier. Defaults to 0.7.
        p (float, optional): probability of zooming the input. Defaults to 0.5.

    Returns:
        im: Augmented tensor.
        y: Adjusted labels.
    """
    
    y = labels.clone() if isinstance(labels, torch.Tensor) else np.copy(labels)
    
    if random.random() < p:
        # retrieve original image shape (this is resized to imgsz x imgsz x imgsz by this point in the dataloader)
        d, w, h = im.shape[1:]
        
        # setting limits for how far augmentation will zoom in or out
        max_zoom_factor = max_zoom
        min_zoom_factor = min_zoom
        
        # choosing the zoom level of the final image
        zoom_factor = random.uniform(min_zoom_factor, max_zoom_factor)

        # add batch dimension for functional interpolate
        im = torch.unsqueeze(im, 0)
        # rescale image to its zoomed size
        im = torch.nn.functional.interpolate(im, scale_factor=zoom_factor, mode='trilinear', align_corners=False)
        # remove batch dimension for compatibility with later code
        im = torch.squeeze(im, 0)

        # retrieve new image shape
        new_d, new_w, new_h = im.shape[1:]
        
        # shrink/expand labels
        y[:, 1:7] = y[:, 1:7]*zoom_factor
        
        # crop/pad the zoomed image back to the input size and position it randomly relative to the new tensor
        if zoom_factor >= 1.:
            # new side lengths longer than original side lengths
            # crop to original im.shape (center needs at least original_length/2 distance to each edge to preserve original image)
            zoom_center_d = random.randint(d//2, new_d-d//2)
            zoom_center_h = random.randint(h//2, new_h-h//2)
            zoom_center_w = random.randint(w//2, new_w-w//2)
            
            zmin = zoom_center_d - d//2
            xmin = zoom_center_w - w//2
            ymin = zoom_center_h - h//2
            zmax = zmin + d
            xmax = xmin + w
            ymax = ymin + h
            
            im = im[:,zmin:zmax, xmin:xmax, ymin:ymax]

            # move labels to correspond to new center of zoom
            y[:, 1] = y[:, 1] - zmin
            y[:, 4] = y[:, 4] - zmin
            y[:, 2] = y[:, 2] - xmin
            y[:, 5] = y[:, 5] - xmin
            y[:, 3] = y[:, 3] - ymin
            y[:, 6] = y[:, 6] - ymin
            
            # crop labels beyond bounds of new image
            if isinstance(y, torch.Tensor):  # faster individually
                y[:, 0].clamp_(0, d)  # z1
                y[:, 1].clamp_(0, w)  # x1
                y[:, 2].clamp_(0, h)  # y1
                y[:, 3].clamp_(0, d)  # z2
                y[:, 4].clamp_(0, w)  # x2
                y[:, 5].clamp_(0, h)  # y2
            else:  # np.array (faster grouped)
                y[:, [0, 3]] = y[:, [0, 3]].clip(0, d)  # z1, z2
                y[:, [1, 4]] = y[:, [1, 4]].clip(0, w)  # x1, x2
                y[:, [2, 5]] = y[:, [2, 5]].clip(0, h)  # y1, y2
             
        else:
            # new side lengths shorter than original side lengths
            # pad to original image shape
            zoom_center_d = random.randint(new_d//2 + 1, d-new_d//2 - 1)
            zoom_center_h = random.randint(new_h//2 + 1, h-new_h//2 - 1)
            zoom_center_w = random.randint(new_w//2 + 1, w-new_w//2 - 1)
                
            zmin = zoom_center_d - new_d//2
            xmin = zoom_center_w - new_w//2
            ymin = zoom_center_h - new_h//2
            
            zmax = zmin + new_d
            xmax = xmin + new_w
            ymax = ymin + new_h
            
            # create a new tensor 
            new_im = torch.rand(1, d, w, h)*(torch.max(im) - torch.min(im)) + torch.min(im)
            new_im[:,zmin:zmax, xmin:xmax, ymin:ymax] = im
            im = new_im
            del new_im
            
            # move labels to correspond to new center of zoom
            y[:, 1] = y[:, 1] + zmin
            y[:, 4] = y[:, 4] + zmin
            y[:, 2] = y[:, 2] + xmin
            y[:, 5] = y[:, 5] + xmin
            y[:, 3] = y[:, 3] + ymin
            y[:, 6] = y[:, 6] + ymin

    return im, y
