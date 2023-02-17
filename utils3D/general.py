"""
General utility functions from YOLOv5 utils/general.py that have needed changes to function in 3D.
"""

# standard library imports
import torch
import time
import numpy as np
from pathlib import Path
import yaml

# 2D YOLO imports

# 3D YOLO imports
from utils3D.lossandmetrics import box_iou


# Configuration
default_size = 350 # edge length for testing


def check_dataset(data):
    """Generates the dataset dictionary used during training.
    Has removed download functionality found in YOLOv5 version.

    Args:
        data (str, pathlib.Path, or Dict): path to dataset yaml file or a dictionary containing similar information.

    Raises:
        Exception: if the validation dataset is not found.

    Returns:
        data (Dict): dictionary containing information in the dataset yaml file.
    """
    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        with open(data, errors='ignore') as f:
            data = yaml.safe_load(f)  # dictionary

    # Parse yaml
    path = Path(data.get('path') or '')  # optional 'path' default to '.'
    for k in 'train', 'val', 'test':
        if data.get(k):  # prepend path
            data[k] = str(path / data[k]) if isinstance(data[k], str) else [str(path / x) for x in data[k]]

    assert 'nc' in data, "Dataset 'nc' key missing."
    if 'names' not in data:
        data['names'] = [f'class{i}' for i in range(data['nc'])]  # assign class names if missing
    train, val, test = (data.get(x) for x in ('train', 'val', 'test'))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            raise Exception('Dataset not found.')
    return data  # dictionary


def zxydwhn2zxyzxy(labels, d=default_size, w=default_size, h=default_size, padd=0, padw=0, padh=0):
    """Convert nx6 boxes from [z, x, y, d, w, h] normalized to [z1, x1, y1, z2, x2, y2] where zxy1=top-left, zxy2=bottom-right

    Args:
        labels (torch.tensor or np.ndarray): normalized image labels in [z, x, y, d, w, h] format
        d (int, optional): Unnormalized image depth.
        w (int, optional): Unnormalized image width.
        h (int, optional): Unnormalized image height.
        padd (int, optional): depth padding to add to each side. Defaults to 0.
        padw (int, optional): width padding to add to each side. Defaults to 0.
        padh (int, optional): height padding to add to each side. Defaults to 0.
    Returns:
        y (torch.tensor or np.ndarray): unnormalized image labels in [z1, x1, y1, z2, x2, y2] format
    """
    y = labels.clone() if isinstance(labels, torch.Tensor) else np.copy(labels)
    y[:, 0] = d * (labels[:, 0] - labels[:, 3] / 2) + padd  # top left z
    y[:, 1] = w * (labels[:, 1] - labels[:, 4] / 2) + padw  # top left x
    y[:, 2] = h * (labels[:, 2] - labels[:, 5] / 2) + padh  # top left y
    y[:, 3] = d * (labels[:, 0] + labels[:, 3] / 2) + padd  # bottom right z
    y[:, 4] = w * (labels[:, 1] + labels[:, 4] / 2) + padw  # bottom right x
    y[:, 5] = h * (labels[:, 2] + labels[:, 5] / 2) + padh  # bottom right y
    return y


def zxydwh2zxyzxy(labels):
    """Convert nx6 boxes from [z, x, y, d, w, h] to [z1, x1, y1, z2, x2, y2] where zxy1=top-left, zxy2=bottom-right

    Args:
        labels (torch.tensor or np.ndarray): unnormalized image labels in [z, x, y, d, w, h] format

    Returns:
        y (torch.tensor or np.ndarray): unnormalized image labels in [z1, x1, y1, z2, x2, y2] format
    """
    y = labels.clone() if isinstance(labels, torch.Tensor) else np.copy(labels)
    y[:, 0] = (labels[:, 0] - labels[:, 3] / 2) # top left z
    y[:, 1] = (labels[:, 1] - labels[:, 4] / 2) # top left x
    y[:, 2] = (labels[:, 2] - labels[:, 5] / 2) # top left y
    y[:, 3] = (labels[:, 0] + labels[:, 3] / 2) # bottom right z
    y[:, 4] = (labels[:, 1] + labels[:, 4] / 2) # bottom right x
    y[:, 5] = (labels[:, 2] + labels[:, 5] / 2) # bottom right y
    return y


def zxyzxy2zxydwhn(labels, d=default_size, w=default_size, h=default_size, clip=False, eps=0.0):
    """Convert nx6 boxes from [z1, x1, y1, z2, x2, y2] to [z, x, y, d, w, h] normalized where zxy1=top-left, zxy2=bottom-right

    Args:
        labels (torch.tensor or np.ndarray): unnormalized image labels in [z1, x1, y1, z2, x2, y2] format
        d (int, optional): Unnormalized image depth.
        w (int, optional): Unnormalized image width.
        h (int, optional): Unnormalized image height.
        clip (bool, optional): Whether or not coordinates of labels should be clipped to the image shape. Defaults to False.
        eps (float, optional): How many pixels to clip off of each side. Defaults to 0.0.

    Returns:
        y (torch.tensor or np.ndarray): normalized image labels in [z, x, y, d, w, h] format
    """
    if clip:
        clip_coords(labels, (d - eps, w - eps, h - eps))  # warning: inplace clip
    y = labels.clone() if isinstance(labels, torch.Tensor) else np.copy(labels)
    y[:, 0] = ((labels[:, 0] + labels[:, 3]) / 2) / d  # z center
    y[:, 1] = ((labels[:, 1] + labels[:, 4]) / 2) / w  # x center
    y[:, 2] = ((labels[:, 2] + labels[:, 5]) / 2) / h  # y center
    y[:, 3] = (labels[:, 3] - labels[:, 0]) / d  # depth
    y[:, 4] = (labels[:, 4] - labels[:, 1]) / w  # width
    y[:, 5] = (labels[:, 5] - labels[:, 2]) / h  # height
    return y


def zxyzxy2zxydwh(labels):
    """Convert nx6 boxes from [z1, x1, y1, z2, x2, y2] to [z, x, y, d, w, h] where zxy1=top-left, zxy2=bottom-right

    Args:
        labels (torch.tensor or np.ndarray): unnormalized image labels in [z1, x1, y1, z2, x2, y2] format

    Returns:
        y (torch.tensor or np.ndarray): unnormalized image labels in [z, x, y, d, w, h] format
    """
    # Convert nx4 boxes from [z1, x1, y1, z2, x2, y2] to [z, x, y, d, w, h] where zxy1=top-left, zxy2=bottom-right
    y = labels.clone() if isinstance(labels, torch.Tensor) else np.copy(labels)
    y[:, 0] = ((labels[:, 0] + labels[:, 3]) / 2)  # z center
    y[:, 1] = ((labels[:, 1] + labels[:, 4]) / 2)  # x center
    y[:, 2] = ((labels[:, 2] + labels[:, 5]) / 2)  # y center
    y[:, 3] = (labels[:, 3] - labels[:, 0])  # depth
    y[:, 4] = (labels[:, 4] - labels[:, 1])  # width
    y[:, 5] = (labels[:, 5] - labels[:, 2])  # height
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale bounding box coordinates (zxyzxy) from img1_shape to img0_shape.

    Args:
        img1_shape (Tuple[int]): size of the resized image that predictions were made on.
        coords (torch.tensor): bounding box coordinates to scale.
        img0_shape (Tuple[int]): size of original image to scale coordinates to.
        ratio_pad (Tuple[Tuple[float]], optional): ((d/d0, h/h0, w/w0), (padd, padx, pady)) Alternate way to determine gain and padding for rescaling. Defaults to None.

    Returns:
        coords (torch.tensor): scaled bounding box coordinates.
    """
    if ratio_pad is None: # calculate from img0_shape
        gainz = img1_shape[0] / img0_shape[0]  # gain = old / new
        gainx = img1_shape[1] / img0_shape[1]
        gainy = img1_shape[2] / img0_shape[2]
        padz = (img1_shape[0] - img0_shape[0] * gainz) / 2  # wh padding
        padx = (img1_shape[1] - img0_shape[1] * gainx) / 2
        pady = (img1_shape[2] - img0_shape[2] * gainy) / 2
    else:
        gainz = ratio_pad[0][0]
        padz = ratio_pad[1][0]
        gainx = ratio_pad[0][1]
        padx = ratio_pad[1][1]
        gainy = ratio_pad[0][2]
        pady = ratio_pad[1][2]

    coords[:, [0, 3]] -= padz  # z padding
    coords[:, [1, 4]] -= padx  # x padding
    coords[:, [2, 5]] -= pady  # y padding
    coords[:, 0] /= gainz
    coords[:, 3] /= gainz
    coords[:, 1] /= gainx
    coords[:, 2] /= gainy
    coords[:, 4] /= gainx
    coords[:, 5] /= gainy
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    """In-place clip zxyzxy bounding boxes to image shape (depth, height, width)

    Args:
        boxes (torch.tensor or np.ndarray): bounding boxes to clip to new shape
        shape (Tuple[int]): new shape bounding boxes should be clipped to
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[0])  # z1
        boxes[:, 1].clamp_(0, shape[1])  # x1
        boxes[:, 2].clamp_(0, shape[2])  # y1
        boxes[:, 3].clamp_(0, shape[0])  # z2
        boxes[:, 4].clamp_(0, shape[1])  # x2
        boxes[:, 5].clamp_(0, shape[2])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 3]] = boxes[:, [0, 3]].clip(0, shape[0])  # z1, z2
        boxes[:, [1, 4]] = boxes[:, [1, 4]].clip(0, shape[1])  # x1, x2
        boxes[:, [2, 5]] = boxes[:, [2, 5]].clip(0, shape[2])  # y1, y2


def _3d_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float):
    """Performs non-maximum suppression (NMS) on bounding boxes according
       to their intersection-over-union (IoU).

       NMS iteratively removes lower scoring boxes which have an
       IoU greater than iou_threshold with another higher scoring
       box.

       If multiple boxes have the exact same score and satisfy the IoU
       criterion with respect to a reference box, the selected box is
       not guaranteed to be the same between CPU and GPU. This is similar
       to the behavior of argsort in PyTorch when repeated values are present.
       
       initially based off: https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/
       and torchvision 2D nms implementation

    Args:
        boxes (torch.Tensor): (Tensor[N, 6])) bounding boxes to perform NMS on.
            They are expected to be in ``(z1, x1, y1, z2, x2, y2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2`` and ``0 <= z1 < z2''.
        scores (torch.Tensor): confidence scores for each of the bounding boxes
        iou_threshold (float): function discards all overlapping boxes with IoU > iou_threshold

    Returns:
        torch.Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted in decreasing order of scores
    """
    # extract coordinates for every prediction box present in boxes
    z1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y1 = boxes[:, 2]
    z2 = boxes[:, 3]
    x2 = boxes[:, 4]
    y2 = boxes[:, 5]

    # calculate volume of every block in P
    volumes = (z2 - z1) * (x2 - x1) * (y2 - y1)

    # sort the prediction boxes in scores according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for filtered prediction boxes
    keep = []

    while len(order) > 0:
        # extract the index of the prediction with highest score
        idx = order[-1]

        # add that box's index to filtered predictions list
        keep.append(idx)

        # remove the box from scores
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # select coordinates of BBoxes according to the indices in order
        zz1 = torch.index_select(z1, dim=0, index=order)
        zz2 = torch.index_select(z2, dim=0, index=order)
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        # find the coordinates of the intersection boxes
        zz1 = torch.max(zz1, z1[idx])
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        zz2 = torch.min(zz2, z2[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        d = zz2 - zz1
        w = xx2 - xx1
        h = yy2 - yy1

        # clamp minimum with 0.0 to avoid negative w and h due to non-overlapping boxes
        d = torch.clamp(d, min=0.0)
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection volume
        inter = d * w * h

        # find the volumes of BBoxes according to the indices in order
        rem_volumes = torch.index_select(volumes, dim=0, index=order)

        # find the union of every box in boxes with the box currently being tested
        # Note that volumes[idx] represents the volume of the current box
        union = (rem_volumes - inter) + volumes[idx]

        # find the IoU of every prediction in boxes with the current box
        IoU = inter / union

        # keep the boxes with IoU less than thresh_iou
        mask = IoU < iou_threshold
        order = order[mask]

    return torch.stack(keep, dim=0)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results for 3D bounding box predictions.

    Args:
        prediction (torch.tensor): tensor of predictions, with shape (batch, num predictions, 7+num_classes) per image where 7 is for zxyzxy and confidence score
        conf_thres (float, optional): Minimum confidence score required for a prediction to be considered. Defaults to 0.25.
        iou_thres (float, optional): Maximum IOU overlap allowed by non-max suppression. Defaults to 0.45.
        classes (List[int], optional): Classes to run NMS on. Defaults to None (no filtering).
        agnostic (bool, optional): Whether to use class agnostic NMS. Defaults to False.
        multi_label (bool, optional): Whether or not to allow multiple labels per box. Defaults to False.
        labels (tuple, optional): Labels to use if autolabelling. Defaults to ().
        max_det (int, optional): Maximum number of detections to allow. Defaults to 300.

    Returns:
        output (torch.tensor): list of detections, on (n,8) tensor per image [zxyzxy, conf, cls]
    """
    nc = prediction.shape[2] - 7  # number of classes
    xc = prediction[..., 6] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_dwh, max_dwh = 4, 41943040  # (pixels) minimum and maximum box depth*width*height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 8), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 3:6] < min_dwh) | (x[..., 3:6] > max_dwh)).any(1), 6] = 0  # depth-width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 7), device=x.device)
            v[:, :6] = l[:, 1:7]  # box
            v[:, 6] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 7] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 7:] *= x[:, 6:7]  # conf = obj_conf * cls_conf

        # Box (center z, center x, center y, depth, width, height) to (z1, x1, y1, z2, x2, y2)
        box = zxydwh2zxyzxy(x[:, :6])

        # Detections matrix nx8 (zxyzxy, conf, cls)
        if multi_label:
            i, j = (x[:, 7:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 7, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 7:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 7:8] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 6].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 7:8] * (0 if agnostic else max_dwh)  # classes
        boxes, scores = x[:, :6] + c, x[:, 6]  # boxes (offset by class), scores
        i = _3d_nms(boxes, scores, iou_thres)  # 3D NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,6) = weights(i,n) * boxes(n,6)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :6] = torch.mm(weights, x[:, :6]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output
