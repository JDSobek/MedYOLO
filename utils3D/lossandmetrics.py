"""
Code used for loss functions and metrics for 3D YOLO.
"""

# standard library imports
import torch
import torch.nn as nn
import numpy as np
import warnings
from pathlib import Path
import matplotlib.pyplot as plt

# 2D YOLO imports
from utils.loss import smooth_BCE, FocalLoss
from utils.torch_utils import is_parallel

# 3D YOLO imports

# Configuration
default_size = 350 # edge length for testing


def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = torch.nn.functional._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def bbox_iov(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 volume given box1, box2. Boxes are z1x1y1z2x2y2
    box1:       np.array of shape(6)
    box2:       np.array of shape(nx6)
    returns:    np.array of shape(n)
    """
    
    box2 = box2.transpose()
    
    # Get the coordinates of bounding boxes
    b1_z1, b1_x1, b1_y1, b1_z2, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3], box1[4], box1[5]
    b2_z1, b2_x1, b2_y1, b2_z2, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3], box2[4], box2[5]
    
    # Intersection volume
    inter_vol = (np.minimum(b1_z2, b2_z2) - np.maximum(b1_z1, b2_z1)).clip(0) * \
                (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)
                 
    # box2 vol
    box2_vol = (b2_z2 - b2_z1) * (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 vol
    return inter_vol / box2_vol


def bbox_iou(box1, box2, z1x1y1z2x2y2=True, eps=1e-7):
    """Returns the IoU of box1 to box2. box1 is 6, box2 is nx6.
    
    Args:
        box1 (torch.Tensor): First bounding box to calculate IoU for.
        box2 (torch.Tensor): Second bounding box to calculate IoU for.
        z1x1y1z2x2y2 (bool, optional): Whether or not the input boxes are in z1, x1, y1, z2, x2, y2 format. Defaults to True.
        eps (float, optional): Smoothing factor, prevents divide by zero errors. Defaults to 1e-7.

    Returns:
        iou (float): Calculated intersection over union
    """   
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if z1x1y1z2x2y2:  # z1, x1, y1, z2, x2, y2 = box1
        b1_z1, b1_x1, b1_y1, b1_z2, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3], box1[4], box1[5]
        b2_z1, b2_x1, b2_y1, b2_z2, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3], box2[4], box2[5]
    else:  # transform from zxydwh to zxyzxy
        b1_z1, b1_z2 = box1[0] - box1[3] / 2, box1[0] + box1[3] / 2
        b1_x1, b1_x2 = box1[1] - box1[4] / 2, box1[1] + box1[4] / 2
        b1_y1, b1_y2 = box1[2] - box1[5] / 2, box1[2] + box1[5] / 2
        b2_z1, b2_z2 = box2[0] - box2[3] / 2, box2[0] + box2[3] / 2
        b2_x1, b2_x2 = box2[1] - box2[4] / 2, box2[1] + box2[4] / 2
        b2_y1, b2_y2 = box2[2] - box2[5] / 2, box2[2] + box2[5] / 2

    # Intersection volume
    inter = (torch.min(b1_z2, b2_z2) - torch.max(b1_z1, b2_z1)).clamp(0) * \
            (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union volume
    d1, w1, h1 = b1_z2 - b1_z1, b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    d2, w2, h2 = b2_z2 - b2_z1, b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = d1 * w1 * h1 + d2 * w2 * h2 - inter + eps

    iou = inter / union
    return iou


def box_iou(box1, box2):
    """Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (z1, x1, y1, z2, x2, y2) format.
    https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): (Tensor[N, 6])
        box2 (torch.Tensor): (Tensor[M, 6])

    Returns:
        iou (torch.Tensor): (Tensor[N, M]) the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """

    def box_volume(box):
        # box is nx6
        return (box[3] - box[0]) * (box[4] - box[1]) * (box[5] - box[2])

    vol1 = box_volume(box1.T)
    vol2 = box_volume(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 3:], box2[:, 3:]) - torch.max(box1[:, None, :3], box2[:, :3])).clamp(0).prod(2)
    return inter / (vol1[:, None] + vol2 - inter)  # iou = inter / (area1 + area2 - inter)


@torch.jit.script
def bbox_centerDist(box1, box2, z1x1y1z2x2y2: bool =True):
    """Measures the distance between the centers of two bounding boxes

    Args:
        box1 (torch.Tensor): First bounding box to calculate distance for.
        box2 (torch.Tensor): Second bounding box to calculate distance for.
        z1x1y1z2x2y2 (bool, optional): Whether or not the input boxes are in z1, x1, y1, z2, x2, y2 format. Defaults to True.

    Returns:
        dist (float): Calculated distance between the centers of the two bounding boxes
    """
    box2 = box2.T
    
    # refactor to get center positions
    # Get the coordinates of bounding boxes
    if z1x1y1z2x2y2:  # z1, x1, y1, z2, x2, y2 = box1 : need to get centers of each box
        z1, x1, y1 = box1[3] - box1[0], box1[4] - box1[1], box1[5] - box1[2]
        z2, x2, y2 = box2[3] - box2[0], box2[4] - box2[1], box2[5] - box2[2]
    else:  # zxydwh
        z1, x1, y1 = box1[0], box1[1], box1[2]
        z2, x2, y2 = box2[0], box2[1], box2[2]
        
    dist = ((z1 - z2).pow(2) + (x1 - x2).pow(2) + (y1 - y2).pow(2)).sqrt() 
    return dist


class ComputeLossVF:
    """Compute losses in 3D"""
    def __init__(self, model, autobalance=False):
        """Initialization for YOLO loss.

        Args:
            model (torch.Module): YOLO model to calculate loss for.
            autobalance (bool, optional): Whether or not to automatically balance the loss contributions, not entirely sure. Defaults to False.
        """
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        self.g = h['fl_gamma']  # focal loss gamma
        if self.g > 0:
            BCEcls = FocalLoss(BCEcls, self.g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # model's Detect() module in the last layer
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        # extract model hyperparameters from Detect() module and store them as class attributes
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))


    def __call__(self, p, targets):
        """Calculate losses

        Args:
            p (torch.Tensor): predictions to test
            targets (torch.Tensor): targets corresponding to predictions

        Returns:
            total loss (torch.Tensor): tensor containing sum of loss components
            loss components (torch.Tensor): tensor containing loss component values
        """
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gi, gk, gj = indices[i]  # image, anchor, gridz, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gi, gk, gj]  # prediction subset corresponding to targets

                # Regression
                pzxy = ps[:, :3].sigmoid() * 2. - 0.5
                pdwh = (ps[:, 3:6].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pzxy, pdwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], z1x1y1z2x2y2=False) #, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                
                # extra box loss factors to test
                center_dist = bbox_centerDist(pbox.T, tbox[i], z1x1y1z2x2y2=False)
                lbox += center_dist.mean()

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gi, gj, gk, score_iou = b[sort_id], a[sort_id], gi[sort_id], gj[sort_id], gk[sort_id], score_iou[sort_id]
                tobj[b, a, gi, gk, gj] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 7:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += (self.BCEcls(ps[:, 7:], t))  # BCE

            # implementing varifocal loss
            alpha = 0.75
            if self.g > 0:
                focal_weight = tobj*(tobj > 0.0).float() + alpha*(pi[...,6].sigmoid() - tobj).abs().pow(self.g)*(tobj<=0.0).float()                
                obji = torch.nn.functional.binary_cross_entropy_with_logits(pi[..., 6], tobj, reduction='none') * focal_weight
                obji = weight_reduce_loss(obji)
            # no varifocal/focal loss       
            else:
                obji = self.BCEobj(pi[..., 6], tobj)
            
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
                    
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, pred, targets):
        """Build targets for compute_loss()
        input targets with format (image,class,z,x,y,d,w,h)    

        Args:
            pred (torch.Tensor): Example prediction, shape used to set gain
            targets (torch.Tensor): Normalized targets to scale to prediction size

        Returns:
            tcls (List[int]): classes corresponding to each target
            tbox (List[torch.Tensor]): bounding boxes corresponding to each target
            indices (List[Tuple[float]]): image, anchor, and grid indices for each detection layer
            anch (List[int]): List of anchors corresponding to each detection layer
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(9, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        # 3D offsets:
        off = torch.tensor([[0, 0, 0],
                            [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1], # i,j,k,l,m,n
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:8] = torch.tensor(pred[i].shape)[[2, 4, 3, 2, 4, 3]]  # zxyzxy gain

            # Match target (normalized) size to anchors
            t = targets * gain           
            if nt:
                # Matches
                r = t[:, :, 5:8] / anchors[:, None]  # dwh ratio of anchors to targets
                # compare - if any dimension of the anchor is off by more than a
                # factor of anchor_t from the target, filter that anchor out
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']
                t = t[j]  # filter

                # Offsets
                gzxy = t[:, 2:5]  # grid zxy
                gzi = gain[[2, 3, 4]] - gzxy  # inverse
                i, j, k = ((gzxy % 1. < g) & (gzxy > 1.)).T
                l, m, n = ((gzi % 1. < g) & (gzi > 1.)).T
                i = torch.stack((torch.ones_like(i), i, j, k, l, m, n))
                t = t.repeat((7, 1, 1))[i]
                offsets = (torch.zeros_like(gzxy)[None] + off[:, None])[i]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gzxy = t[:, 2:5]  # grid zxy
            gdwh = t[:, 5:8]  # grid dwh
            gijk = (gzxy - offsets).long()
            gi, gj, gk = gijk.T  # grid xyz indices

            # Append
            a = t[:, 8].long()  # anchor indices
            indices.append((b, a, gi.clamp_(0, gain[2] - 1), gk.clamp_(0, gain[4] - 1), gj.clamp_(0, gain[3] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gzxy - gijk, gdwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


class ConfusionMatrix:   
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        """WIP - doesn't fill confusion matrices, could be due to bad predictions
    
        Create confusion matrices to save training results
        Updated version of https://github.com/kaanakan/object_detection_confusion_matrix

        Args:
            nc (int): number of classes
            conf (float, optional): minimum confidence for prediction to be considered. Defaults to 0.25.
            iou_thres (float, optional): minimum IoU to consider a detection to match a label. Defaults to 0.45.
        """
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (z1, x1, y1, z2, x2, y2) format.
        Arguments:
            detections (Array[N, 8]): z1, x1, y1, z2, x2, y2, conf, class
            labels (Array[M, 7]): class, z1, x1, y1, z2, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 6] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 7].int()
        iou = box_iou(labels[:, 1:], detections[:, :6])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def plot(self, normalize=True, save_dir='', names=()):
        """Plots confusion matrix

        Args:
            normalize (bool, optional): whether to normalize column values. Defaults to True.
            save_dir (str, optional): directory to save the confusion matrix to. Defaults to ''.
            names (tuple, optional): names for tick labels. Defaults to ().
        """
        try:
            import seaborn as sn

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                           xticklabels=names + ['background FP'] if labels else "auto",
                           yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
            plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))

