"""
Automatic anchor generation utility functions
Starting from scratch using YOLOv5 code as reference.
"""

# standard library imports
import torch
import numpy as np
import yaml
from tqdm import tqdm
import random

# 2D YOLO imports

# 3D YOLO imports
from utils3D.datasets import LoadNiftisAndLabels


# Configuration
default_size = 350 # edge length for testing


def nifti_check_anchors(dataset, model, thr=4.0, imgsz=default_size):
    """Checks anchor fit to data and recomputes if necessary.

    Args:
        dataset (torch.Dataset): Dataset the anchors will be used with.
        model (torch.Module): Model that will be trained with the anchors
        thr (float, optional): Threshold of ratio between anchors and labels for a given anchor to be considered valid. Defaults to 4.0.
        imgsz (int, optional): Image size used during the training process. Defaults to default_size.

    """
    prefix = 'autoanchor: '
    print(f'\n{prefix}Analyzing anchors... ', end='')
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()

    # converting to pixel widths instead of fractional width using dimensions of resized images
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(dataset.shapes.shape[0],1))  # augment scale
    dwh = torch.tensor(np.concatenate([l[:, 4:7] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # convert dwh from proportion to voxel values

    def metric(k):  # compute metrics for anchors
        r = dwh[:, None] / k[None]
        x = torch.min(r, 1./r).min(2)[0] # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1. / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1. / thr).float().mean()  # best possible recall
        return bpr, aat

    anchors = m.anchors.clone() * m.stride.to(m.anchors.device).view(-1, 1, 1)  # current anchors
    bpr, aat = metric(anchors.cpu().view(-1, 3))
    print(f'anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}', end='')

    # Recompute the anchors if the metric is too low
    if bpr < 0.98:  # threshold to recompute
        print('. Attempting to improve anchors, please wait...')
        na = m.anchors.numel() // 3  # number of anchors
        try:
            anchors = nifti_kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        except Exception as e:
            print(f'{prefix}ERROR: {e}')
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
            # check_anchor_order(m)  # behavior is hard to control and makes setting custom anchors unintuitive
            print(f'{prefix}New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            print(f'{prefix}Original anchors better than new anchors. Proceeding with original anchors.')
    print('')  # newline


def nifti_kmean_anchors(dataset='./data/example.yaml', n=9, img_size=default_size, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset using niftis

            Arguments:
                dataset (str or torch.Dataset): path to data.yaml, or a loaded dataset
                n (int): number of anchors
                img_size (int): image size used for training
                thr (float): anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
                gen (int): generations to evolve anchors using genetic algorithm
                verbose (bool): print all results

            Return:
                k (List[float]): kmeans evolved anchors

            Usage:
                from Experimental3D.AnchorMaker3D import *; _ = nifti_kmean_anchors()
        """
    from scipy.cluster.vq import kmeans

    thr = 1. / thr
    prefix = 'niftianchors: '

    def metric(k, dwh):
        r = dwh[:, None] / k[None]
        x = torch.min(r, 1./r).min(2)[0]  # ratio metric
        return x, x.max(1)[0]  #x, best_x

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), dwh)
        return (best * (best > thr).float()).mean() # fitness

    def print_results(k):
        k = k[np.argsort(k.prod(1))] # sort small to large
        x, best = metric(k, dwh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n # best possible recall, anch > thr
        print(f'{prefix}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr')
        print(f'{prefix}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, '
              f'past_thr={x[x > thr].mean():.3f}-mean: ', end='')
        for i, x in enumerate(k):
            print('%i,%i,%i' % (round(x[0]), round(x[1]), round(x[2])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    # open dataset passed into function, either from instantiated dataset class or by creating one from a yaml file
    if isinstance(dataset, str):  # *.yaml file
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)  # model dict
        dataset = LoadNiftisAndLabels(data_dict['train'], augment=False)

    # converting to pixel widths instead of fractional width using dimensions of resized images
    shapes = img_size * dataset.shapes / dataset.shapes
    dwh0 = np.concatenate([l[:,4:7] * s for s, l in zip(shapes, dataset.labels)]) # dwh

    # Filter very small objects
    size_thresh = 4.0
    i = (dwh0.prod(1) < size_thresh).sum()  # check whether any label bounding box has a volume less than 4 voxels
    if i:
        print(f'{prefix}WARNING: Extremely small objects found. {i} of {len(dwh0)} labels are < {size_thresh} voxels in size.')
    dwh = dwh0[(dwh0.prod(1) >= size_thresh)]  # filter out any label bounding box with a volume less than 4 voxels

    # Kmeans calculation
    print(f'{prefix}Running kmeans for {n} anchors on {len(dwh)} points...')
    s = dwh.std(0)  # sigmas for whitening
    k, dist = kmeans(dwh / s, n, iter=30)  # points, mean distance
    assert len(k) == n, f'{prefix}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}'
    k *= s
    dwh = torch.tensor(dwh, dtype=torch.float32)  # filtered
    dwh0 = torch.tensor(dwh0, dtype=torch.float32)  # unfiltered
    k = print_results(k)

    # Evolve
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc=f'{prefix}Evolving anchors with Genetic Algorithm:')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{prefix}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k)

    return print_results(k)


def check_anchor_order(m):
    """    
    Check anchor order against stride order for YOLO3D Detect() module m, and correct if necessary.
    
    Suggested not to run
    Reverses anchor order when it should not because it compares the volume of anchors in a very naive fashion
    This makes the behavior hard to control, and it makes choosing custom anchors less transparent and more confusing
    Included for parity with YOLOv5 code

    Args:
        m (List[float]): anchors to be checked
    """
    v = m.anchors.prod(-1).view(-1)  # anchor volume
    dv = v[-1] - v[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if dv.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
