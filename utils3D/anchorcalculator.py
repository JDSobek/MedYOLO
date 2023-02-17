"""
Analyzes a YOLO3D Training dataset's labels and attempts to calculate ideal initial anchor hyperparameters for the model yaml file.
"""

# standard library imports
import torch
import numpy as np
import yaml
from tqdm import tqdm
import random
from scipy.cluster.vq import kmeans
import sys
from pathlib import Path
import os

# set path for local imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO3D root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# 2D YOLO imports

# 3D YOLO imports
from utils3D.datasets import LoadNiftisAndLabels, nifti_dataloader


# Configuration
default_size = 350 # edge length for testing


def AnchorCalculator(dataset, model, imgsz=default_size, thr=4.0, gen=1000, opt_hypers=False):
    # Calculate best anchors for a YOLO3D model using a dataset's training labels.
    thr = 1. / thr
    prefix = 'AnchorCalculator: '

    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
    nl = len(m.anchors)
    na = m.anchors.numel() // 3  # total number of anchors, all layers combined
    
    def metric(k, dwh):
        # k is a list of lists with shape [num_anchors, 3]
        # dwh is a tensor of shape [num_labels, 3]
        # r has shape [num_labels, num_anchors, 3]
        # x has shape [num_labels, num_anchors]
        # best_x has shape [num_labels]
        
        # calculate the length ratio of the edges of the anchors to the edges of the labels
        r = dwh[:, None] / k[None]
        # finds the worst mismatch (ratio furthest below 1) between each edge of each anchor and the corresponding edge of each label
        x = torch.min(r, 1./r).min(2)[0]  # ratio metric
        # finds the best (closest to 1) ratio of "worst anchor edge to label edge ratio" among every anchor for each label
        # i.e. looks at the most mismatched edge of each anchor to find the least mismatched anchor
        best_x = x.max(1)[0]
        return x, best_x

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), dwh)
        return (best * (best > thr).float()).mean() # fitness
    
    def print_results(k):
        k = k[np.argsort(k.prod(1))] # sort small to large
        x, best = metric(k, dwh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * na # best possible recall, anch > thr
        print(f'{prefix}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr')
        print(f'{prefix}n={na}, img_size={imgsz}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, '
              f'past_thr={x[x > thr].mean():.3f}-mean: ', end='')
        for i, x in enumerate(k):
            print('%i,%i,%i' % (round(x[0]), round(x[1]), round(x[2])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    def optimize_hypers(dwh, max_anch=99):
        # saves an image of the kmeans mean distance vs. number of anchors to help determine optimal number of anchors
        import matplotlib.pyplot as plt
        
        s = dwh.std(0)  # sigmas for whitening
        norm_dwh = dwh / s
        dist_list = []
        for i in range(1, max_anch):
            k, dist = kmeans(norm_dwh, i, iter=30)  # points, mean distance
            dist_list.append(dist)
        
        plt.figure(figsize=(10,6))
        plt.plot(range(1, max_anch), dist_list, color='blue', linestyle='dashed')
        plt.title('Mean Distance vs. Number of Anchors')
        plt.xlabel('Number of Anchors')
        plt.ylabel('Mean Distance')
        savepath = ROOT / 'utils3D/anchoropt.png'
        print('Saving results of anchor number optimization to ', savepath)
        plt.savefig(savepath)
        plt.close()

    # open dataset passed into function, either from instantiated dataset class or by creating one from a yaml file
    if isinstance(dataset, str):  # *.yaml file
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)  # model dict
        dataset = LoadNiftisAndLabels(data_dict['train'], augment=False)
        
    # converting to pixel widths instead of fractional width using dimensions of resized images
    shapes = imgsz * dataset.shapes / dataset.shapes  # Converts imgsz into correct len 3 shape
    dwh0 = np.concatenate([l[:,4:7] * s for s, l in zip(shapes, dataset.labels)]) # dwh
    
    # Filter very small objects
    size_thresh = 4.0
    i = (dwh0.prod(1) < size_thresh).sum()  # check whether any label bounding box has a volume less than 4 voxels
    if i:
        print(f'{prefix}WARNING: Extremely small objects found. {i} of {len(dwh0)} labels are < {size_thresh} voxels in size.')
    dwh = dwh0[(dwh0.prod(1) >= size_thresh)]  # filter out any label bounding box with a volume less than 4 voxels
    
    if opt_hypers:
        print('Determining optimal number of anchors')
        optimize_hypers(dwh)
    
    # Kmeans calculation
    print(f'{prefix}Running kmeans for {na} anchors on {len(dwh)} points...')
    s = dwh.std(0)  # sigmas for whitening
    k, _ = kmeans(dwh / s, na, iter=30)  # points, mean distance
    
    assert len(k) == na, f'{prefix}ERROR: scipy.cluster.vq.kmeans requested {na} points but returned only {len(k)}'
    k *= s
    dwh = torch.tensor(dwh, dtype=torch.float32)  # filtered
    dwh0 = torch.tensor(dwh0, dtype=torch.float32)  # unfiltered
    
    # Reference pre-evolution calculated anchors
    print('\nnew, pre-evolution anchors calculated from dataset:')
    k = print_results(k)

    # Evolve anchors
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc=f'\n{prefix}Evolving anchors with Genetic Algorithm:')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{prefix}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
    
    # Reference post-evolution calculated anchors
    print('\nnew, post-evolution anchors calculated from dataset:')
    print_results(k)
    
    # convert anchors into easy copy/paste format for model yaml file
    anchors_per_layer = na//nl
    anchor_list = []
    for i in range(nl):
        anchor_list.append([])
        for j in range(3*i*anchors_per_layer, 3*(i+1)*anchors_per_layer):
            anchor_list[i].append(int(round(k.item(j))))
            
    print('\nlist of anchors for model yaml file:')
    print(anchor_list)
    

if __name__=='__main__':
    from models3D.model import Model
    
    # choose model hyperparameters
    hyp = ROOT / 'data/hyps/hyp.scratch.yaml'
    with open(hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)
    data = ROOT / 'data/test.yaml'  # This will need to be replaced with the yaml file for your data
    with open(data, errors='ignore') as f:
        data_dict = yaml.safe_load(f)  # model dict

    nc = int(data_dict['nc'])  # number of classes
    
    # Model
    model = Model(cfg = ROOT / 'models3D/yolo3Ds.yaml', ch=1, nc=nc, anchors=hyp.get('anchors'))
    
    # datasets
    train_path, val_path = data_dict['train'], data_dict['val']
    train_loader, train_dataset = nifti_dataloader(train_path, imgsz=default_size, batch_size=1, stride=model.stride)
    
    AnchorCalculator(train_dataset, model, opt_hypers=True)
