"""
Dataloaders and dataset utils for nifti datasets for YOLO3D
"""

# standard library imports
import nibabel as nib
import numpy as np
from pathlib import Path
import glob
import os
from multiprocessing.pool import Pool
from tqdm import tqdm
from itertools import repeat
from typing import List
import torch
from torch.utils.data import Dataset

# 2D YOLO imports
from utils.torch_utils import torch_distributed_zero_first
from utils.datasets import InfiniteDataLoader, get_hash

# 3D YOLO imports
from utils3D.general import zxyzxy2zxydwhn, zxydwhn2zxyzxy
from utils3D.augmentations import tensor_cutout, random_zoom


# Configuration
IMG_FORMATS = ['nii', 'nii.gz']  # acceptable image suffixes, note nii.gz compatible by checking for presence of 'nii' in -2 place
NUM_THREADS = min(8, os.cpu_count())  # number of multiprocessing threads
default_size = 350 # edge length for testing


def file_lister_train(parent_dir: List[str], prefix=''):
    """Takes a parent directory or list of parent directories and
    looks for files within those directories.  Output organized to fit
    YOLO training requirements.

    Args:
        parent_dir (List[str] or str): Folders to be searched.  Text files allowed.
        prefix (str, optional): Prefix for error messages. Defaults to ''.

    Raises:
        Exception: If parent_dir is neither a directory nor file.

    Returns:
        file_list (List[str]): a list of paths to the files found.
        p (pathlib.PosixPath): the path to the parent directory, for caching purposes
    """

    file_list = []
    for p in parent_dir if isinstance(parent_dir, list) else [parent_dir]:
        p = Path(p)
        if p.is_dir():  # dir
            file_list += glob.glob(str(p / '**' / '*.*'), recursive=True)
        elif p.is_file():  # file
            with open(p) as t:
                t = t.read().strip().splitlines()
                parent = str(p.parent) + os.sep
                file_list += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                # file_list += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
        else:
            raise Exception(f'{prefix}{p} does not exist')
    return file_list, p


def file_lister_detect(parent_dir: str):
    """Takes a parent directory and looks for files within those directories.
    Output organized to fit YOLO inference requirements.

    Args:
        parent_dir (str): parent folder to search for files.

    Raises:
        Exception: if parent_dir is not a file, directory, or glob search pattern.

    Returns:
        files (List[str]): a list of paths to the files found.
    """
    p = str(Path(parent_dir).resolve())
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return files


class LoadNiftis(Dataset):
    """YOLO3D Pytorch Dataset for inference."""
    def __init__(self, path: str, img_size=default_size, stride=32):
        """Initialization for the inference Dataset

        Args:
            path (str): parent directory for the Dataset's files
            img_size (int, optional): edge length for the cube input will be reshaped to. Defaults to default_size (currently 350).
            stride (int, optional): model stride, used for resizing and augmentation, currently unimplemented. Defaults to 32.
        """
        
        # Find files in the given path and filter to leave only .nii and .nii.gz files in the list
        files = file_lister_detect(path)
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS or x.split('.')[-2].lower() in IMG_FORMATS]

        self.nf = len(images)
        self.files = images
        self.img_size = img_size
        self.stride = stride

        assert self.nf > 0, f'No images found in {path}. Supported formats are: {IMG_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        # Iterate through the list of files
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read current image
        self.count += 1
        # img0, affine = open_nifti(path)
        img0, _ = open_nifti(path)
        assert img0 is not None, 'Image Not Found ' + path
        print(f'\nimage {self.count}/{self.nf} {path}: ', end='')

        # Reshape image to fit model requirements
        img = transpose_nifti_shape(img0)
        img = change_nifti_size(img, self.img_size)

        return path, img, img0

    def __len__(self):
        return self.nf  # number of files


class LoadNiftisAndLabels(Dataset):
    """YOLO3D Pytorch Dataset for training."""
    cache_version = 0.61  # dataset labels *.cache version

    def __init__(self, path, img_size=default_size, batch_size=4, augment=False, hyp=None, single_cls=False,
                 stride=32, pad=0.0, prefix=''):
        """Initialization for the training Dataset

        Args:
            path (str): parent directory for the Dataset's files
            img_size (int, optional): edge length for the cube input will be reshaped to. Defaults to default_size (currently 350).
            batch_size (int, optional): size of the batch to return. Defaults to 4.
            augment (bool, optional): determines whether data will be augmented, currently unimplemented. Defaults to False.
            hyp (Dict, optional): dictionary containing augmentation configuration hyperparameters, currently unimplemented. Defaults to None.
            stride (int, optional): model stride, used for resizing and augmentation, currently unimplemented. Defaults to 32.
            pad (float, optional): image padding, used for resizing and augmentation, currently unimplemented. Defaults to 0.0.
            prefix (str, optional): Prefix for error messages. Defaults to ''.

        Raises:
            Exception: if unable to load data in given path.
        """       
        self.img_size = img_size
        self.stride = stride
        self.path = path
        self.augment = augment
        self.hyp = hyp

        # Find files in the given path and filter to leave only .nii and .nii.gz files in the list
        try:
            f, p = file_lister_train(path, prefix)
            self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS or x.split('.')[-2].lower() in IMG_FORMATS) # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}')

        # Check cache and find the labels
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # same version
            assert cache['hash'] == get_hash(self.label_files + self.img_files)  # same hash
        except:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels.'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        # nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, label in enumerate(self.labels):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0

        self.imgs, self.img_npy = [None] * n, [None] * n

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        """Caches dataset labels, verifies images and reads their shapes.
        See: verify_image_label function

        Args:
            path (pathlib.Path, optional): Path to write cache to. Defaults to Path('./labels.cache').
            prefix (str, optional): prefix for error messages. Defaults to ''.

        Returns:
            x (Dict): Dictionary containing the results of the image search.
        """
        x = {}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix))),
                        desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
        except Exception as e:
            print(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        """Loads niftis and converts to torch tensor to be fed as input to the model.

        Args:
            index (int): dataset index of image to be read.

        Returns:
            img (torch.tensor): image data from loaded nifti, potentially augmented
            labels_out (torch.tensor): labels corresponding to loaded nifti, with augmentation accounted for
            self.img_files[index] (str): path to the loaded nifti
            shapes (Tuple[Tuple[float]]): Tuple containing Tuples of relative shape information for original image, resized image, and padding
        """
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp # used to configure augmentation
        
        # Load image
        img, (d0, h0, w0), (d, h, w), _ = load_nifti(self, index)

        # Letterbox
        # shape = (self.img_size, self.img_size, self.img_size) # not adding rectangular training yet
        # img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment) # not implemented
        # ratio = (1, 1, 1) # no letterboxing so the shape doesn't change and the ratios are all 1
        pad = (0, 0, 0) # shape not changing so not padding any side
        shapes = (d0, h0, w0), ((d/d0, h/h0, w/w0), pad)

        labels = self.labels[index].copy()
        nl = len(labels)  # number of labels

        if self.augment:           
            # Label transformation is done to make certain augmentations more straightforward
            if labels.size:  # normalized zxydwh to pixel zxyzxy format
                labels[:, 1:] = zxydwhn2zxyzxy(labels[:, 1:], d, w, h, pad[0], pad[1], pad[2])                    

            # random zoom
            img, labels = random_zoom(img, labels, hyp['max_zoom'], hyp['min_zoom'], hyp['prob_zoom'])
        
            # transformation of labels back to standard format
            if nl:
                labels[:, 1:7] = zxyzxy2zxydwhn(labels[:, 1:7], d=img.shape[1], w=img.shape[3], h=img.shape[2], clip=True, eps=1E-3)
            
            # Albumentations
            
            # HSV color-space
            
            # Flip up-down
            
            # Flip left-right
            
            # Cutouts
            labels = tensor_cutout(img, labels, hyp['cutout_params'], hyp['prob_cutout'])            
            # update after cutout
            nl = len(labels)  # number of labels
        
        labels_out = torch.zeros((nl, 8))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        return img, labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        """Used to collate images to create the input batches"""
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


def nifti_dataloader(path: str, imgsz: int, batch_size: int, stride: int, single_cls=False, hyp=None, augment=False, pad=0.0,
                     rank=-1, workers=8, prefix=''):
    """This is the dataloader used in the training process
    The same as that of 2D YOLO, just built around a different Dataset definition

    Args:
        path (str): path to the directory containing the training files
        imgsz (int): edge length for the cube input will be reshaped to.
        batch_size (int): size of the batch to return.
        stride (int): model stride, used for resizing and augmentation
        hyp (Dict, optional): dictionary containing augmentation configuration hyperparameters. Defaults to None.
        augment (bool, optional): whether or not augmentation should be enabled. Defaults to False.
        pad (float, optional): image padding, used for resizing and augmentation. Defaults to 0.0.
        rank (int, optional): determines whether to use distributed sampling. Defaults to -1.
        workers (int, optional): number of dataloader workers. Defaults to 8.
        prefix (str, optional): Prefix for error messages. Defaults to ''.

    Returns:
        dataloader: dataloader for training loop
        dataset: training dataset
    """
    
    with torch_distributed_zero_first(rank):
        dataset = LoadNiftisAndLabels(path, imgsz, batch_size,
                                      augment=augment,
                                      hyp=hyp,
                                      # rect=rect,  # rectangular training
                                      single_cls=single_cls,
                                      stride=stride,
                                      pad=pad,
                                      prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = InfiniteDataLoader
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadNiftisAndLabels.collate_fn)
    return dataloader, dataset


def img2label_paths(img_paths):
    """Defines label paths as a function of the image paths.  Filters for .nii and .nii.gz files.

    Args:
        img_paths (List[str]): list of image file paths to convert to label file paths.

    Returns:
        List[str]: list of label file paths
    """
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    label_paths = []

    # to handle both compressed and uncompressed niftis
    for x in img_paths:
        if x.endswith('.nii.gz'):
            label_paths.append(sb.join(x.rsplit(sa, 1)).rsplit('.', 2)[0] + '.txt')
        elif x.endswith('.nii'):
            label_paths.append(sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt')

    return label_paths


def verify_image_label(args):
    """Verify one image-label pair.  Works for .nii and .nii.gz files.

    Args:
        args (Tuple[str]): contains the image path, label path, and error message prefix

    Returns:
        im_file (str): path to the image file
        l (List[float]): labels corresponding to the image file
        shape (List[int]): 3D shape of the image file
        segments: Alternate representation of image shape, currently not supported but necessary for compatibility with YOLOv5 code.
        nm (int): 1 if label missing, 0 if label found
        nf (int): 1 if label found, 0 if label not found
        ne (int): 1 if label empty, 0 if label not empty
        nc (int): 1 if label corrupted and Exception found, 0 if not
        msg (str): Message returned in the event an error occurs
    """
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = np.array(nib.load(im_file).dataobj)
        shape = (im.shape[2], im.shape[0], im.shape[1]) # need to transpose to account for depth reshaping that will happen to image tensors
        # assert call may need to be reworked depending on model requirements, this is just an estimate
        assert (shape[0] > 9) & (shape[1] > 99) & (shape[2] > 99), f'image size {shape} < 10x100x100 voxels'
        assert im_file.split('.')[-1].lower() in IMG_FORMATS or im_file.split('.')[-2].lower() in IMG_FORMATS, f'invalid image format {im_file}'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1 # label found
            with open(lb_file) as f:
                l = [x.split() for x in f.read().strip().splitlines() if len(x)]
                # segments aren't supported for simplicity
                l = np.array(l, dtype=np.float32)
            nl = len(l)
            if nl:
                assert l.shape[1] == 7, f'labels require 7 columns, {l.shape[1]} columns detected'
                assert (l >= 0).all(), f'negative label values {l[l < 0]}'
                assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}'
                l = np.unique(l, axis=0)  # remove duplicate rows
                if len(l) < nl:
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(l)} duplicate labels removed'
            else:
                ne = 1  # label empty
                l = np.zeros((0, 7), dtype=np.float32)
        else:
            nm = 1  # label missing
            l = np.zeros((0, 7), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def open_nifti(filepath: str):
    """Reads a nifti file and converts it to a torch tensor

    Args:
        filepath (str): Path to the nifti file

    Returns:
        nifti_tensor (torch.tensor): Tensor containing the nifti image data
        nifti_affine: affine array for the nifti
    """
    nifti = nib.load(filepath)
    nifti_array = np.array(nifti.dataobj)
    nifti_affine = nifti.affine
    assert nifti_array is not None, 'Image Not Found ' + filepath
    nifti_tensor = torch.tensor(nifti_array, dtype=torch.float)
    return nifti_tensor, nifti_affine


def transpose_nifti_shape(nifti_tensor: torch.Tensor):
    """Reshapes the tensor from height, width, depth order to depth, height, width
    to make it compatible with torch convolutions.

    Args:
        nifti_tensor (torch.tensor): tensor to be reshaped

    Returns:
        nifti_tensor (torch.tensor): reshaped tensor
    """
    nifti_tensor = torch.transpose(nifti_tensor, 0, 2)
    nifti_tensor = torch.transpose(nifti_tensor, 1, 2)
    return nifti_tensor


def change_nifti_size(nifti_tensor: torch.Tensor, new_size: int):
    """Resizes a 3D tensor to a cube with edge length new_size.
    Also adds the channel dimension.

    Args:
        nifti_tensor (torch.Tensor): The tensor to be resized
        new_size (int): The edge length for the resized, cubic tensor

    Returns:
        nifti_tensor (torch.tensor): Resized, cubic tensor
    """
    # add channel dimension for compatibility with later code
    nifti_tensor = torch.unsqueeze(nifti_tensor, 0)
    # add batch dimension for functional interpolate
    nifti_tensor = torch.unsqueeze(nifti_tensor, 0)
    # resize image to a cube of size new_size
    nifti_tensor = torch.nn.functional.interpolate(nifti_tensor, size=(new_size, new_size, new_size), mode='trilinear', align_corners=False)
    # remove batch dimension for compatibility with later code
    nifti_tensor = torch.squeeze(nifti_tensor, 0)
    return nifti_tensor


def load_nifti(self, i):
    """Reads a nifti file, converts it to a torch.tensor, and reshapes and resizes it for use in the YOLO3D model.

    Args:
        i (int): Dataset index for the nifti to be loaded

    Returns:
        im (torch.tensor): YOLO3D input tensor containing the nifti image data
        d0 (int): original image depth
        h0 (int): original image height
        w0 (int): original image width
        im.size()[1:] (List(int)): current image depth, height, and width
    """
    # loads 1 image from dataset index 'i'
    path = self.img_files[i]
    im, affine = open_nifti(path)

    # reshape im from height, width, depth to depth, height, width to make it compatible with torch convolutions
    im = transpose_nifti_shape(im)

    d0, h0, w0 = im.size()

    # resize im to self.img_size
    im = change_nifti_size(im, self.img_size)

    return im, (d0, h0, w0), im.size()[1:], affine


def normalize_CT(imgs):
    """Normalizes 3D CTs in Hounsfield Units (+/- 1024) to within 0 and 1.

    Args:
        imgs (torch.tensor): unnormalized model input

    Returns:
        imgs (torch.tensor): normalized model input
    """
    imgs = (imgs + 1024.) / 2048.0  # int to float32, -1024-1024 to 0.0-1.0
    return imgs


def normalize_MR(imgs):
    """Volume normalizes 3D MR images to mean 0 and standard deviation 1.

    Args:
        imgs (torch.tensor): unnormalized model input

    Returns:
        imgs (torch.tensor): normalized model input
    """
    means = torch.mean(imgs, dim=[1,2,3,4], keepdim=True)
    std_devs = torch.std(imgs, dim=[1,2,3,4], keepdim=True)
    imgs = (imgs - means)/std_devs
    return imgs
