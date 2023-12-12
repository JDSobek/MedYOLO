"""
Example script for converting MedYOLO predictions to nifti masks.
Label to mask conversion functions may need adaptation for different projects.
"""

# standard library imports
import argparse
import os
import nibabel as nib
import numpy as np
import torch
import math


def torch_to_nifti(data_tensor: torch.Tensor, nifti_path: str, affine, size):
    """
    Converts unnormalized torch tensor into a nifti image.
    Useful for debugging new (e.g. non-NIfTI) dataloader pipelines.
    """
    # add batch dimension for functional interpolate
    data_tensor = torch.unsqueeze(data_tensor, 0)
    data_tensor = torch.nn.functional.interpolate(data_tensor, size=(size[0], size[1], size[2]), mode='trilinear', align_corners=False)
    # remove batch dimension
    data_tensor = torch.squeeze(data_tensor, 0)
    # remove channel dimension
    data_tensor = torch.squeeze(data_tensor, 0)
    data_tensor = torch.transpose(data_tensor, 0, 2)
    data_tensor = torch.transpose(data_tensor, 0, 1)
    nifti = nib.Nifti1Image(data_tensor.cpu().numpy(), affine)
    nib.save(nifti, nifti_path)
    return nifti


def multilabel_mask_maker(bbox_path: str, nifti_path: str, mask_path: str):
    """
    Makes nifti masks out of a YOLO label txt file.  Saves highest confidence mask for each class.
    Args:
        bbox_path: path to the YOLO label file.
        nifti_path: path to the corresponding nifti image file.
        mask_path: path to save the resultant mask as.
    """
    f = open(bbox_path, 'r')
    label = list(filter(None, f.read().split('\n')))  # filtering out blank lines

    # load nifti
    nifti = nib.load(nifti_path)
    nifti_array = np.array(nifti.dataobj)
    # mask_array = np.zeros_like(nifti_array)

    # might need to flip order of height and width...
    height = nifti_array.shape[0]
    width = nifti_array.shape[1]
    depth = nifti_array.shape[2]

    box_dict = {}
    for target in label:
        cls, z, x, y, d, w, h, conf = target.split(' ')
        cls = int(cls)
        z = float(z)
        x = float(x)
        y = float(y)
        d = float(d)
        w = float(w)
        h = float(h)
        conf = float(conf)

        if cls not in box_dict.keys() or box_dict[cls][-1] < conf:
            box_dict[cls] = z, x, y, d, w, h, conf

    for cls in box_dict.keys():
        # create empty mask
        mask_array = np.zeros_like(nifti_array)
        z, x, y, d, w, h, conf = box_dict[cls]

        z_center = z * depth
        x_center = x * width
        y_center = y * height
        z_length = d * depth
        x_length = w * width
        y_length = h * height

        min_z = int(math.floor(z_center - z_length / 2))
        max_z = int(math.ceil(z_center + z_length / 2))
        min_x = int(math.floor(x_center - x_length / 2))
        max_x = int(math.ceil(x_center + x_length / 2))
        min_y = int(math.floor(y_center - y_length / 2))
        max_y = int(math.ceil(y_center + y_length / 2))

        min_z = max(0, min_z)
        max_z = min(depth, max_z)
        min_y = max(0, min_y)
        max_y = min(height, max_y)
        min_x = max(0, min_x)
        max_x = min(width, max_x)

        mask_array[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1] = 1

        cls_mask_path = mask_path.split('.')[0] + '_' + str(cls) + '.nii.gz'
        mask_nifti = nib.Nifti1Image(mask_array, nifti.affine)
        nib.save(mask_nifti, cls_mask_path)
   

def mask_maker(bbox_path: str, nifti_path: str, mask_path: str):
    """
    Makes nifti masks out of YOLO label txt files.  Only works for one label per mask.
    Labels should have one prediction without confidence metric.
    Args:
        bbox_path: path to the MedYOLO label file.
        nifti_path: path to the corresponding nifti image file.
        mask_path: path to save the resultant mask as.
    """
    f = open(bbox_path, 'r')
    label = list(filter(None, f.read().split('\n'))) # filtering out blank lines
    
    # load nifti and create empty mask
    nifti = nib.load(nifti_path)
    mask_array = np.zeros_like(np.array(nifti.dataobj))
    
    # might need to flip order of height and width...
    height = mask_array.shape[0]
    width = mask_array.shape[1]           
    depth = mask_array.shape[2]
    
    for target in label:
        cls, z, x, y, d, w, h = target.split(' ')
        cls = int(cls)
        z = float(z)
        x = float(x)
        y = float(y)
        d = float(d)
        w = float(w)
        h = float(h)

        z_center = z*depth
        x_center = x*width
        y_center = y*height
        z_length = d*depth
        x_length = w*width
        y_length = h*height

        min_z = int(math.floor(z_center - z_length / 2))
        max_z = int(math.ceil(z_center + z_length / 2))
        min_x = int(math.floor(x_center - x_length / 2))
        max_x = int(math.ceil(x_center + x_length / 2))
        min_y = int(math.floor(y_center - y_length / 2))
        max_y = int(math.ceil(y_center + y_length / 2))

        min_z = max(0, min_z)
        max_z = min(depth, max_z)
        min_y = max(0, min_y)
        max_y = min(height, max_y)
        min_x = max(0, min_x)
        max_x = min(width, max_x)

        mask_array[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1] = 1
        
    mask_nifti = nib.Nifti1Image(mask_array, nifti.affine)
    nib.save(mask_nifti, mask_path)


def run(nifti_dir, bbox_dir, mask_dir, mask_tag, single_mask=False):
    """
    Creates nifti masks for MedYOLO bounding boxes found in bbox_dir that have a corresponding nifti image in nifti_dir.
    See mask maker functions above for more details.
    """
    file_list = []
    for dirpath, subdirs, files in os.walk(nifti_dir):
        for file in files:
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                file_list.append(os.path.join(dirpath, file))

    for file in file_list:
        try:
            print(file)
            if file.endswith('.nii'):
                label = file.split('/')[-1][:-4] + '.txt'
                label_path = os.path.join(bbox_dir, label)
                mask_path = os.path.join(mask_dir, file.split('/')[-1][:-4] + mask_tag + '.nii.gz')
        
            if file.endswith('.nii.gz'):
                label = file.split('/')[-1][:-7] + '.txt'
                label_path = os.path.join(bbox_dir, label)
                mask_path = os.path.join(mask_dir, file.split('/')[-1][:-7] + mask_tag + '.nii.gz')
            if single_mask:
                mask_maker(label_path, file, mask_path)
            else:
                multilabel_mask_maker(label_path, file, mask_path)
        except FileNotFoundError:
            continue


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nifti-dir', type=str, default='', help='directory containing the niftis masks are needed for')
    parser.add_argument('--bbox-dir', type=str, default='', help='directory containing MedYOLO predictions for the niftis')
    parser.add_argument('--mask-dir', type=str, default='', help='directory to save nifti masks in')
    parser.add_argument('--mask-tag', type=str, default='', help='tag appended to distinguish mask filenames from their corresponding input')
    # This option is generally intended for single class tasks but can generate masks for multi-label tasks
    # Leaving it false will generate a separate mask file for every prediction that saves the highest confidence prediction for each class
    parser.add_argument('--single-mask', action='store_true', help='generate one mask file with all classes flattened into class 1')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
