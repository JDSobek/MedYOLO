"""
Utility functions to test nifti functionality.
"""

# standard library imports
import nibabel as nib
import numpy as np
import torch


def torch_to_nifti(data_tensor: torch.Tensor, nifti_path: str, affine, size):
    """
    Converts unnormalized torch tensor into a nifti image
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
    data = data_tensor.cpu().numpy()

    nifti = nib.Nifti1Image(data, affine)
    nib.save(nifti, nifti_path)
    return nifti
    

def mask_maker(label, nifti, mask_path: str):
    """Makes nifti masks out of YOLO labels.
    Only works with 1 label per mask, will need changes for multiple label masks
    Args:
        label (torch.Tensor or np.ndarray): YOLO label
        mask_dir (str): folder to save created nifti mask to
    """
    
    nifti_array = np.array(nifti.dataobj)
    mask_array = np.zeros_like(nifti_array)
    
    # might need to flip order of height and width...
    height = mask_array.shape[0]
    width = mask_array.shape[1]           
    depth = mask_array.shape[2]
    
    for target in label:
        cls, z, x, y, d, w, h = target
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
        
        min_z = int(z_center - z_length/2)
        max_z = int(z_center + z_length/2)
        min_x = int(x_center - x_length/2)
        max_x = int(x_center + x_length/2)
        min_y = int(y_center - y_length/2)
        max_y = int(y_center + y_length/2)
        
        mask_array[min_x:max_x, min_y:max_y, min_z:max_z] = 1
        
    mask_nifti = nib.Nifti1Image(mask_array, nifti.affine)
    nib.save(mask_nifti, mask_path)


if __name__ == '__main__':
    pass
