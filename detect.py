"""
Detection script for 3D YOLO.  Used to generate bounding boxes from nifti scans.
Example cmd line call: python detect.py --source ./data/nifti_folder/ --weights ./runs/train/exp/weights/best.pt
"""

# standard library imports
import argparse
import os
import sys
from pathlib import Path
import torch

# set path for local imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO3D root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 2D YOLO imports
from utils.general import print_args, increment_path, check_suffix, check_img_size, colorstr
from utils.torch_utils import select_device

# 3D YOLO imports
from models3D.model import attempt_load
from utils3D.datasets import LoadNiftis
from utils3D.general import non_max_suppression, scale_coords, zxyzxy2zxydwh


# Configuration
default_size = 350 # edge length for testing


@torch.no_grad()
def run(weights,  # model.pt path(s)
        source=ROOT / 'data/images',  # directory containing images to run inference on
        imgsz=default_size,
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=100,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        norm='CT'  # normalization mode, options: CT, MR, Other
        ):
    source = str(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    check_suffix(w)  # check weights have acceptable (.pt) suffix
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    
    model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    imgsz = check_img_size(imgsz, s=stride)[0]  # check image size - since cubic only need one index
    
    # Dataloader
    dataset = LoadNiftis(source, img_size=imgsz, stride=stride)
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 1, imgsz, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    seen = 0
    
    for path, img, im0s in dataset:
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32

        if len(img.shape) == 4: # if only has channel, z, x, y dimensions but no batch
            img = img[None]  # expand for batch dim
        
        if norm.lower() == 'ct':
            # Normalization for Hounsfield units, may see performance improvements by clipping images to +/- 1024.
            img = (img + 1024.) / 2048.0  # int to float32, -1024-1024 to 0.0-1.0
        elif norm.lower() == 'mr':
            mean = torch.mean(img, dim=[1,2,3,4], keepdim=True)
            std_dev = torch.std(img, dim=[1,2,3,4], keepdim=True)
            img = (img - mean)/std_dev
        else:
            raise NotImplementedError("You'll need to write your own normalization algorithm here.")
        
        # Inference
        pred = model(img)[0]

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True, max_det=max_det)
        
        # Process predictions
        for _, det in enumerate(pred):  # per image
            seen += 1
            p, s, im0, _ = path, '', im0s.clone(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            if p.name[-4:] == '.nii':
                txt_path = str(save_dir / 'labels' / p.name[:-4])  # img.txt
            elif p.name[-7:] == '.nii.gz':
                txt_path = str(save_dir / 'labels' / p.name[:-7])  # img.txt
            s += '%gx%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[2, 1, 0, 2, 1, 0]]  # normalization gain dwhdwh - might be [[2, 0, 1, 2, 0, 1]], hard to tell with cubic/square input
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                im0_reshape = [im0.shape[2], im0.shape[1], im0.shape[0]]
                det[:, :6] = scale_coords(img.shape[2:], det[:, :6], im0_reshape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # testing since default run doesn't set save_txt=True
                for *zxyzxy, conf, cls in reversed(det):
                    zxydwh = (zxyzxy2zxydwh(torch.tensor(zxyzxy).view(1, 6)) / gn).view(-1).tolist()  # normalized zxydwh
                    line = (cls, *zxydwh, conf)  # label format
                    print('\ncls z x y d w h conf')
                    print(('%g ' * len(line)).rstrip() % line)
                    
                # Write results
                if save_txt:  # Write to file
                    for *zxyzxy, conf, cls in reversed(det):
                        zxydwh = (zxyzxy2zxydwh(torch.tensor(zxyzxy).view(1, 6)) / gn).view(-1).tolist()  # normalized zxydwh
                        line = (cls, *zxydwh, conf) if save_conf else (cls, *zxydwh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            
    if save_txt: # or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='', help='model path(s)')
    parser.add_argument('--source', type=str, default='', help='file/dir')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[default_size], help='inference size characteristic length')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=100, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--norm', type=str, default='CT', help='normalization type, options: CT, MR, Other')
    opt = parser.parse_args()
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    run(**vars(opt))
    
    
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
