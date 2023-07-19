"""
Model definition script for 3D YOLO.  Defines the modules and the overall model.
Modified with extra detection layer
Mostly contains 3D versions of code from models/yolo.py and models/common.py
"""

# standard library imports
import sys
import torch
from pathlib import Path
import os
import warnings
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math

# set path for local imports
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO3D root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# 2D YOLO imports
from models.common import autopad, Concat
from utils.general import make_divisible

# 3D YOLO imports


# Configuration
default_size = 350 # edge length for testing


def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    Called during validation.

    Args:
        weights (str or List[str]): path to weights file or list of paths to weights files to use in Ensemble of models.
        map_location (torch.device or str or Dict or callable, optional): Where to load model. Defaults to None.
        inplace (bool, optional): Whether the model's detect layer should use in-place slice assignment for pytorch 1.7.0 compatibility.  Don't think it's implemented yet. Defaults to True.
        fuse (bool, optional): Whether to fuse conv and batchnorm layers. Defaults to True.

    Returns:
        model (torch.Module): Loaded model.
    """
    from models.experimental import Ensemble

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(str(Path(str(w).strip().replace("'", ''))), map_location=map_location)  # load
        if fuse:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
        else:
            model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())  # without layer fuse

    # Compatibility updates
    # modules not prefixed by nn are from 3D YOLO code and aren't native pytorch
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
            if type(m) is Detect:
                if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble


def fuse_conv_and_bn(conv, bn):
    """Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/

    Args:
        conv (torch.Module): Conv layer to fuse.
        bn (torch.Module): Batchnorm layer to fuse.

    Returns:
        fusedconv (torch.Module): Fused layers.
    """
    
    fusedconv = nn.Conv3d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False):
    """Model information.

    Args:
        model (torch.Module): Model to get information from
        verbose (bool, optional): Whether or not to print named parameters. Defaults to False.
        img_size (int, optional): Image size for the loaded model. Defaults to default_size.
    """
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))


# YOLO layers
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """3D convolution and 3D batchnorm layer for 3D YOLO models

        Args:
            c1 (int): channels in
            c2 (int): channels out
            k (int, optional): Kernel size. Defaults to 1.
            s (int, optional): Stride. Defaults to 1.
            p (int, optional): Padding. Defaults to None.
            g (int, optional): Groups. Defaults to 1.
            act (bool, optional): Whether to include activation layer or not. Defaults to True.
        """
        super().__init__()
        self.conv = nn.Conv3d(c1, c2, k, s, autopad(k,p), groups=g, bias=False)
        self.bn = nn.BatchNorm3d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Bottleneck layers for 3D YOLO models

        Args:
            c1 (int): channels in
            c2 (int): channels out
            shortcut (bool, optional): Whether to include a residual connection.  Requires c1 == c2. Defaults to True.
            g (int, optional): Groups. Defaults to 1.
            e (float, optional): Expansion/contraction factor for hidden channels. Defaults to 0.5.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """CSP Bottleneck with 3 convolutions.
        CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks

        Args:
            c1 (int): channels in
            c2 (int): channels out
            n (int, optional): Number of Bottleneck layers to include. Defaults to 1.
            shortcut (bool, optional): Whether to include a residual connection in the Bottleneck layer.  Requires c1 == c2. Defaults to True.
            g (int, optional): Groups. Defaults to 1.
            e (float, optional): Expansion/contraction factor for hidden channels. Defaults to 0.5.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher

        Args:
            c1 (int): channels in
            c2 (int): channels out
            k (int, optional): kernel size. Defaults to 5.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool3d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Detect(nn.Module):
    stride = None  # strides computed during build

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        """YOLO Detection layer, adjusted for 3D

        Args:
            nc (int, optional): Number of classes. Defaults to 80.
            anchors (Tuple[int], optional): Anchor hyperparameters to use. Defaults to ().
            ch (Tuple[int], optional): Channels for the output convolution at each detection layer. Defaults to ().
            inplace (bool, optional): Whether to use in-place operations for pytorch 1.7.0 compatibility.  Don't think it's implemented yet. Defaults to True.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 7  # number of outputs per anchor, (zxydwh) + nc + 1
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 3 # number of anchors per detection layer
        self.grid = [torch.zeros(1)] * self.nl  # initialize grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # initialize anchor grid
        
        # This tells torch to save the anchors to the state_dict for the model as parameters that should not be trained
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 3))  # shape(nl,na,3)

        self.m = nn.ModuleList(nn.Conv3d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = [] # inference output
        for i in range(self.nl):           
            # The module list receives inputs from different layers of the network, one for each detection layer
            # and has separate Conv layers to handle each of those inputs
            x[i] = self.m[i](x[i])  # conv

            # x(bs, self.na*self.no, 20, 20) to x(bs, self.na, 20, 20, self.no)
            # 20 is an example spatial extent for a given detection layer, which varies depending on model imgsz parameter
            bs, _, nz, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, nz, ny, nx).permute(0, 1, 3, 4, 5, 2).contiguous()

            # self.training is an internal attribute set to True by model.train(), this is only run in eval mode
            # during training anchors are set by hyperparameters and nifti_check_anchors
            if not self.training: # inference
                if self.grid[i].shape[2:5] != x[i].shape[2:5]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nz, nx, ny, i)

                y = x[i].sigmoid()
                # in training mode these are calculated in the loss function
                y[..., 0:3] = (y[..., 0:3] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # zxy
                y[..., 3:6] = (y[..., 3:6] * 2) ** 2 * self.anchor_grid[i]  # dwh

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nz=20, nx=20, ny=20, i=0):
        # i is the detection layer index
        d = self.anchors[i].device
        zv, yv, xv = torch.meshgrid([torch.arange(nz).to(d), torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        grid = torch.stack((zv, xv, yv), 3).expand((1, self.na, nz, ny, nx, 3)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 1, 3)).expand((1, self.na, nz, ny, nx, 3)).float()
        return grid, anchor_grid


def initialize_weights(model):
    """Initializes parameters for layers that need extra parameters set.

    Args:
        model (torch.Module): YOLO model to initialize weights for
    """
    for m in model.modules():
        t = type(m)
        if t is nn.Conv3d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm3d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """Scales img(bs,1,z,y,x) by ratio optionally constrained to a multiple of gs

    Args:
        img (torch.Tensor): image to be resized
        ratio (float, optional): ratio by which to resize image. Defaults to 1.0.
        same_shape (bool, optional): Whether image should maintain the same shape or be padded/cropped to a multiple of gs. Defaults to False.
        gs (int, optional): Factor to constrain model size to a multiple of. Usually stride. Defaults to 32.

    Returns:
        (torch.Tensor): rescaled image
    """
    if ratio == 1.0:
        return img
    else:
        d, h, w = img.shape[2:]
        s = (int(d * ratio), int(h * ratio), int(w * ratio))  # new size
        # img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        img = F.interpolate(img, size=s, mode='trilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            d, h, w = (math.ceil(x * ratio / gs) * gs for x in (d, h, w))
        # YOLOv5 reverses x and y here, I'm not sure why but have emulated that
        return F.pad(img, [0, d - s[0], 0, w - s[2], 0, h - s[1]], mode='reflect')


def parse_model(d, ch):
    """Configures the model using hyperparameters from the model yaml file.
    Note: model yaml file (e.g. yolov3Ds.yaml) is distinct from the
    training hyperparameter file (e.g. hyp.finetune.yaml)

    Args:
        d (Dict[str]): dictionary of model hyperparameters.
        ch (int): input channels.

    Returns:
        (torch.Module): Configured model.
        (List[int]): List of layers from which to save output.
    """
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 3) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 7)  # number of outputs = anchors * (classes + 7 [zxydwh + conf (I think)])

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # this loop configures all of the model's layers
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['neck'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        # n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, Bottleneck, C3, SPPF]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [C3]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm3d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Detect:
            args.append([ch[x] for x in f])
            # args[1] for the Detect layer stores the nested list of anchors
            if isinstance(args[1], int):  # number of anchors given instead of a list of anchors
                # args[1] = [list(range(args[1] * 2))] * len(f)
                args[1] = [list(range(args[1] * 3))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number of parameters
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class Model(nn.Module):
    def __init__(self, cfg=ROOT / 'models3D/yolo3Ds.yaml', ch=1, nc=None, anchors=None):
        """3D YOLO model base class.

        Args:
            cfg (Dict[str] or str, optional): Model configuration dictionary or path to yaml file containing model hyperparameters. Defaults to ROOT/'models3D/yolov3Ds.yaml'.
            ch (int, optional): input channels. Defaults to 1.
            nc (int, optional): number of classes. Defaults to None.
            anchors (List[int], optional): Anchors for Detect layer. Defaults to None.
        """
        super().__init__()
        # loads or sets the model dictionary, which stores its hyperparameters
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            # self.yaml_file = Path(cfg).name
            with open(cfg, errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # parse_model configures the layers and passes in the anchors hyperparameter to the Detect layer
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.inplace = self.inplace

            s = 256
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s, s))])  # forward           
            self.stride = m.stride
            m.anchors /= m.stride.view(-1, 1, 1)

            self._initialize_biases()  # only run once

        # Initialize weights and biases
        initialize_weights(self)

    def forward(self, x):
        return self._forward_once(x)


    def _forward_once(self, x):
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # Currently the 2D code, may need to be adjusted in the future if performance is poor
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (default_size / s) ** 3)  # obj (8 objects per 350^3 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def fuse(self):   # fuse model Conv3d() + BatchNorm3d() layers
        for m in self.model.modules():
            if isinstance(m, (Conv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False):
        model_info(self, verbose)
