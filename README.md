# MED_YOLO3D
A 3D bounding box detection model for medical data.

Model architecture and input pipeline are based off YOLOv5's (https://github.com/ultralytics/yolov5) approach.  Layers have been modified to use 3-D convolutions and other modules compatible with 3-D input data.  Data pipelines are constructed to read 3-D NIfTI images.  The development repository was initially forked from YOLOv5 commit ed887b5976d94dc61fa3f7e8e07170623dc7d6ee.  YOLOv5 code, particularly the utils folder, is reused where possible with unused code and unimplemented functionality (such as download/coco functionality, certain methods of augmentation, and extra classification models) removed.  Logging is also unimplemented.

Training and inference should work largely the same as YOLOv5.  Data should be organized into folders like this:
/parent_directory/images/train/
/parent_directory/images/val/
/parent_directory/labels/train/
/parent_directory/labels/val/

Data yaml files can be modeled after /data/example.yaml and should be saved in the data folder.

Label format: Class-number Center-Z Center-X Center-Y Depth Width Height
Center positions and edge lengths should be given as fractions of the whole, like in YOLOv5.

The model tested is based on YOLOv5s, but should scale up using the width_multiple and depth_multiple parameters similarly to YOLOv5 models.


Installation process:
If using CUDA, install appropriate versions of cudnn and cudatoolkit for your hardware before installing requirements.txt, e.g.:

```bash
$ conda install cudnn=8.2 cudatoolkit=11.3
$ conda install --file requirements.txt -c pytorch
```

Training:

```bash
$ python train.py --data example.yaml --epochs 200
```

Inference:

```bash
$ python detect.py --source /parent_directory/nifti_folder/ --weights ./runs/train/exp/weights/best.pt
```


Additional Details:
Similarly to how YOLOv5 reshapes input data to squares, MedYOLO reshapes input data to cubic volumes.  An equivalent to rectangular training is currently not implemented, so input examples of dramatically different shapes may be distorted relative to each other and may have different performance.
Because of this reshaping, the imgsz parameter should be chosen carefully to balance GPU resources, input resolution, and batch size.
Depending on GPU resources, this reshaping will probably reduce the X and Y resolution of your data.  As a result, during testing the model has not performed well on small objects.
In testing, results were found to depend heavily on the number of training examples, with approximately 400 training examples resulting in poor results and 600 training examples resulting in much better results.