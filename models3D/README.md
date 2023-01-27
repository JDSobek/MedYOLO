This folder contains the code necessary to build the 3D models.

Primary changes from YOLOv5:

Relevant modules changed from 2D versions to 3D versions.

Reorganization of model from backbone + head to backbone + neck + head

Addition of 4th detection layer

Download functionality removed, there are currently no initial weights to draw from.

The overall architecture largely follows YOLOv5's architecture.  Changing depth_multiple and width_multiple in the model yaml file should allow you to generate larger models.

Anchors are dataset dependent.  utils3D/anchorcalculator.py can help calculate an initial set of anchors using kmeans clustering and your training set bounding boxes.  This requires an initial set of values which will be used to determine how many anchors to calculate and how many detection layers to calculate anchors for.

It will also save a plot of the number of anchors vs. the reported distortion from kmeans to help choose how many anchors to use.