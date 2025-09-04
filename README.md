# Differences Between YOLOv12 and YOLOv11: Architecture and Attention Mechanisms

This repository contains the comparison of YOLOv11 and YOLOv12 and an example of hyperparameter optimization on top of them.

## Overview

The paper compares YOLOv11’s CNN-based architecture with YOLOv12’s attention-centric design using VOC dataset metrics and ablation studies. Hyperparameter optimization, guided by Ultralytics documentation, utilized Ray Tune to adjust learning rate, weight decay, momentum, and batch size. YOLOv12’s attention mechanisms benefited from larger batch sizes for improved performance.
## Dataset

This implementation uses the ADE20K dataset:
- A benchmark dataset for object detection, segmentation, and classification, containing 20 object categories.
- ~17,000 images split into training (~11,500) and validation/test (~5,500) sets with bounding boxes and class labels.
- Used for evaluating object detection models like YOLO, providing diverse scenes with multiple objects per image.

The dataset should be organized in the following structure:
```
datasets/VOC/
├── images
│   ├── test2007
│   ├── train2007
│   ├── train2012
│   ├── val2007
│   ├── val2012
│   └── VOCdevkit
│       ├── VOC2007
│       └── VOC2012
└── labels
    ├── test2007
    ├── train2007
    ├── train2012
    ├── val2007
    └── val2012
```

You can download the dataset from the [PASCAL VOC 2012 DATASET](https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset).
  
## Getting Started

To run this code, please use the train.py to initiate training

## Performance

Hyperparameter optimization using Ray Tune, as guided by Ultralytics documentation, resulted in YOLOv12 achieving a post-training mAP50 of 77.37% and mAP50-95 of 56.65% on the COCO dataset, slightly underperforming YOLOv11’s 83.92% mAP50 and 64.49% mAP50-95, despite its attention-based enhancements.
