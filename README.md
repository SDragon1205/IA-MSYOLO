## 1. Download Pretrained Model
Download yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT

----
## 2. Download Dataset Follow the DATASET_INSTALLATION.md

----
## 3. Modify core/config.py to train model on your own dataset
```=python
#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

__C.YOLO.CLASSES              = "./data/classes/1cls.names"
__C.YOLO.ANCHORS              = [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
__C.YOLO.ANCHORS_V3           = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
__C.YOLO.ANCHORS_TINY         = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.STRIDES_TINY         = [16, 32]
__C.YOLO.XYSCALE              = [1.2, 1.1, 1.05]
__C.YOLO.XYSCALE_TINY         = [1.05, 1.05]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5


# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATHS          = [
    "datasets/data_selection/anno/train_1cls.txt",
    "datasets/data_selection_2/anno/train_1cls.txt",
    "datasets/data_selection_3/anno/train_1cls.txt",
    "datasets/Taiwan_trafficlight.v1.coco/anno/train_1cls.txt"
]

__C.TRAIN.BATCH_SIZE          = 8 #2
# __C.TRAIN.INPUT_SIZE        = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = 608
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 20
__C.TRAIN.SECOND_STAGE_EPOCHS   = 40



# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATHS           = [
    "datasets/data_selection/anno/val_1cls.txt",
    "datasets/data_selection_2/anno/val_1cls.txt",
    "datasets/data_selection_3/anno/val_1cls.txt"
]
__C.TEST.BATCH_SIZE           = 2
__C.TEST.INPUT_SIZE           = 608
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.25
__C.TEST.IOU_THRESHOLD        = 0.5



```

## 4. Train model and evaluate your own model
```
# train your yolov4 tiny model on custom dataset 
# You can modify dataset you want to train  on core/config.py 
# Use pretrained weight can get better result.
python ./train.py --tiny --model yolov4 --save_dir ./checkpoints/test --weights ./data/yolov4-tiny.weights

# convert to tensorflow save_model format
python ./save_model.py --weights ./checkpoints/test/ckpt/final.ckpt --output ./checkpoints/test/save_model_final --tiny --input_size 608

# calculate mAP
python ./evaluate_map.py --weights ./checkpoints/test/save_model_final/ --framework tf --input_size 608 --annotation_path ./datasets/data_selection_mix/anno/val_1cls_filter_small.txt
```


## 5. Other Usuage of script
### Visualize Anntation
-   Usuage
    ```
    # draw bounding box on the image so that you can checkout whether there are noise in the labels.
    python visulaize_anno.py --anno ANNOTATION_PATH
    ```
-   Output file structure
    ```
    visualize_anno\
        image_w_box\
            0_image_[serial_number].jpg
            1_image_[serial_number].jpg
            2_image_[serial_number].jpg
            ......
        image_wo_box\
            0_image_[serial_number].jpg
            1_image_[serial_number].jpg
            2_image_[serial_number].jpg
            ......
    ```
----

### Visualize Augmentation 
-   Usuage
    ```
    # show the augmentation result on image to check whether your augmentation result are correct or wrong
    # You can modify dataset you want to show on core/config.py 
    python visualize_augmentation.py --model --yolov4
    ```
-   Output file structure
    ```
    visualize_anno\
        augmentation\
            1.jpg
            2.jpg
            3.jpg
            ......
    ```
----

### Detection on image: Use Case 1 => detect single image
-   Usuage
    ```
    # image_type=image => detected single image, and output detection result to result.png
    python detect_new.py --framework tf --weights PATH_TO_SAVE_MODEL \
        --size 608\
        --tiny --model yolov4\
        --image_type image
        --image_path PATH_TO_IMAGE\
    ```
-   Output file structure
    ```
    result.png
    ```

----
### Detection on image: Use Case 2 => detect images in a folder
-   Usuage
    ```
    # image_type=folder => detected all the image in the folder, and output detection result to "output" folder 
    python detect_new.py --framework tf --weights PATH_TO_SAVE_MODEL \
        --size 608\
        --tiny --model yolov4\
        --image_type folder
        --image_path PATH_TO_FOLDER\
    ```
-   Output file structure
    ```
    output\
        all_box.txt
        0_image_[serial_number].jpg
        1_image_[serial_number].jpg
        2_image_[serial_number].jpg
        3_image_[serial_number].jpg
        ......
    ```

----
### Detection on image: Use Case 3 => detect images in a annotation file(contain list of image path)
-   Usuage
    ```
    # image_type=folder => detected all the image in the folder, and output detection result to "output" folder 
    python detect_new.py --framework tf --weights PATH_TO_SAVE_MODEL \
        --size 608\
        --tiny --model yolov4\
        --image_type file \
        --image_path PATH_TO_ANNOTATION_FILE\
    ```
-   Output file structure
    ```
    output\
        all_box.txt
        0_image_[serial_number].jpg
        1_image_[serial_number].jpg
        2_image_[serial_number].jpg
        3_image_[serial_number].jpg
        ......
    ```

----

### Other usuage of script to be add
```bash
python benchmarks.py --size 416 --model yolov4 --weights ./data/yolov4.weights
python detectvideo.py
```
