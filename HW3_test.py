import sys, cv2
from core.common import MishLayer,BatchNormalization
from absl import app, flags, logging
from absl.flags import FLAGS
from core.accumulator import Accumulator
import os, shutil
import tensorflow as tf
from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset_tiny import Dataset, tfDataset, tfAdversailDataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all
from tqdm import tqdm
import tensorflow_model_optimization as tfmot
import time

flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_string('weights', None, 'pretrained weights')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_boolean('qat', False, 'train w/ or w/o quatize aware')
flags.DEFINE_string('save_dir', 'checkpoints/yolov4_tiny', 'save model dir')
tf.config.optimizer.set_jit(True)

def apply_quantization(layer):
    if isinstance(layer, tf.python.keras.engine.base_layer.TensorFlowOpLayer):
         return layer
    return tfmot.quantization.keras.quantize_annotate_layer(layer)

def qa_train(model):
    # qa_train part
    quantize_model = tfmot.quantization.keras.quantize_model
    quantize_scope = tfmot.quantization.keras.quantize_scope
    
    annotated_model = tf.keras.models.clone_model(model,
        clone_function=apply_quantization,
    )

    quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    return quant_aware_model

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')


    trainset = Dataset(FLAGS, is_training=True, filter_area=123)
    testset = Dataset(FLAGS, is_training=False, filter_area=123)
    trainset_source = tfDataset(FLAGS, is_training=True, filter_area=123, use_imgaug=True).dataset_gen()
    trainset_target = tfAdversailDataset(FLAGS, is_training=True, filter_area=123, use_imgaug=True).dataset_gen()


    logdir = os.path.join(FLAGS.save_dir, 'logs')
    isfreeze = False
    steps_per_epoch = len(trainset)
    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch

    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

    freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)

    # [conv_mbox(38,38), conv_lbbox(19,19)]
    output_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny, use_dc_head=True)
    feature_maps = output_maps[:2]
    da_maps = output_maps[2:]

    # Decoding YOLOv4 Output
    if FLAGS.tiny:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    else:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, {
        'raw_bbox_m': bbox_tensors[0],          # tensor size of feature map
        'bbox_m': bbox_tensors[1],
        'da_m': da_maps[0],
        'raw_bbox_l': bbox_tensors[2],
        'bbox_l': bbox_tensors[3],
        'da_l': da_maps[1],
    })

    for data_dict in trainset_target:
        images=data_dict['images']
        result_dict=model(images)
        for key, value in result_dict.items():
            print(f'{key}: {value.shape}')
        break

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
