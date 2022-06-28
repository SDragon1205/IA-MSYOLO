from core.common import MishLayer,BatchNormalization
from absl import app, flags, logging
from absl.flags import FLAGS
from core.accumulator import Accumulator
import os, shutil
import tensorflow as tf
from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset_tiny import Dataset, tfDataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import draw_bbox
import cv2
from tqdm import tqdm
import tensorflow_model_optimization as tfmot
import time

flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
tf.config.optimizer.set_jit(True)

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def main(_argv):
    folder='aug_img'
    os.makedirs('visualize_anno', exist_ok=True)
    os.makedirs(f'visualize_anno/{folder}', exist_ok=True)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    # trainset = Dataset(FLAGS, is_training=True, filter_area=123)
    # testset = Dataset(FLAGS, is_training=False, filter_area=123)
    #########################################################
    mytrainset = tfDataset(FLAGS, is_training=False, filter_area=123).dataset_gen()
    trainset = Dataset(FLAGS, is_training=False, filter_area=123)

    for i, (my_batch, ori_batch) in enumerate(zip(mytrainset, trainset)):
        ############################################################
        ############################################################
        # (b,h,w,3)
        my_batch_image=(my_batch[0].numpy() * 255.0).astype(np.uint8)
        my_fpn1 = my_batch[1].numpy()
        my_fpn2 = my_batch[3].numpy()
        ############################################################
        ############################################################        
        batch_image=(ori_batch[0] * 255.0).astype(np.uint8)
        fpn1 = ori_batch[1][0][0]
        fpn2 = ori_batch[1][1][0]
    
        ############
        for var, var_name in zip(
            [my_batch_image,my_fpn1,my_fpn2,batch_image,fpn1,fpn2],
            ['my_batch_image','my_fpn1','my_fpn2','batch_image','fpn1','fpn2']):
            print(f'{var_name:20s}, {str(type(var)):20s}, {str(var.shape):20s}, {str(var.dtype):10s} max: {var.max():6.2f} min: {var.min():6.2f} mean: {var.mean():6.2f}')
        

        img_eq = (my_batch_image==batch_image).astype(np.int32).sum()
        fpn1_eq = (my_fpn1 == fpn1).astype(np.int32).sum()
        fpn2_eq = (my_fpn2 == fpn2).astype(np.int32).sum()
        print(f'img_eq: {img_eq}/{np.prod(my_batch_image.shape)} fpn1_eq: {fpn1_eq}/{np.prod(my_fpn1.shape)} fpn2_eq: {fpn2_eq}/{np.prod(my_fpn2.shape)}')
        print('*'*40)
        cv2.imwrite('batch_image_my.png', my_batch_image[0])
        cv2.imwrite('batch_image.png', batch_image[0])

        if i == 100: break


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
