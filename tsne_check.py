from pickle import FALSE
import sys, os, cv2
from core.common import MishLayer,BatchNormalization
from absl import app, flags, logging
from absl.flags import FLAGS
from core.accumulator import Accumulator
import os, shutil
import tensorflow as tf
from core.yolov4 import YOLO, compute_da_loss_instance, compute_loss, compute_da_loss, decode_train
from core.dataset_tiny import Dataset, tfDataset, tfAdversailDataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all, draw_bbox
from tqdm import tqdm
import tensorflow_model_optimization as tfmot
import time
import core.common as common
from filters_lowlight import *
from sklearn import manifold

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration(
#     gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
import shutil

dirPath1 = '/mnt/HDD1/iamsyolo/ia-yolov4-tflite/tsne/config/__pycache__'
dirPath2 = '/mnt/HDD1/iamsyolo/ia-yolov4-tflite/tsne/config/augmentation'

flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_boolean('qat', False, 'train w/ or w/o quatize aware')
flags.DEFINE_string('save_dir', 'tsne', 'save model dir')
flags.DEFINE_string('weights', './checkpoints/test_iayolo1/ckpt/final.ckpt', 'path to weights file')
flags.DEFINE_string('weights2', './checkpoints/cnnpp-416', 'path to weights file')
tf.config.optimizer.set_jit(True)

def apply_quantization(layer):
    if isinstance(layer, tf.python.keras.engine.base_layer.TensorFlowOpLayer):
         return layer
    return tfmot.quantization.keras.quantize_annotate_layer(layer)

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def main(_argv):
    try:
        shutil.rmtree(dirPath1)
        shutil.rmtree(dirPath2)
    except OSError as e:
        print(f"Error:{ e.strerror}")
    print("""
    **************          ****            **            **      ************
    **************       ****  ****         ***           **      **
          **           ****      ****       ** **         **      **
          **             ****               **   **       **      **
          **                ****            **     **     **      ************
          **                   ****         **       **   **      **
          **           ****      ****       **         ** **      **
          **             ****  ****         **           ***      **
          **                ****            **            **      ************
    """)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #tf.config.experimental.set_visible_devices( devices = physical_devices [0], device_type = 'GPU' )
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    os.makedirs(os.path.join(FLAGS.save_dir, 'config'), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.save_dir, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.save_dir, 'pic'), exist_ok=True)
    '''
    testset_source = tfDataset(FLAGS, is_training=False, filter_area=123, use_imgaug=False, adverserial=False).dataset_gen()
    testset_target = tfDataset(FLAGS, is_training=False, filter_area=64, use_imgaug=False, adverserial=True).dataset_gen()
    testset = tf.data.Dataset.zip((testset_source, testset_target))
    '''
    trainset_source = tfDataset(FLAGS, is_training=True, filter_area=123, use_imgaug=False, adverserial=False).dataset_gen()
    trainset_target = tfDataset(FLAGS, is_training=True, filter_area=64, use_imgaug=False, adverserial=True).dataset_gen()
    trainset = tf.data.Dataset.zip((trainset_source, trainset_target))

    copytree('./core', os.path.join(FLAGS.save_dir, 'config'))
    shutil.copy2(sys.argv[0], os.path.join(FLAGS.save_dir, 'config', os.path.basename(sys.argv[0])))
    with open(os.path.join(FLAGS.save_dir, 'command.txt'), 'w') as f:
        f.writelines(' '.join(sys.argv))
    f.close()

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
    # !!!!!!!!!
    output_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny, dc_head_type=2) # [1] dc_head_type=2 for tSNE
    # !!!!!!!!!
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
        'da_m': da_maps[0],                     # 底層Feature map, 38x38x256
        'raw_bbox_l': bbox_tensors[2],
        'bbox_l': bbox_tensors[3],
        'da_l': da_maps[1],                     # 底層Feature map, 19x19x512
    })
    model.summary() # [2] model.load_weights()

    #infer = tf.keras.models.load_model(FLAGS.weights)
    model.load_weights(FLAGS.weights)
    CNN_PP = tf.keras.models.load_model(FLAGS.weights2)
    dip = DIP(cfg)
    '''
    print('Restoring weights from: %s ... ' % FLAGS.weights)
    model.summary()
    print('Restoring weights2 from: %s ... ' % FLAGS.weights2)
    CNN_PP.summary()
    '''
    

    def get_tsne_image(image_data, train_target):
        ##get the mask
        batch_size = tf.shape(image_data)[0]
        #print('\ntrain_target[0][0]的 shape: ', np.array(train_target[0][0]).shape())
        source_mask_da_m = tf.reduce_any(train_target[0][0][:batch_size//2, ...,5:] > 0, axis=(3,4)) ##train_target[0][0]:training label, shape: batch, train size(長), train size(寬), 3(anchor), 5+class
        #print('\ntrain_target[0][0]的new shape: ', np.array(train_target[0][0]).shape())
        
        #print('\nsource mask da m 的 type: ', type(source_mask_da_m))
        #print('\nsource mask da m 的 shape: ', source_mask_da_m.shape)
        '''
        source_mask_da_l = tf.reduce_any(train_target[1][0][:batch_size//2, ...,5:] > 0, axis=(3,4)) #true false
        source_mask=[
            tf.cast(tf.expand_dims(source_mask_da_m, axis=-1), dtype=tf.float32), 
            tf.cast(tf.expand_dims(source_mask_da_l, axis=-1), dtype=tf.float32)
        ]#0, 1
        '''
        target_mask_da_m = tf.reduce_any(train_target[0][0][batch_size//2:, ...,5:] > 0, axis=(3,4)) 
        #print('\ntarget mask da m 的 shape: ', target_mask_da_m.shape)
        '''
        target_mask_da_l = tf.reduce_any(train_target[1][0][batch_size//2:, ...,5:] > 0, axis=(3,4))
        target_mask=[
            tf.cast(tf.expand_dims(target_mask_da_m, axis=-1), dtype=tf.float32), 
            tf.cast(tf.expand_dims(target_mask_da_l, axis=-1), dtype=tf.float32), 
        ]
        '''

        mask_da_m = tf.concat([source_mask_da_m, target_mask_da_m], axis=0 )
        #print('\nmask_da_m 的 shape: ', mask_da_m.shape)

        mask_da_m = tf.expand_dims(mask_da_m, axis=-1)
        #print('\nmask_da_m 的 new shape: ', mask_da_m.shape)

        expand_mask_da_m = mask_da_m
        for _ in range(256-1):
            expand_mask_da_m = tf.concat([expand_mask_da_m, mask_da_m], axis=3)

        expand_mask_da_m = tf.cast(expand_mask_da_m, tf.float32)
        #print('\nexpand_mask_da_m 的 shape: ', expand_mask_da_m.shape)
        #print('\nexpand_mask_da_m : ', expand_mask_da_m)
        #get feature
        filtered_image_batch = image_data
        input_data = tf.image.resize(image_data, [256, 256], method=tf.image.ResizeMethod.BILINEAR)#調整圖片大小為256*256，方法為雙線性插植
        filter_features = CNN_PP(input_data, training = False)#CNN-PP module
        filtered_image_batch = dip(filtered_image_batch, filter_features)
        print('\nfiltered_image_batch = ', filtered_image_batch)
        dict_result = model(filtered_image_batch, training=False)
        #for key, tensor in dict_result.items():
        #    print(f'{key:10s} {tensor.shape}')
        conv_mdc = dict_result['da_m']
        #print('\nconv_mdc 的 type: ', type(conv_mdc))
        #print('\nconv_mdc 的 shape: ', conv_mdc.shape)
        
        feature_masked = tf.multiply(expand_mask_da_m, conv_mdc)

        batch_size = 16
        width = height = 38
        fm = np.empty(shape=256)
        for batch in range(batch_size):
            for w in range(width):
                for h in range(height):
                    if(tf.math.reduce_sum(feature_masked[batch][w][h]) > 0):
                        fm = np.append(fm, feature_masked[batch][w][h])
        #feature_masked = tf.reduce_any(feature_masked[...] > 0, axis=(1,2)) 
        fm = np.reshape(fm, (-1, 256))
        #print('\nfeature_masked的new shape: ', fm.shape)
            
        #print('\nfeature_masked 的 shape: ', feature_masked.shape)
        #print(feature_masked)
        
        # a5=0
        # for a1 in range(16):
        #     if(a5==0):
        #         for a2 in range(38):
        #             if(a5==0):
        #                 for a3 in range(38):
        #                     if(a5==0):
        #                         for a4 in range(256):
        #                             if(feature_masked[a1][a2][a3][a4]>0):
        #                                 a5+=1
        #                                 print('a1 = ',a1)
        #                                 print('a2 = ',a2)
        #                                 print('a3 = ',a3)
        #                                 print('a4 = ',a4)
        #                                 break
        #                     else: break
        #             else: break
        #     else: break
        # print('a5 = ',a5)
        
        # for i in range(256):
        #     print(feature_masked[a1-1][a2-1][a3-1][i])
        
        return fm
        #sys.exit()
        

    
    def tSNE(feature_masked):
        
        #t-SNE
        tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(feature_masked)


        # scale and move the coordinates so they fit [0; 1] range

        def scale_to_01_range(x):
            # compute the distribution range
            value_range = (np.max(x) - np.min(x))
            # move the distribution so that it starts from zero
            # by extracting the minimal value from all its values
            starts_from_zero = x - np.min(x)
            # make the distribution fit [0; 1] by dividing by its range
            return starts_from_zero / value_range

        # extract x and y coordinates representing the positions of the images on T-SNE plot
        tx = tsne[:, 0]
        ty = tsne[:, 1]
        
        tx = scale_to_01_range(tx)
        ty = scale_to_01_range(ty)

        print(tx)
        print(ty)
    

    feature_masked = np.empty(shape=(0,256))
    ### run 迴圈
    #for epoch in range(1, 1+first_stage_epochs + second_stage_epochs):
    #tmp = time.time()
    total=len(trainset)
    with tqdm(total=len(trainset), ncols=200) as pbar:
        for iter_idx, data_item in enumerate(trainset):
            source_data_dict=data_item[0]
            target_data_dict=data_item[1]

            source_images=source_data_dict['images']
            # source_data_dict['label_bboxes_m']=(batch, 38, 38,4+1+)
            source_train_targets=[
                [source_data_dict['label_bboxes_m'], source_data_dict['bboxes_m']], 
                [source_data_dict['label_bboxes_l'], source_data_dict['bboxes_l']], 
            ]
            target_images=target_data_dict['images']
            target_train_targets=[
                [target_data_dict['label_bboxes_m'], target_data_dict['bboxes_m']], 
                [target_data_dict['label_bboxes_l'], target_data_dict['bboxes_l']], 
            ]

            images = tf.concat([source_images, target_images], axis=0)
            train_targets = [
                [
                    tf.concat([source_data_dict['label_bboxes_m'], target_data_dict['label_bboxes_m']], axis=0),
                    tf.concat([source_data_dict['bboxes_m'], target_data_dict['bboxes_m']], axis=0),
                ],
                [
                    tf.concat([source_data_dict['label_bboxes_l'], target_data_dict['label_bboxes_l']], axis=0),
                    tf.concat([source_data_dict['bboxes_l'], target_data_dict['bboxes_l']], axis=0),
                ]
            ]

            #data_time=time.time()-tmp
            batch_size = images.shape[0]
            #tmp=time.time()

            # if(iter_idx == 0):
            #     feature_masked = get_tsne_image(images, train_targets)
            #     tSNE(feature_masked)
            #     sys.exit()
            # else:
            feature_masked = tf.concat([feature_masked, get_tsne_image(images, train_targets)], axis=0 )
            sys.exit()
            print('\nfeature_masked 的 shape: ', feature_masked.shape)
            if(iter_idx%25==0):
                np.save('big_array', feature_masked)
            print('iter = ', iter_idx, "/", total)
    print('\nfinal feature masked shape: ',feature_masked.shape)       
    

    

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
