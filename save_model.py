import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from core.yolov4 import YOLO, decode, decode_train, filter_boxes
import core.utils as utils
from core.config import cfg
from keras_flops import get_flops
import tensorflow_model_optimization as tfmot
import core.common as common
from filters_lowlight import *
import sys

flags.DEFINE_string('weights', './checkpoints/test_iayolo1/ckpt/final.ckpt', 'path to weights file')
flags.DEFINE_string('weights2', './checkpoints/test_iayolo1/ckpt/cnn_pp_final.ckpt', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4-417', 'path to output')
flags.DEFINE_string('output2', './checkpoints/cnnpp-417', 'path to output')
flags.DEFINE_boolean('tiny', True, 'is yolo-tiny or not')
flags.DEFINE_integer('input_size', 608, 'define input size of export model')
flags.DEFINE_float('score_thres', 0.2, 'define score threshold')
flags.DEFINE_string('framework', 'tf', 'define what framework do you want to convert (tf, trt, tflite)')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('dataset', 'data/dataset/gis_val_1.txt', 'yolov3 or yolov4')
flags.DEFINE_boolean('qat', False, 'For Qauntize Aware Training')

def apply_quantization(layer):
    
    #if isinstance(layer, tf.keras.layers.UpSampling2D) or isinstance(layer, tf.keras.layers.BatchNormalization) or isinstance(layer, tf.python.keras.engine.base_layer.TensorFlowOpLayer):
    #   return layer
    if isinstance(layer, tf.python.keras.engine.base_layer.TensorFlowOpLayer):
         return layer
    return tfmot.quantization.keras.quantize_annotate_layer(layer)

    # if isinstance(layer , tf.keras.layers.Conv2D):
    #     return tfmot.quantization.keras.quantize_annotate_layer(layer)
    # return layer

def qa_train(model):
    # qa_train part
    quantize_model = tfmot.quantization.keras.quantize_model
    quantize_scope = tfmot.quantization.keras.quantize_scope
    
    annotated_model = tf.keras.models.clone_model(model,
        clone_function=apply_quantization,
    )

    quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    # quant_aware_model.summary()

    return quant_aware_model

def save_tf():
  STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

  input_layer = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
  feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)

  # CNN-PP module
  input_layer2 = tf.keras.layers.Input([256, 256, 3])
  filter_parameters = common.extract_parameters_2(input_layer2, cfg)

  bbox_tensors = []
  prob_tensors = []
  if FLAGS.tiny:
    for i, fm in enumerate(feature_maps):
      if i == 0:
        output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      else:
        output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])
  else:
    for i, fm in enumerate(feature_maps):
      if i == 0:
        output_tensors = decode(fm, FLAGS.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      elif i == 1:
        output_tensors = decode(fm, FLAGS.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      else:
        output_tensors = decode(fm, FLAGS.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])
  pred_bbox = tf.concat(bbox_tensors, axis=1)
  pred_prob = tf.concat(prob_tensors, axis=1)
  if FLAGS.framework == 'tflite':
    pred = (pred_bbox, pred_prob)
  elif FLAGS.framework == 'tf':
    boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=FLAGS.score_thres, input_shape=tf.constant([FLAGS.input_size, FLAGS.input_size]))
    pred = tf.concat([boxes, pred_conf], axis=-1)
  else:
    raise NotImplementedError(f'No such framework {FLAGS.framework}')
  ## ori code
  print('========================================== Load model ========================================')
  model = tf.keras.Model(input_layer, pred)
  CNN_PP = tf.keras.Model(input_layer2, filter_parameters)

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
  weight_model = tf.keras.Model(input_layer, {
        'raw_bbox_m': bbox_tensors[0],          # tensor size of feature map
        'bbox_m': bbox_tensors[1],
        'raw_bbox_l': bbox_tensors[2],
        'bbox_l': bbox_tensors[3],
    })
  weight_model2 = tf.keras.Model(input_layer2, filter_parameters)
  # model.summary()
  #####################################################################################################
  ############################################ Quantize structure #####################################
  print (FLAGS.model+"      ....................................... ")
  if (FLAGS.qat):
    model = qa_train(model)
    CNN_PP = qa_train(CNN_PP)
    print('Quantization Model .....................................')
  #####################################################################################################

  print('========================================== Load weight ========================================')
  weight_model.load_weights(FLAGS.weights)
  weight_model2.load_weights(FLAGS.weights2)
  print('========================================== model summary ========================================')
  model.summary()
  CNN_PP.summary()
  print('========================================== save model ========================================')
  model.save(FLAGS.output)
  CNN_PP.save(FLAGS.output2)

  # converter = tf.lite.TFLiteConverter.from_keras_model(model)

  # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  # converter.optimizations = [tf.lite.Optimize.DEFAULT]
  # converter.allow_custom_ops = True
  # converter.representative_dataset = representative_data_gen

  # tflite_model = converter.convert()
  # open("./checkpoints/yolov4-tiny-gis-320-int8.tflite", 'wb').write(tflite_model)

  # flops = get_flops(model, batch_size=1)
  # print(f"FLOPS: {flops / 10 ** 9:.03} G")
  

def main(_argv):
  save_tf()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
