import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os , shutil

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_integer('size', 608, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image_type', "image", '(image, folder or file)')
flags.DEFINE_string('image_path', '~/data/gis/images/val', 'path to input image')
flags.DEFINE_string('image_path_prefix', '', 'the prefix path of image')
flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float('score', 0.2, 'score threshold')


def detect_func(original_image, input_size, interpreter):
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    if FLAGS.framework == 'tflite':
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], images_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
            boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
        else:
            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
    else:
        # saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        # infer = saved_model_loaded.signatures['serving_default']
        # batch_data = tf.constant(images_data)
        # pred_bbox = infer(batch_data)
        # for key, value in pred_bbox.items():
        #     boxes = value[:, :, 0:4]
        #     pred_conf = value[:, :, 4:]

        batch_data = tf.constant(images_data)
        pred_bbox = interpreter(batch_data)
        boxes = pred_bbox[:, :, 0:4]
        pred_conf = pred_bbox[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    num_of_box = pred_bbox[3][0]
    per_img_box = []
    for i in range(num_of_box):
        final_box=pred_bbox[0][0][i][:4]
        per_img_box.append(list(final_box))
        
    #  valid_detections.numpy() 最後被保留的框的數量
    image = utils.draw_bbox(original_image, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    return image, per_img_box
     

def yolobbox2bbox(size,box):
    W=size[0]
    H=size[1]
    x,y = box[0],box[1]
    w,h = box[2],box[3]
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    x1, y1, x2, y2= int(round(x1*W)),int(round(y1*H)), int(round(x2*W)), int(round(y2*H))
    return x1, y1, x2, y2

def main(_argv):
    output_path = './output'
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    image_path = FLAGS.image_path
    image_type = FLAGS.image_type
    shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        interpreter = tf.keras.models.load_model(FLAGS.weights)

    
    if image_type == 'image':
        original_image = cv2.imread(image_path)
        image = detect_func(original_image,input_size, interpreter)
        cv2.imwrite(output_path+'result.png', image)
    elif image_type == 'folder':
        all_listdir=os.listdir(image_path)
        with open(os.path.join(output_path, 'all_box.txt'),'w') as f: 
            for s,img_name in enumerate(all_listdir):
                original_image = cv2.imread(os.path.join(image_path,img_name))
                H, W = original_image.shape[:2]
                image, per_img_box = detect_func(original_image,input_size,interpreter)
                per_line=img_name+' '
                for q in per_img_box:
                    k=list(yolobbox2bbox((H, W),q))
                    for q2 in range(len(k)):
                        if q2==3:
                            per_line+=str(k[q2])+' '
                        else:
                            per_line+=str(k[q2])+','
                print('per_line:',per_line)
                f.write(per_line+'\n')
                cv2.imwrite(os.path.join(output_path, img_name), image)
    elif image_type == 'file':
        outer_image_path=FLAGS.image_path_prefix
        with open(image_path, 'r') as f:
            lines = f.readlines()
            inner_image_paths = [line.split()[0] for line in lines]

        with open(os.path.join(output_path, 'all_box.txt'),'w') as f: 
            for s, inner_image_path in enumerate(inner_image_paths):
                print(os.path.join(outer_image_path, inner_image_path))
                original_image = cv2.imread(os.path.join(outer_image_path, inner_image_path))
                H, W = original_image.shape[:2]
                img_name = os.path.split(inner_image_path)[-1]

                image, per_img_box = detect_func(original_image, input_size, interpreter)
                per_line=inner_image_path+' '
                for q in per_img_box:
                    k=list(yolobbox2bbox((H, W),q))
                    for q2 in range(len(k)):
                        if q2==3:
                            per_line+=str(k[q2])+' '
                        else:
                            per_line+=str(k[q2])+','
                print('per_line:',per_line)
                f.write(per_line+'\n')
                cv2.imwrite(os.path.join(output_path, f'{s:4d}_{img_name}'), image)
    else:
        print('type format wrong')

    

    


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass