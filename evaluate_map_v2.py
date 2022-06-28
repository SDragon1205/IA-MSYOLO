from builtins import NotImplementedError
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
from core.yolov4 import filter_boxes,decode
from tensorflow.python.saved_model import tag_constants
import core.utils as utils
from tqdm import tqdm
from core.config import cfg
import json
from pycocotools.coco import COCO
from mAP.cocoeval import COCOeval
import sys


flags.DEFINE_string('weights', './data/yolov4.weights', 'path to weights file')
flags.DEFINE_string('annotation_path', "./data/dataset/val2017.txt", 'annotation path')
flags.DEFINE_string('class_path', './data/classes/1cls.names', 'class name path')

flags.DEFINE_string('framework', 'tf', 'select model type in (tf, tflite, tf_ckpt, trt)path to weights file')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('input_size', 608, 'resize images to')
flags.DEFINE_boolean('draw_image', False, 'write image path')
flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

def yolo2coco(label_path, pred_path, class_list):
    # id_dict = {}
    if label_path[-1] == '/':
        label_path = label_path[:-1]
    if pred_path[-1] == '/':
        pred_path = pred_path[:-1]
        
    if '.json' == label_path[-5:]:
        label_json = label_path
        print('Loading GT JSON file from '+label_json)
    else:
        files = os.listdir(label_path)

        labels = []
        images = []
        classes = [
            {"supercategory": "all", "id": cls_id, "name": class_name }
            for cls_id, class_name in enumerate(class_list)
        ]
        i = 0
        print('Ground Truth yolo txt to coco json...')
        for file in tqdm(files):
            if '.txt' not in file:
                continue
            # img = cv2.imread('imgs/'+file[:-4]+'.jpg')

            # W = img.shape[1]
            # H = img.shape[0]

            W = 3000
            H = 3000
            images.append({'file_name':file[:-4]+'.jpg', 'height': H, 'width': W, 'id':int(file[:-4])})

            f1 = open(label_path+'/'+file, 'r')
            for line in f1.readlines():
                c, x1, y1, x2, y2 = line.split()
                x1 = float(x1)
                y1 = float(y1)
                w = float(x2) - x1
                h = float(y2) - y1
                bb = [x1, y1, w, h]
                try:
                    category_idx = class_list.index(c)
                except:
                    print(f'Unknown class name in ground truth: "{c}"')
                    exit()
                labels.append({
                    'id':i, 
                    'image_id':int(file[:-4]), 
                    'category_id':category_idx, 
                    'bbox':[round(b, 3) for b in bb], 'area':bb[2]*bb[3], 'iscrowd':0})
                i += 1
            f1.close()

        label_json = os.path.join(os.path.dirname(label_path), label_path.split('/')[-1])+'.json'
        print('Saving JSON file to '+label_json)
        with open(label_json, 'w') as f:
            json.dump({'annotations':labels, 'images':images, 'categories':classes}, f)
    #########################################################################################################
    #########################################################################################################
    #########################################################################################################
    #########################################################################################################

    if '.json' == pred_path[-5:]:
        pred_json = pred_path
        print('Loading DT JSON file from '+pred_json)
    else:
        files = os.listdir(pred_path)

        predictions = []
        print('Predictions yolo txt to coco json...')
        for file in tqdm(files):
            if '.txt' not in file:
                continue

            f2 = open(pred_path+'/'+file, 'r')
            for line in f2.readlines():
                c, s, x1, y1, x2, y2 = line.split()
                # if float(s) < args.ct:
                #     continue
                x1 = float(x1)
                y1 = float(y1)
                w = float(x2) - x1
                h = float(y2) - y1
                bb = [x1, y1, w, h]
                try:
                    category_idx = class_list.index(c)
                except:
                    print(f'Unknown class name in prediction: "{c}"')
                    exit()
                predictions.append({
                    'image_id':int(file[:-4]), 
                    'category_id':category_idx, 
                    'bbox':[round(b, 3) for b in bb], 'score':round(float(s), 5)})
            f2.close()
        pred_json = os.path.join(os.path.dirname(pred_path), pred_path.split('/')[-1])+'.json'
        print('Saving JSON file to '+pred_json)
        with open(pred_json, 'w') as f:
            json.dump(predictions, f)
    return label_json, pred_json


def main(_argv):
    INPUT_SIZE = FLAGS.input_size
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    CLASSES = utils.read_class_names(FLAGS.class_path)
    print(f'CLASSES: {CLASSES}')

    predicted_dir_path = './mAP/predicted'
    ground_truth_dir_path = './mAP/ground-truth'
    should_pass = False
    if not should_pass:
        if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
        if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)

        os.mkdir(predicted_dir_path)
        os.mkdir(ground_truth_dir_path)
        os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)

    # Build Model According to different model weight format
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # elif FLAGS.framework == 'tf_ckpt':
    #     infer = tf.keras.models.load_model(FLAGS.weights)
    #     infer.summary()
    elif FLAGS.framework == 'tf':
        infer = tf.keras.models.load_model(FLAGS.weights)
    else:
        raise NotImplementedError(f"no such frame detected: {FLAGS.framework}")

    if not should_pass:
        with open(FLAGS.annotation_path, 'r') as annotation_file:
            lines = annotation_file.readlines()
            print(FLAGS.annotation_path, '  #########################################')
            with tqdm(total=len(lines), ncols=150, desc="Evaluating: ") as pbar:
                for num, line in enumerate(lines):
                    # Load Image From Annotation File
                    annotation = line.strip().split()
                    image_path = annotation[0]
                    image_name = image_path.split('/')[-1]
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Process Annotation in each line 
                    bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])
                    if len(bbox_data_gt) == 0:
                        bboxes_gt = []
                        classes_gt = []
                    else:
                        bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                    ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

                    num_bbox_gt = len(bboxes_gt)
                    # Write ground truth  Of each image and write them to another folder
                    with open(ground_truth_path, 'w') as f:
                        for i in range(num_bbox_gt):
                            class_name = CLASSES[classes_gt[i]]
                            xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                            bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                            f.write(bbox_mess)

                    predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
                    # Predict Process
                    image_size = image.shape[:2]
                    image_data = cv2.resize(np.copy(image), (INPUT_SIZE, INPUT_SIZE))
                    image_data = image_data / 255.
                    image_data = image_data[np.newaxis, ...].astype(np.float32)

                    # Inference Code for different format of model
                    if FLAGS.framework == 'tflite':
                        interpreter.set_tensor(input_details[0]['index'], image_data)
                        interpreter.invoke()
                        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                        if FLAGS.model == 'yolov4' and FLAGS.tiny == True:
                            boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape = tf.constant([INPUT_SIZE,INPUT_SIZE]))
                        else:
                            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape = tf.constant([INPUT_SIZE,INPUT_SIZE]))
                    # elif FLAGS.framework == 'tf_ckpt':
                    #     batch_data = tf.constant(image_data)
                    #     pred_bbox = infer(batch_data)
                    #     boxes = pred_bbox[0] / np.array([INPUT_SIZE,INPUT_SIZE,INPUT_SIZE,INPUT_SIZE], dtype='float32')
                    #     boxes = tf.concat([
                    #         (boxes[..., :2] - boxes[..., 2:] / 2.0)[...,::-1],
                    #         (boxes[..., :2] + boxes[..., 2:] / 2.0)[...,::-1],
                    #     ], axis=-1)
                    #     pred_conf = pred_bbox[1]
                    elif FLAGS.framework == 'tf':
                        batch_data = tf.constant(image_data)
                        pred_bbox = infer(batch_data)
                        boxes = pred_bbox[..., 0:4]
                        pred_conf = pred_bbox[..., 4:]
                    else:
                        raise NotImplementedError()

                    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                        max_output_size_per_class=20,
                        max_total_size=20,
                        iou_threshold=FLAGS.iou,
                        score_threshold=FLAGS.score
                    )

                    boxes, scores, classes, valid_detections = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()] 

                    if cfg.TEST.DECTECTED_IMAGE_PATH is not None and FLAGS.draw_image:
                        image_result = utils.draw_bbox(np.copy(image), [boxes, scores, classes, valid_detections])
                        cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH + image_name, image_result)

                    with open(predict_result_path, 'w') as f:
                        image_h, image_w, _ = image.shape
                        for i in range(valid_detections[0]):
                            if int(classes[0][i]) < 0 or int(classes[0][i]) > NUM_CLASS: continue
                            coor = boxes[0][i]
                            coor[0] = int(coor[0] * image_h)
                            coor[2] = int(coor[2] * image_h)
                            coor[1] = int(coor[1] * image_w)
                            coor[3] = int(coor[3] * image_w)

                            score = scores[0][i]
                            class_ind = int(classes[0][i])
                            class_name = CLASSES[class_ind]
                            score = '%.4f' % score
                            ymin, xmin, ymax, xmax = list(map(str, coor))
                            bbox_mess = f'{class_name:6s} {xmin:>4s} {ymin:>4s} {xmax:>4s} {ymax:>4s} {score}'
                            f.write(' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n')
                            
                    pbar.set_postfix({
                        'detected_obj': f'{valid_detections[0]:2d}',
                        'image_path': f"{image_path}",
                    })
                    pbar.update(1)


    l_json, p_json = yolo2coco(ground_truth_dir_path, predicted_dir_path, list(CLASSES.values()))
    cocoGt=COCO(l_json)
    cocoDt=cocoGt.loadRes(p_json)
    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    # cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 1022], [1022, 3809], [3809, 1e5 ** 2]]
    cocoEval.params.areaRng = [[123, 1e5 ** 2], [123, 341], [341, 1e5 ** 2],  [1e5 ** 2, 1e5 ** 2+1]]
    # cocoEval.params.areaRng = [[41, 1e5 ** 2], [41, 133], [133, 256],  [256, 1e5 ** 2]]
    cocoEval.params.areaRngLbl=['all','small', 'medium', 'large']
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    sys.stdout = open(os.path.join(os.path.dirname(FLAGS.weights), 'mAP.txt'), 'w')
    print(FLAGS.annotation_path)
    print(FLAGS.weights)
    print(cocoEval.params.areaRng)
    cocoEval.summarize()
    sys.stdout.close()



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


