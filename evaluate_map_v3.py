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
from filters_lowlight import *

flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_string('weights2', './checkpoints/cnnpp-416', 'path to weights file')
flags.DEFINE_string('annotation_path', "./datasets/night_dataset/anno/val_1cls.txt", 'annotation path')
flags.DEFINE_string('class_path', './data/classes/1cls.names', 'class name path')

flags.DEFINE_string('framework', 'tf', 'select model type in (tf, tflite, tf_ckpt, trt)path to weights file')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('input_size', 608, 'resize images to')
flags.DEFINE_boolean('draw_image', True, 'write image path')
flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

def iou(bboxes1, bboxes2):
    """
    bboxes1=(n1,4)
    bboxes2=(n2,4)
    """
    left_up =    np.maximum(bboxes1[..., np.newaxis, :2], bboxes2[..., :2])             # (n1, n2, 2)
    right_down = np.minimum(bboxes1[..., np.newaxis, 2:], bboxes2[..., 2:])             # (n1, n2, 2)

    bboxes_area1 = np.prod(bboxes1[..., 2:] - bboxes1[..., :2], axis=-1)                # (n1,)
    bboxes_area2 = np.prod(bboxes2[..., 2:] - bboxes2[..., :2], axis=-1)                # (n2,)

    inter_section = np.maximum(right_down - left_up, 0.0)                               # (n1, n2, 2)
    inter_area = inter_section[..., 0] * inter_section[..., 1]                          # (n1, n2)
    union_area = bboxes_area1.reshape((-1,1)) + bboxes_area2.reshape((1,-1)) - inter_area   # (n1,1)+(1,n2)-(n1,n2)=(n1,n2)
    iou = inter_area /  union_area
    iou = np.nan_to_num(iou)
    return iou

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
    error_image_dir = './mAP/error'
    correct_image_dir = './mAP/correct'
    shutil.rmtree(error_image_dir, ignore_errors=True)
    shutil.rmtree(correct_image_dir, ignore_errors=True)
    os.makedirs(error_image_dir, exist_ok=True)
    os.makedirs(correct_image_dir, exist_ok=True)

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
    elif FLAGS.framework == 'tf':
        infer = tf.keras.models.load_model(FLAGS.weights)
    else:
        raise NotImplementedError(f"no such frame detected: {FLAGS.framework}")
    CNN_PP = tf.keras.models.load_model(FLAGS.weights2)
    dip = DIP(cfg)

    if not should_pass:
        with open(FLAGS.annotation_path, 'r') as annotation_file:
            lines = annotation_file.readlines()
            print(FLAGS.annotation_path, '  #########################################')
            total_img = 0
            wrong_img = 0

            total_gt_instance = 0
            total_dt_instance = 0
            correct_detected_instance = 0
            mis_detected_instance = 0

            with tqdm(total=len(lines), ncols=200, desc="Evaluating: ") as pbar:
                for num, line in enumerate(lines):
                    # Load Image From Annotation File
                    annotation = line.strip().split()
                    image_path = annotation[0]
                    image_name = '_'.join(image_path.split('/')[-2:])
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    #image = np.expand_dims(image, axis = 0)

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


                    if cfg.isp_flag:
                        # with tf.compat.v1.variable_scope('extract_parameters_2'):
                        input_data = tf.image.resize(image_data, [256, 256], method=tf.image.ResizeMethod.BILINEAR)
                        filter_features = CNN_PP(input_data, training = False)#CNN-PP module
                        #print('filter_features: ', type(filter_features), filter_features.shape)#cfg.num_filter_parameters = 14
                        #sys.exit()
                        image_data = dip(image_data, filter_features)
                        # print('image: ', type(image), image.shape)#cfg.num_filter_parameters = 14
                        # print('image_data: ', type(image_data), image_data.shape)
                        # image_size = image_data[:2]
                        # image = image_data


                    # Inference Code for different format of model
                    if FLAGS.framework == 'tflite':
                        interpreter.set_tensor(input_details[0]['index'], image_data)
                        interpreter.invoke()
                        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                        if FLAGS.model == 'yolov4' and FLAGS.tiny == True:
                            boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape = tf.constant([INPUT_SIZE,INPUT_SIZE]))
                        else:
                            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape = tf.constant([INPUT_SIZE,INPUT_SIZE]))
                    elif FLAGS.framework == 'tf':
                        batch_data = tf.constant(image_data)
                        pred_bbox = infer(batch_data)
                        boxes = pred_bbox[..., 0:4]             # ymin,xmin,ymax,xmax
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
                    boxes = boxes[:,:valid_detections[0]]
                    scores = scores[:,:valid_detections[0]]
                    classes = classes[:,:valid_detections[0]]

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

                    any_error = False
                    
                    boxes_x1y1x2y2 = np.stack([boxes[...,1],boxes[...,0],boxes[...,3],boxes[...,2]], axis=-1)
                    if len(classes_gt) > 0:
                        for cls_id in range(NUM_CLASS):
                            gt_class_bbox = bboxes_gt[classes_gt==cls_id]
                            dt_class_bbox = boxes_x1y1x2y2[0][classes[0]==cls_id]

                            if cls_id not in classes_gt: continue
                            if cls_id not in classes:
                                total_gt_instance += len(gt_class_bbox)
                                total_dt_instance += 0
                                correct_detected_instance += 0
                                mis_detected_instance += 0
                                continue

                            iou_matrix = iou(gt_class_bbox, dt_class_bbox)

                            sub_match = 0
                            matched_gt_instance = []
                            matched_dt_instance = []

                            # check the recall rate
                            for gt_idx in range(len(gt_class_bbox)):
                                dt_idx = np.argmax(iou_matrix[gt_idx])
                                if iou_matrix[gt_idx][dt_idx] < 0.3 or gt_idx in matched_gt_instance:
                                    any_error = True
                                else:
                                    sub_match += 1
                                    matched_gt_instance.append(gt_idx)
                                    matched_dt_instance.append(dt_idx)
                            if len(np.unique(matched_dt_instance)) != len(matched_dt_instance):
                                print(f'[Error] {image_path} have smae dt result match different gt')
                                print(f'\t class: {cls_id}')
                                print(f'\t matched_gt_instance: {matched_gt_instance}')
                                print(f'\t matched_dt_instance: {matched_dt_instance}')

                            
                            # check the accuracy 
                            if len(gt_class_bbox) != len(dt_class_bbox):
                                any_error = True
                            # counting instance
                            total_gt_instance += len(gt_class_bbox)
                            total_dt_instance += len(dt_class_bbox)
                            correct_detected_instance += sub_match
                            mis_detected_instance += (len(gt_class_bbox) - sub_match)                    
                    else:
                        if len(boxes) > 0:
                            any_error = True
                    total_img += 1
                    if any_error: wrong_img += 1

                    if FLAGS.draw_image:
                        h,w = image_size
                        img_size_ratio = np.array([[h,w,h,w]])
                        bboxes_gt_y1x1y2x2 = np.stack([bboxes_gt[...,1],bboxes_gt[...,0],bboxes_gt[...,3],bboxes_gt[...,2]], axis=-1)
                        #print('image: ', type(image), image.shape)
                        image = np.clip((image_data * 255.0), a_min=0, a_max=255).astype(np.uint8)
                        image = np.squeeze(image, axis=0)
                        #print('image2: ', type(image), image.shape)
                        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
                        #print('image3: ', type(image), image.shape)
                        #sys.exit()
                        image_result = utils.draw_bbox(np.copy(image), \
                            [boxes/img_size_ratio,                 scores,     classes,        valid_detections])
                        image_result = utils.draw_bbox(np.copy(image_result), \
                            [[bboxes_gt_y1x1y2x2/img_size_ratio],  None,       [classes_gt],   [len(bboxes_gt)]], is_gt=True)
                        if any_error:
                            cv2.imwrite(os.path.join(error_image_dir, image_name), image_result[...,::-1])
                        else:
                            cv2.imwrite(os.path.join(correct_image_dir, image_name), image_result[...,::-1])

                    pbar.set_postfix({
                        'image_path': f"{image_name}",
                        "correct_img":  f'{(total_img-wrong_img)/total_img:5.2f}({total_img-wrong_img}/{total_img})',
                        "recall_instance": f'{correct_detected_instance/total_gt_instance:5.2f}({correct_detected_instance}/{total_gt_instance})',
                        "accuracy_instance":   f'{correct_detected_instance/total_dt_instance:5.2f}({correct_detected_instance}/{total_dt_instance})',
                    })
                    pbar.update(1)


    l_json, p_json = yolo2coco(ground_truth_dir_path, predicted_dir_path, list(CLASSES.values()))
    cocoGt=COCO(l_json)
    cocoDt=cocoGt.loadRes(p_json)
    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    # cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 1022], [1022, 3809], [3809, 1e5 ** 2]]
    # cocoEval.params.areaRng = [[123, 1e5 ** 2], [123, 341], [341, 1e5 ** 2],  [1e5 ** 2, 1e5 ** 2+1]]
    cocoEval.params.areaRng = [[123, 1e5 ** 2], [123, 341], [341, 768],  [768, 1e5 ** 2]]
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


