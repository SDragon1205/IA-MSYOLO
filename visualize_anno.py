import argparse
from ast import Tuple, parse
from tqdm import tqdm
import numpy as np
import os
import cv2
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--anno', default=None, type=str)
args = parser.parse_args()

if __name__ == '__main__':
    shutil.rmtree('visualize_anno')
    os.makedirs('visualize_anno', exist_ok=True)
    os.makedirs('visualize_anno/image_w_box', exist_ok=True)
    os.makedirs('visualize_anno/image_wo_box', exist_ok=True)
    with open(args.anno, 'r') as f:
        lines = f.readlines()
    
    for idx, line in tqdm(enumerate(lines), total=len(lines)):
        line_split = line.split()
        bboxes = np.array([list(map(int, box_str.split(','))) for box_str in line_split[1:]])
        image_path = line_split[0]
        video_idx  = line_split[0].split('/')[-2]
        image_name = os.path.basename(image_path)

        image_ori = cv2.imread(image_path)
        image = image_ori.copy()
        for box in bboxes:
            x1y1 = tuple(box[:2])
            x2y2 = tuple(box[2:4])
            cls_str = f'class_{box[-1]}'
            image = cv2.rectangle(image, x1y1, x2y2, (0, 0, 255), 2)
            image = cv2.putText(image, cls_str, x1y1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join('visualize_anno','image_w_box', f'{idx}_{video_idx}_'+image_name), image)
        cv2.imwrite(os.path.join('visualize_anno','image_wo_box', f'{idx}_{video_idx}_'+image_name), image_ori)

