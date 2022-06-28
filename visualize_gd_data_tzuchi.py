'''
Author: pha123661 pha123661@gmail.com
Date: 2022-06-12 03:41:17
LastEditors: pha123661 pha123661@gmail.com
LastEditTime: 2022-06-16 13:41:42
FilePath: /iamsyolo/ia-yolov4-tflite/tsne.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
from PIL import Image
import sys
import numpy
# Open the image form working directory

# 設定輸入與輸出
inputLabelPath = 'datasets/night_dataset/anno/train_1cls.txt'
# 圖片大小固定為 608＊608
img_w= img_h = 608
# 載入圖片
j=0

jpg_n = 0


with open(inputLabelPath) as f:
  for line in f:
    jpg_n += 1
    if jpg_n!=1000:
      continue
    #print("line: ", line)
    line = line.replace(","," ") # 刪除多餘的空白
    data = line.split() # 將 YOLO 一列的內容轉成一維陣列
    #print("data: ", data)
    # 將 YOLO 格式轉換為邊界框的左上角和右下角座標
    #print("len: ", len(data))
    image = Image.open(data[0])
    cv2_img = cv2.imread(data[0])
    #draw_bbox = []
    leng = len(data)
    ran = int((leng - 1)/5)
    for i in range(ran): 
        #crop((left, top, right, bottom))
        left = int(data[5*i+1])
        right = int(data[5*i+3])
        top = int(data[5*i+2])
        bottom = int(data[5*i+4])
        print(left, top, right, bottom)
        box = cv2.rectangle(cv2_img, (left,top), (right, bottom), (0,0,255), 2)
        #draw_bbox.append(box)
        '''
        imc = image.crop((left, top, right, bottom))
        #print(left, top, right, bottom)
        pix = numpy.array(imc)
        print(pix.shape)
        # imc.save("tsne_test/test.png","png")
        '''
    cv2.imwrite("tsne_test/bbox_test_{}.jpg".format(j),cv2_img)
    j = j+1
    if j == 5: sys.exit()
  f.close()