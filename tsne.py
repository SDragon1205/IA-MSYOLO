# '''
# Author: pha123661 pha123661@gmail.com
# Date: 2022-06-12 03:41:17
# LastEditors: pha123661 pha123661@gmail.com
# LastEditTime: 2022-06-12 06:04:14
# FilePath: /iamsyolo/ia-yolov4-tflite/tsne.py
# Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
# '''
# import cv2
# from PIL import Image
# import sys
# import numpy as np
# # Open the image form working directory

# # 設定輸入與輸出
# inputLabelPath = 'datasets/data_selection_mix/anno/train_1cls.txt'
# # 圖片大小固定為 608＊608
# img_w= img_h = 608
# # 載入圖片


# with open(inputLabelPath) as f:
#   for line in f:
#     #print("line: ", line)
#     line = line.replace(","," ") # 刪除多餘的空白
#     data = line.split() # 將 YOLO 一列的內容轉成一維陣列
#     #print("data: ", data)
#     # 將 YOLO 格式轉換為邊界框的左上角和右下角座標
#     #print("len: ", len(data))
#     image = Image.open(data[0])
#     leng = len(data)
#     ran = int((leng - 1)/5)
#     for i in range(ran): 
#         #crop((left, top, right, bottom))
#         left = int(data[5*i+1])
#         right = int(data[5*i+3])
#         top = int(data[5*i+2])
#         bottom = int(data[5*i+4])
#         imc = image.crop((left, top, right, bottom))
#         #print(left, top, right, bottom)
#         pix = np.array(imc)
#         print(pix.shape)
#         flat_image = np.reshape(pix, [-1,3])
#         print(flat_image.shape)
#         # imc.save("tsne_test/test.png","png")
#         sys.exit()

        
#   f.close()
# import os
# import random
# import numpy as np
# import json
# import matplotlib.pyplot
# import pickle
# from matplotlib.pyplot import imshow
# from PIL import Image
# from sklearn.manifold import TSNE
# import sys

# images, pca_features, pca = pickle.load(open('features_caltech101.p', 'rb'))

# for img, f in list(zip(images, pca_features))[0:5]:
#     print("image: %s, features: %0.2f,%0.2f,%0.2f,%0.2f... "%(img, f[0], f[1], f[2], f[3]))
#     print("img: ", img)
#     print("f: ", f[0])
#     sys.exit()


# num_images_to_plot = 1000

# if len(images) > num_images_to_plot:
#     sort_order = sorted(random.sample(range(len(images)), num_images_to_plot))
#     images = [images[i] for i in sort_order]
#     pca_features = [pca_features[i] for i in sort_order]
# print("images: ", images.shape)
# print("pca_features: ", pca_features.shape)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
#Prepare the data
digits = datasets.load_digits(n_class=6)
X, y = digits.data, digits.target
n_samples, n_features = X.shape
n = 20  
img = np.zeros((10 * n, 10 * n))
for i in range(n):
    ix = 10 * i + 1
    for j in range(n):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))
plt.figure(figsize=(8, 8))
plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.show()
# xx = np.array(X)
# print("xx shape: ", type(xx))
# print("xx shape: ", xx.shape())
print("x shape: ", len(X))
print("x shape: ", len(X[0]))
print("x shape: ", len(X[0][0]))
print("x shape: ", len(X[0][0][0]))

#t-SNE
X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)

#Data Visualization
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()