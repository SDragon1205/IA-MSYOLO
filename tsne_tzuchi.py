'''
Author: pha123661 pha123661@gmail.com
Date: 2022-06-25 01:53:23
LastEditors: pha123661 pha123661@gmail.com
LastEditTime: 2022-06-28 15:09:28
FilePath: /iamsyolo/ia-yolov4-tflite/tsne_tzuchi.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from absl import app
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
#from mlxtend.plotting import category_scatter

import sys
import os
import shutil

def get_feature_file_data(tsne_data_Path):
    # 列出指定路徑底下所有檔案(包含資料夾，這裡只有檔案)
    allFileList = os.listdir(tsne_data_Path)
    # 檔案名稱按照數字順序排列
    allFileList.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))
    # 逐一查詢檔案清單
    feature_masked = np.empty(shape=([0,256]), dtype=float)
    for file in allFileList:
    #   使用isfile判斷是否為檔案
        if os.path.isfile(tsne_data_Path + file) and file.split('.')[-1]=='npy':
            #print(file)
            feature_masked = np.concatenate((feature_masked, np.load(tsne_data_Path + file)), axis=0)
        else:
            print('OH MY GOD !!')
            sys.exit()
    print('feature_masked shape: ',feature_masked.shape)
    # 把 NAN 的資料清除
    feature_masked = feature_masked[~np.isnan(feature_masked).any(axis=1)]
    #print('No NAN feature_masked shape: ',feature_masked.shape)
    #print('feature_masked_source', feature_masked[0][:10])
    
    # 把np.empty清除掉
    feature_masked_del = np.delete(feature_masked, 0, 0)
    #print('No NAN feature_masked_del: ',feature_masked_del.shape)
    #print('feature_masked_del', feature_masked_del[0][:10])
    return feature_masked_del

def main(_argv):
    
    '''
    # 移動檔案到目錄
    src = '/mnt/HDD1/iamsyolo/ia-yolov4-tflite/tsne_images/big_array_'
    dst1 = '/mnt/HDD1/iamsyolo/ia-yolov4-tflite/tsne_images/test1/big_array_'
    dst2 = '/mnt/HDD1/iamsyolo/ia-yolov4-tflite/tsne_images/test2/big_array_'

    for i in range(200):
        shutil.move(src+f'{i}'+'.npy',dst1+f'{i}'+'.npy')
    for i in range(200, 350):
        shutil.move(src+f'{i}'+'.npy',dst2+f'{i}'+'.npy')
    '''
    
    # 分別取得 source 與 target 之 mask 過的 feature
    feature_masked_source = get_feature_file_data('tsne_images/adversarial/source/')
    feature_masked_target = get_feature_file_data('tsne_images/adversarial/target/')

    # 紀錄各自的shape
    fms_shape = feature_masked_source.shape[0]
    fmt_shape = feature_masked_target.shape[0]

    print('fms_shape', feature_masked_source.shape)
    print('fmt_shape', feature_masked_target.shape)

    # 合併回去feature
    feature_masked = np.concatenate((feature_masked_source, feature_masked_target), axis=0)
    print('\nFinal feature_masked shape: ',feature_masked.shape)
    #print(feature_masked)
    # t-SNE
    X_tsne = manifold.TSNE(n_components=2,  verbose=1).fit_transform(feature_masked)
    #init='random', random_state=12,

    # Normalize
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  
    print(X_norm.shape)
    '''
    plt.figure(figsize=(8, 8))
    # plot source with red scatter
    for i in range(fms_shape):
        plt.scatter(X_norm[i][0], X_norm[i][1],c="red")
    # plot source with blue scatter
    for i in range(fms_shape, X_norm.shape[0]):
        plt.scatter(X_norm[i][0], X_norm[i][1],c="blue")
    plt.xticks([])
    plt.yticks([])
    '''

    point_x = np.array([X_norm[i][0] for i in range(X_norm.shape[0])])
    print('point_x shape: ', point_x.shape)
    point_y = np.array([X_norm[i][1] for i in range(X_norm.shape[0])])
    print('point_y shape: ', point_y.shape)
    label_source = np.array(['Source' for i in range(fms_shape)])
    print('label_source shape: ', label_source.shape)
    label_target = np.array(['Target' for i in range(fmt_shape)])
    print('label_target shape: ', label_target.shape)
    labels = np.concatenate((label_source,label_target), axis=0)

    df = pd.DataFrame(dict(x=point_x, y=point_y, label=labels))

    #grouped = df.groupby('label')

    #fig, ax = plt.subplots()

    colors = {'Source':'red', 'Target':'blue'}

    #fig = category_scatter(x='x', y='y', label_col='label', 
    #                   data=df, legend_loc='upper left')

    plt.scatter(df['x'], df['y'], s=1, c=df['label'].map(colors))  
    
    # displaying the title 
    #plt.title("Baseline network (No DAN) in TSNE analysis") 
    plt.title("Baseline network + DAN in TSNE analysis") 

    plt.savefig('tSNE_ad.png')
    


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


