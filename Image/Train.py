from skimage import io, transform
import glob
import os
import numpy as np

import keras
from keras.utils import np_utils
from core.main import readImage, ImageDataAugmentation, saveTrainModels
from core.model import buildLeNetModel
# from core.main import readImage, saveTrainModels_gen

if __name__ == "__main__":
    #參數設定
    img_height, img_width, img_channl = 28, 28, 1 #224, 224, 3
    num_classes = 10
    batch_size = 32
    epochs = 50
    dataSplitRatio = 0.8
    readDataPath = "./Data/Train/"
    saveModelPath = "./Model/Keras_LeNet"
    saveTensorBoardPath = "./Model/Tensorboard/"
    writeLabelPath = "./Model/Keras_LeNet_Classes.txt"
    num_GPU = 1
    num_DataAug = 0

    #載入資料
    data, label = readImage(readDataPath, img_height, img_width, img_channl, writeLabelPath)
    
    #順序隨機
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]
    
    #切割資料
    s = np.int(num_example * dataSplitRatio)
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]
    
    #資料增強
    if(num_DataAug > 0):
        print(x_train.shape, 'train samples')
        print(x_val.shape, 'validation samples')
        x_train,y_train = ImageDataAugmentation(x_train, y_train, x_train.shape[0], num_DataAug)

    #重新調整大小
    if(img_channl == 1):
        x_train = x_train.reshape(x_train.shape[0], img_height, img_width, img_channl)
        x_val = x_val.reshape(x_val.shape[0], img_height, img_width, img_channl)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')

    #將數字轉為 One-hot 向量
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_val = np_utils.to_categorical(y_val, num_classes)
    
    # 建構模型
    model = buildLeNetModel(img_height, img_width, img_channl, num_classes, num_GPU)
    
    #訓練及保存模型
    saveTrainModels(model, saveModelPath, saveTensorBoardPath, epochs, batch_size, x_train, y_train, x_val, y_val)
    # saveTrainModels_gen(model, saveModelPath, saveTensorBoardPath, epochs, batch_size, x_train, y_train, x_val, y_val)