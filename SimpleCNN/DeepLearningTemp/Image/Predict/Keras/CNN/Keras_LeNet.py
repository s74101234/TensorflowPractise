from skimage import io,transform
import numpy as np
import re

import keras
from keras.models import load_model

#讀取圖片
def read_img(readPath,img_height,img_width,readClasses):
    imgs=[]
    labels=[]

    # image
    img=io.imread(readPath)
    img_resize=transform.resize(img,(img_height,img_width))
    imgs.append(img_resize)

    # label
    f = open(readClasses, "r")
    readText = f.read()
    labels = re.split(',|\n',readText)
    f.close()

    return np.asarray(imgs,np.float32),labels

if __name__ == "__main__":
    #參數設定
    img_height, img_width, img_channl = 28, 28 , 1
    readDataPath = "./../../../Input/input.jpg"
    readClassPath = "./../../../Model/Keras_LeNet_Classes.txt"
    loadModelPath = "./../../../Model/Keras_LeNet.h5"

    #載入資料
    data,label = read_img(readDataPath,img_height,img_width,readClassPath)

    #載入模型
    model = load_model(loadModelPath)

    #重新調整大小
    data = data.reshape(1,img_height, img_width, img_channl)

    #predict
    result = model.predict(data)
    print(label[np.argmax(result[0])])
    