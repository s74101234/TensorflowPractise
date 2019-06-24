from skimage import io,transform
import glob
import os
import numpy as np
import matplotlib.image as mpimg

import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils

#讀取圖片
def read_img(path,img_height,img_width):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s'%(im))
            img=io.imread(im)
#            img = mpimg.imread(im)
            img_resize=transform.resize(img,(img_height,img_width))
            imgs.append(img_resize)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

def saveTrainModels(model,saveModelPath,epochs,batch_size,x_train,y_train,x_test,y_test):
    #設置checkpoint
    checkpoint = ModelCheckpoint(
            monitor='val_acc', verbose=1, 
            save_best_only=True, mode='auto',
            filepath=('%s_{epoch:02d}_{val_acc:.2f}.h5' %(saveModelPath)))
    callbacks_list = [checkpoint]
    #訓練模型
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              shuffle = True,
              validation_data =(x_test, y_test),
              callbacks=callbacks_list)
    
def buildLeNetModel(img_channl,img_height,img_width,num_classes):
    #建立模型,(LeNet架構)
    model = Sequential()
    
    model.add(Convolution2D(20, (5, 5), activation='relu', 
                            input_shape = (img_height, img_width,img_channl)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Convolution2D(50, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(500 ,activation='relu'))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.summary()
    
    model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
    return model

if __name__ == "__main__":
    #參數設定
    img_height, img_width, img_channl = 28, 28 , 1
    num_classes = 10
    batch_size = 20
    epochs = 10
    dataSplitRatio=0.8
    readDataPath = "./../../../Data/"
    saveModelPath = "./../../../Model/Keras_LeNet"

    #載入資料
    data,label = read_img(readDataPath,img_height,img_width)
    
    #順序隨機
    num_example=data.shape[0]
    arr=np.arange(num_example)
    np.random.shuffle(arr)
    data=data[arr]
    label=label[arr]
    
    #切割資料
    #num_example=data.shape[0]
    s=np.int(num_example*dataSplitRatio)
    x_train=data[:s]
    y_train=label[:s]
    x_val=data[s:]
    y_val=label[s:]

    #重新調整大小
    x_train = x_train.reshape(x_train.shape[0],img_height, img_width, img_channl)
    x_val = x_val.reshape(x_val.shape[0],img_height, img_width, img_channl)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')

    #將數字轉為 One-hot 向量
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_val = np_utils.to_categorical(y_val, num_classes)
    
    model = buildLeNetModel(img_channl,img_height,img_width,num_classes)
    
    #訓練及保存模型
    saveTrainModels(model,saveModelPath,epochs,batch_size,x_train,y_train,x_val,y_val)
    