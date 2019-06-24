from skimage import io,transform
import glob
import os
import numpy as np
# import matplotlib.image as mpimg

import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from keras import backend as kerasBnd

#讀取圖片
def read_img(path,img_height,img_width,writePath):
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
    writeLabels(path,writePath)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

#寫檔(classes)
def writeLabels(readPath,writePath):
    cate = [x for x in os.listdir(readPath) if os.path.isdir(readPath+x)]
    f = open(writePath, "w")
    for i in range(0,len(cate),1):
        f.write(str(cate[i]) + ",")
    f.close()

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
              nb_epoch=epochs,
              verbose=1,
              shuffle = True,
              validation_data =(x_test, y_test),
              callbacks=callbacks_list)
    
def buildResNetModel(img_channl,img_height,img_width,num_classes,num_GPU):
    #建立模型,(ResNet架構)
    model = Sequential()

    inputs = Input(shape=(img_height, img_width,img_channl))

    x = Convolution2D(64, (7, 7), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # residual
    residual = Convolution2D(64, (3, 3))(x)
    residual = Convolution2D(64, (3, 3))(residual)
    
    x = shortcut(x,residual)

    x = Flatten()(x)
    x = Dense(500 ,activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    
    model = multi_gpu_model(model, gpus=num_GPU)

    model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
    return model

def shortcut(input, residual):
    input_shape = kerasBnd.int_shape(input)
    residual_shape = kerasBnd.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input
    # 判斷兩個tensor是否相同大小，如果不相同則進行Conv2D之後才相加。
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(residual_shape[3],kernel_size=(1, 1),strides=(stride_width, stride_height),padding="valid")(input)

    return add([shortcut, residual])

if __name__ == "__main__":
    #參數設定
    img_height, img_width, img_channl = 28, 28 , 1
    num_classes = 10
    batch_size = 20
    epochs = 10
    dataSplitRatio=0.8
    readDataPath = "./../../../Data/"
    saveModelPath = "./../../../Model/Keras_ResNet"
    writeLabelPath = "./../../../Model/Keras_ResNet_Classes.txt"
    num_GPU = 2

    #載入資料
    data,label = read_img(readDataPath,img_height,img_width,writeLabelPath)
    
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
    
    model = buildResNetModel(img_channl,img_height,img_width,num_classes,num_GPU)
    
    #訓練及保存模型
    saveTrainModels(model,saveModelPath,epochs,batch_size,x_train,y_train,x_val,y_val)
    