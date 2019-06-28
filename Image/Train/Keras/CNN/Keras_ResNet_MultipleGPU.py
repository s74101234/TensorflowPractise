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
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from keras import backend as kerasBnd
from keras.callbacks import TensorBoard

#參考 keras ResNet : https://keras.io/examples/cifar10_resnet/
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

def saveTrainModels(model,saveModelPath,saveTensorBoardPath,epochs,batch_size,
                    x_train,y_train,x_test,y_test):
    #TensorBoard
    tbCallBack = TensorBoard(log_dir=saveTensorBoardPath,batch_size=batch_size,
                 write_graph=True,write_grads=True,write_images=True,
                 embeddings_freq=0,embeddings_layer_names=None,embeddings_metadata=None)

    #設置checkpoint
    checkpoint = ModelCheckpoint(
            monitor='val_acc', verbose=1, 
            save_best_only=True, mode='auto',
            filepath=('%s_{epoch:02d}_{val_acc:.2f}.h5' %(saveModelPath)))
    callbacks_list = [checkpoint,tbCallBack]
    
    #訓練模型
    model.fit(x_train, y_train,
              batch_size=batch_size,
              nb_epoch=epochs,
              verbose=1,
              shuffle = True,
              validation_data =(x_test, y_test),
              callbacks=callbacks_list)
    
def resnet_layer(inputs,num_filters=16,kernel_size=3,strides=1,
                 activation='relu',batch_normalization=True,conv_first=True):
    
    conv = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,
                  padding='same',kernel_initializer='he_normal')

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def buildResNetModel_v2(img_channl,img_height,img_width,
                        depth,num_classes,num_GPU):

    inputs = Input(shape = (img_height, img_width,img_channl))
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    x = resnet_layer(inputs=inputs,num_filters=num_filters_in,conv_first=True)
          
    #resNetBlock 3層
    for stage in range(3):
        #深度迴圈
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            #判斷是否是第一階段或第一層
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:
                    strides = 2
            #resNetBlock 進行
            y = resnet_layer(inputs=x,num_filters=num_filters_in,kernel_size=1,
                             strides=strides,activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)

            y = resnet_layer(inputs=y,num_filters=num_filters_in,conv_first=False)
            y = resnet_layer(inputs=y,num_filters=num_filters_out,kernel_size=1,conv_first=False)
            
            #合併resNetBlock(shortcut)
            if res_block == 0:
                x = resnet_layer(inputs=x,num_filters=num_filters_out, kernel_size=1,
                                 strides=strides,activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # final layer
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = AveragePooling2D(pool_size=8)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    model = Model(inputs=inputs, outputs=outputs)
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
    saveTensorBoardPath = "./../../../Model/TensorBoard"
    writeLabelPath = "./../../../Model/Keras_ResNet_Classes.txt"
    num_GPU = 2
    resNetVersion = 2
    depthNum = 3

    if(resNetVersion == 1):
        depthNum = depthNum * 6 + 2
    elif(resNetVersion == 2):
        depthNum = depthNum * 9 + 2

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
    
    model = buildResNetModel_v2(img_channl,img_height,img_width,
                                depthNum,num_classes,num_GPU)
    
    #訓練及保存模型
    saveTrainModels(model,saveModelPath,saveTensorBoardPath,epochs,batch_size,x_train,y_train,x_val,y_val)
    