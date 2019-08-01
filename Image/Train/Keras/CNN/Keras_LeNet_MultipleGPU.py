from skimage import io,transform
import glob
import os
import numpy as np
# import matplotlib.image as mpimg

import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

#參考 https://blog.csdn.net/wmy199216/article/details/71171401
#讀取圖片
def read_img(path,img_height,img_width,img_channl,writePath):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s'%(im))
            img=io.imread(im)
            # img = mpimg.imread(im)
            img_resize=transform.resize(img,(img_height,img_width))
            # --------------------------------------------------------------------
            if(img_channl > 1):
                if(img_resize.shape != (img_height, img_width, img_channl)):
                    print('img_resize.shape error:%s'%(im))
                else:
                    imgs.append(img_resize)
                    labels.append(idx)
            else:
                if(img_resize.shape != (img_height, img_width)):
                    print('img_resize.shape error:%s'%(im))
                else:
                    imgs.append(img_resize)
                    labels.append(idx)
            # --------------------------------------------------------------------
    writeLabels(path,writePath)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

#寫檔(classes)
def writeLabels(readPath,writePath):
    cate = [x for x in os.listdir(readPath) if os.path.isdir(readPath+x)]
    f = open(writePath, "w")
    for i in range(0,len(cate),1):
        f.write(str(cate[i]) + ",")
    f.close()

def ImageDataAugmentation(ImageData,Labels,batch_size,number):
    datagen = ImageDataGenerator(rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
                                   
    for i in range(0,number,1): 
        for batch in datagen.flow(ImageData,Labels, batch_size=5,
                        save_to_dir='./../Temp/', save_prefix='DataAug-', save_format='jpg'):
            x,y=batch
            ImageData = np.append(ImageData, x, axis=0)
            Labels = np.append(Labels, y, axis=0)
            break
            
    return ImageData,Labels

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
    
def buildLeNetModel(img_height,img_width,img_channl,num_classes,num_GPU):
    #建立模型,(LeNet架構)
    model = Sequential()

    model.add(Conv2D(32,(5,5),strides=(1,1),input_shape=(img_height, img_width,img_channl),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(5,5),strides=(1,1),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(num_classes,activation='softmax'))

    model.summary()    
    model = multi_gpu_model(model, gpus=num_GPU)
    model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
    return model

if __name__ == "__main__":
    #參數設定
    img_height, img_width, img_channl = 28, 28 , 1 #28,28,1
    num_classes = 10
    batch_size = 32
    epochs = 100
    dataSplitRatio=0.8
    readDataPath = "./../../../Data/"
    saveModelPath = "./../../../Model/Keras_LeNet"
    saveTensorBoardPath = "./../../../Model/TensorBoard"
    writeLabelPath = "./../../../Model/Keras_LeNet_Classes.txt"
    num_GPU = 2
    num_DataAug = 0

    #載入資料
    data,label = read_img(readDataPath,img_height,img_width,img_channl,writeLabelPath)
    
    # #資料增強
    # if(num_DataAug > 0):
    #     print(data.shape)
    #     data,label = ImageDataAugmentation(data,label,data.shape[0],num_DataAug)
    #     print(data.shape)

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
    
    #資料增強
    if(num_DataAug > 0):
        print(x_train.shape, 'train samples')
        print(x_val.shape, 'validation samples')
        x_train,y_train = ImageDataAugmentation(x_train,y_train,x_train.shape[0],num_DataAug)

    #重新調整大小
    if(img_channl == 1):
        x_train = x_train.reshape(x_train.shape[0],img_height, img_width, img_channl)
        x_val = x_val.reshape(x_val.shape[0],img_height, img_width, img_channl)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')

    #將數字轉為 One-hot 向量
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_val = np_utils.to_categorical(y_val, num_classes)
    
    model = buildLeNetModel(img_height,img_width,img_channl,num_classes,num_GPU)
    
    #訓練及保存模型
    saveTrainModels(model,saveModelPath,saveTensorBoardPath,epochs,batch_size,x_train,y_train,x_val,y_val)