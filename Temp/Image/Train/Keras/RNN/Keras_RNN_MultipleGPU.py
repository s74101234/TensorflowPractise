from skimage import io,transform
import glob
import os
import numpy as np
import matplotlib.image as mpimg

import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import SimpleRNN
from keras.layers import Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard

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
    
def buildRNNModel(num_units,img_height,img_width,num_classes,num_GPU):
    #建立模型,(SimpleRNN架構)
    model = Sequential()
    
    model.add(SimpleRNN(
        batch_input_shape=(None,img_height, img_width), 
        units= num_units,
        unroll=True,
    )) 
    
    model.add(Dense(units=num_classes, kernel_initializer='normal', activation='softmax'))

    
    model.summary()
    
    model = multi_gpu_model(model, gpus=num_GPU)

    model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
    return model

if __name__ == "__main__":
    #參數設定
    #在RNN中是以向量作為運算，因此不會有多個維度，如RGB 3個維度。
    img_height, img_width = 28, 28 
    num_units = 100
    num_classes = 10
    batch_size = 20
    epochs = 10
    dataSplitRatio=0.8
    readDataPath = "./../../Data/"
    saveTensorBoardPath = "./../../../Model/TensorBoard"
    saveModelPath = "./../../Model/Keras_RNN"
    num_GPU = 2

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
    x_train = x_train.reshape(x_train.shape[0],img_height, img_width)
    x_val = x_val.reshape(x_val.shape[0],img_height, img_width)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')

    #將標籤轉為 One-hot 向量
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_val = np_utils.to_categorical(y_val, num_classes)
    
    model = buildRNNModel(num_units,img_height,img_width,num_classes,num_GPU)
    
    #訓練及保存模型
    saveTrainModels(model,saveModelPath,saveTensorBoardPath,epochs,batch_size,x_train,y_train,x_val,y_val)
    