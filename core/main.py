from skimage import io,transform
import glob
import os
import numpy as np

import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

#讀取圖片
def readImage(path, img_height, img_width, img_channl, writeClassNamePath = None):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s'%(im))
            img = io.imread(im)
            img_resize = transform.resize(img, (img_height, img_width))
            # --------------------------------------------------------------------
            if(img_channl > 1):
                if(img_resize.shape != (img_height, img_width, img_channl)):
                    print('img_resize.shape error:%s'%(im))
                else:
                    img_resize = img_resize / 255
                    imgs.append(img_resize)
                    labels.append(idx)
            else:
                if(img_resize.shape != (img_height, img_width)):
                    print('img_resize.shape error:%s'%(im))
                else:
                    img_resize = img_resize / 255
                    imgs.append(img_resize)
                    labels.append(idx)
            # --------------------------------------------------------------------
    if(writeClassNamePath != None):
        writeLabels(path, writeClassNamePath)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

#寫檔(classes)
def writeLabels(readPath, writePath):
    cate = [x for x in os.listdir(readPath) if os.path.isdir(readPath + x)]
    fw = open(writePath, "w")
    for i in range(0, len(cate), 1):
        fw.write(str(cate[i]) + ",")
    fw.close()

# ImageDataAugmentation
def ImageDataAugmentation(ImageData, Labels, batch_size, number = 1, save_to_dir = None, 
                            rotation_range = 20, width_shift_range = 0.1, height_shift_range = 0.1, 
                            shear_range = 0.1, zoom_range = 0.1, horizontal_flip = True, fill_mode = 'nearest'):
    datagen = ImageDataGenerator(rotation_range = rotation_range,
                                   width_shift_range = width_shift_range,
                                   height_shift_range = height_shift_range,
                                   shear_range = shear_range,
                                   zoom_range = zoom_range,
                                   horizontal_flip = horizontal_flip,
                                   fill_mode = fill_mode)
                                   
    for i in range(0, number, 1): 
        for batch in datagen.flow(ImageData, Labels, batch_size = batch_size,
                        save_to_dir = save_to_dir, save_prefix = 'DataAug-', save_format = 'jpg'):
            x, y = batch
            ImageData = np.append(ImageData, x, axis=0)
            Labels = np.append(Labels, y, axis=0)
            break
            
    return ImageData, Labels

def saveTrainModels_gen(model, saveModelPath, saveTensorBoardPath, epochs, batch_size,
                    x_train, y_train, x_val, y_val, 
                    rotation_range = 20, width_shift_range = 0.1, height_shift_range = 0.1, 
                    shear_range = 0.1, zoom_range = 0.1, horizontal_flip = True, fill_mode = 'nearest'):
	# DataGen
    datagen = ImageDataGenerator(rotation_range = rotation_range,
                                width_shift_range = width_shift_range,
                                height_shift_range = height_shift_range,
                                shear_range = shear_range,
                                zoom_range = zoom_range,
                                horizontal_flip = horizontal_flip,
                                fill_mode = fill_mode)

    #設置TensorBoard
    tbCallBack = TensorBoard(log_dir = saveTensorBoardPath, batch_size = batch_size,
                            write_graph = True, write_grads = True, write_images = True,
                            embeddings_freq = 0, embeddings_layer_names = None, embeddings_metadata = None)

    #設置checkpoint
    checkpoint = ModelCheckpoint(
                            monitor = 'val_accuracy', verbose = 1, 
                            save_best_only = True, mode = 'max',
                            filepath = ('%s_{epoch:02d}_{accuracy:.4f}_{val_accuracy:.4f}.h5' %(saveModelPath)))

    #設置ReduceLROnPlateau
    Reduce = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.9, patience = 5, cooldown = 1, verbose = 1)

    #設置EarlyStopping
    Early = EarlyStopping(monitor = 'val_accuracy', patience = 25, verbose = 1)

    callbacks_list = [checkpoint, tbCallBack, Reduce, Early]

    #訓練模型
    model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size),
                steps_per_epoch = len(x_train),
                epochs = epochs,
                verbose = 1,
                shuffle = True,
                validation_data = (x_val, y_val),
                callbacks = callbacks_list)
    
def saveTrainModels(model, saveModelPath, saveTensorBoardPath, epochs, batch_size,
                    x_train, y_train, x_val, y_val):
    #設置TensorBoard
    tbCallBack = TensorBoard(log_dir = saveTensorBoardPath, batch_size = batch_size,
                            write_graph = True, write_grads = True, write_images = True,
                            embeddings_freq = 0, embeddings_layer_names = None, embeddings_metadata = None)

    #設置checkpoint
    checkpoint = ModelCheckpoint(
                            monitor = 'val_accuracy', verbose = 1, 
                            save_best_only = True, mode = 'max',
                            filepath = ('%s_{epoch:02d}_{accuracy:.4f}_{val_accuracy:.4f}.h5' %(saveModelPath)))

    #設置ReduceLROnPlateau
    Reduce = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.9, patience = 5, cooldown = 1, verbose = 1)

    #設置EarlyStopping
    Early = EarlyStopping(monitor = 'val_accuracy', patience = 25, verbose = 1)

    callbacks_list = [checkpoint, tbCallBack, Reduce, Early]

    #訓練模型
    model.fit(x_train, y_train,
                batch_size = batch_size,
                nb_epoch = epochs,
                verbose = 1,
                shuffle = True,
                validation_data = (x_val, y_val),
                callbacks = callbacks_list)
