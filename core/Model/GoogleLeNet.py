import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, AveragePooling2D, ZeroPadding2D, concatenate, Dropout
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model

#參考 https://blog.csdn.net/wmy199216/article/details/71171401
def Conv2d_BN(x, nb_filter, kernel_size, padding = 'same', strides = (1, 1)):
    x = Conv2D(nb_filter, kernel_size, padding = padding, strides = strides, activation = 'relu')(x)
    x = BatchNormalization(axis = 3)(x)
    return x

def Inception(x, nb_filter):
    branch1x1 = Conv2d_BN(x, nb_filter, (1, 1), padding = 'same', strides = (1, 1))
 
    branch3x3 = Conv2d_BN(x, nb_filter, (1, 1), padding = 'same', strides = (1, 1))
    branch3x3 = Conv2d_BN(branch3x3, nb_filter, (3, 3), padding = 'same', strides = (1, 1))
 
    branch5x5 = Conv2d_BN(x, nb_filter, (1, 1), padding = 'same', strides = (1, 1))
    branch5x5 = Conv2d_BN(branch5x5, nb_filter, (1, 1), padding = 'same', strides = (1, 1))
 
    branchpool = MaxPooling2D(pool_size = (3, 3), strides = (1, 1), padding = 'same')(x)
    branchpool = Conv2d_BN(branchpool, nb_filter, (1, 1), padding = 'same', strides = (1, 1))
 
    x = concatenate([branch1x1, branch3x3, branch5x5, branchpool], axis = 3)
    return x

def buildGoogleLeNetModel(img_height, img_width, img_channl, num_classes, num_GPU):
    inputs = Input(shape = (img_height, img_width, img_channl))

    x = Conv2d_BN(inputs, 64, (7, 7), strides = (2, 2), padding = 'same')
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    x = Conv2d_BN(x, 192, (3, 3), strides = (1, 1), padding = 'same')
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    x = Inception(x ,64)
    x = Inception(x ,120)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    x = Inception(x, 128)
    x = Inception(x, 128)
    x = Inception(x, 128)
    x = Inception(x, 132)
    x = Inception(x, 208)
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)
    x = Inception(x, 208)
    x = Inception(x, 256)
    x = AveragePooling2D(pool_size = (7, 7), strides = (7, 7), padding = 'same')(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(1000, activation = 'relu')(x)
    outputs = Dense(num_classes, activation = 'softmax')(x)

    model = Model(inputs = inputs, outputs = outputs)

    model.summary()    
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    model.compile(loss = categorical_crossentropy,
              optimizer = Adam(lr = 0.001),
              metrics = ['accuracy'])
    return model
