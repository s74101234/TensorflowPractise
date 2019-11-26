import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, AveragePooling2D, ZeroPadding2D, add
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

#參考 https://blog.csdn.net/wmy199216/article/details/71171401
def Conv2d_BN(x, nb_filter, kernel_size, strides = (1,1), padding = 'same'):
    x = Conv2D(nb_filter,kernel_size, padding = padding, strides = strides, activation = 'relu')(x)
    x = BatchNormalization(axis = 3)(x)
    return x
 
def Conv_Block_2(inputs, nb_filter, kernel_size, strides = (1, 1), with_conv_shortcut = False):
    x = Conv2d_BN(inputs, nb_filter = nb_filter, kernel_size = kernel_size, strides = strides, padding = 'same')
    x = Conv2d_BN(x, nb_filter = nb_filter, kernel_size = kernel_size, padding = 'same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inputs, nb_filter = nb_filter, strides = strides, kernel_size = kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inputs])
        return x
 
def Conv_Block_3(inputs, nb_filter, kernel_size, strides = (1, 1), with_conv_shortcut = False):
    x = Conv2d_BN(inputs, nb_filter = nb_filter[0], kernel_size = (1, 1), strides = strides, padding='same')
    x = Conv2d_BN(x, nb_filter = nb_filter[1], kernel_size = (3,3), padding = 'same')
    x = Conv2d_BN(x, nb_filter = nb_filter[2], kernel_size = (1,1), padding = 'same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inputs, nb_filter = nb_filter[2], strides = strides, kernel_size = kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inputs])
        return x

def buildResNetModel_34(img_height, img_width, img_channl, num_classes, num_GPU):
    inputs = Input(shape = (img_height, img_width, img_channl))
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2d_BN(x, nb_filter = 64, kernel_size = (7, 7), strides = (2, 2), padding = 'valid')
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = Conv_Block_2(x, nb_filter = 64, kernel_size = (3, 3))
    x = Conv_Block_2(x, nb_filter = 64, kernel_size = (3, 3))
    x = Conv_Block_2(x, nb_filter = 64, kernel_size = (3, 3))

    x = Conv_Block_2(x, nb_filter = 128, kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)
    x = Conv_Block_2(x, nb_filter = 128, kernel_size = (3, 3))
    x = Conv_Block_2(x, nb_filter = 128, kernel_size = (3, 3))
    x = Conv_Block_2(x, nb_filter = 128, kernel_size = (3, 3))

    x = Conv_Block_2(x, nb_filter = 256, kernel_size = (3,3), strides = (2, 2), with_conv_shortcut = True)
    x = Conv_Block_2(x, nb_filter = 256, kernel_size = (3, 3))
    x = Conv_Block_2(x, nb_filter = 256, kernel_size = (3, 3))
    x = Conv_Block_2(x, nb_filter = 256, kernel_size = (3, 3))
    x = Conv_Block_2(x, nb_filter = 256, kernel_size = (3, 3))
    x = Conv_Block_2(x, nb_filter = 256, kernel_size = (3, 3))

    x = Conv_Block_2(x, nb_filter = 512, kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)
    x = Conv_Block_2(x, nb_filter = 512, kernel_size = (3, 3))
    x = Conv_Block_2(x, nb_filter = 512, kernel_size = (3, 3))
    x = AveragePooling2D(pool_size = (7, 7))(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation = 'softmax')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    model.compile(loss = categorical_crossentropy,
            optimizer = Adam(lr = 0.001),
            metrics = ['accuracy'])
    return model

def buildResNetModel_50(img_height, img_width, img_channl, num_classes, num_GPU):
    inputs = Input(shape = (img_height, img_width, img_channl))
    
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2d_BN(x, nb_filter = 64, kernel_size = (7, 7), strides = (2, 2), padding = 'valid')
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = Conv_Block_3(x, nb_filter = [64, 64, 256], kernel_size = (3, 3), strides = (1, 1), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [64, 64, 256], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [64, 64, 256], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [512, 512, 2048], kernel_size = (3, 3), strides = (2,2), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [512, 512, 2048], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [512, 512, 2048], kernel_size = (3, 3))
    x = AveragePooling2D(pool_size = (7, 7))(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation = 'softmax')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    model.compile(loss = categorical_crossentropy,
            optimizer = Adam(lr = 0.001),
            metrics = ['accuracy'])
    return model

def buildResNetModel_101(img_height, img_width, img_channl, num_classes, num_GPU):
    inputs = Input(shape = (img_height, img_width, img_channl))
    
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2d_BN(x, nb_filter = 64, kernel_size = (7, 7), strides = (2, 2), padding = 'valid')
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = Conv_Block_3(x, nb_filter = [64, 64, 256], kernel_size = (3, 3), strides = (1, 1), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [64, 64, 256], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [64, 64, 256], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))

    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))

    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [512, 512, 2048], kernel_size = (3, 3), strides = (2,2), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [512, 512, 2048], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [512, 512, 2048], kernel_size = (3, 3))
    x = AveragePooling2D(pool_size = (7, 7))(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation = 'softmax')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    model.compile(loss = categorical_crossentropy,
            optimizer = Adam(lr = 0.001),
            metrics = ['accuracy'])
    return model

def buildResNetModel_152(img_height, img_width, img_channl, num_classes, num_GPU):
    inputs = Input(shape = (img_height, img_width, img_channl))
    
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2d_BN(x, nb_filter = 64, kernel_size = (7, 7), strides = (2, 2), padding = 'valid')
    x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same')(x)

    x = Conv_Block_3(x, nb_filter = [64, 64, 256], kernel_size = (3, 3), strides = (1, 1), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [64, 64, 256], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [64, 64, 256], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [128, 128, 512], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3), strides = (2, 2), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))

    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))

    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [256, 256, 1024], kernel_size = (3, 3))
    
    x = Conv_Block_3(x, nb_filter = [512, 512, 2048], kernel_size = (3, 3), strides = (2,2), with_conv_shortcut = True)
    x = Conv_Block_3(x, nb_filter = [512, 512, 2048], kernel_size = (3, 3))
    x = Conv_Block_3(x, nb_filter = [512, 512, 2048], kernel_size = (3, 3))
    x = AveragePooling2D(pool_size = (7, 7))(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation = 'softmax')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    model.compile(loss = categorical_crossentropy,
            optimizer = Adam(lr = 0.001),
            metrics = ['accuracy'])
    return model
