import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, AveragePooling2D, ZeroPadding2D, add
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensroflow.keras.utils import multi_gpu_model

#參考 https://keras.io/examples/cifar10_resnet/
def resnet_layer(inputs, num_filters = 16, kernel_size = 3, strides = 1,
                 activation = 'relu', batch_normalization = True, conv_first = True):
    
    conv = Conv2D(num_filters, kernel_size = kernel_size, strides = strides,
                  padding = 'same', kernel_initializer = 'he_normal')

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

def shortcut(input, residual):
    input_shape = kerasBnd.int_shape(input)
    residual_shape = kerasBnd.int_shape(residual)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    shortcut = input
    # 判斷兩個tensor是否相同大小，如果不相同則進行Conv2D之後才相加。
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(residual_shape[3], kernel_size = (1, 1), strides = (stride_width, stride_height), padding = "valid")(input)

    return add([shortcut, residual])

def buildResNetV1Model(img_height, img_width, img_channl,
                        num_classes, num_GPU, depth = 16):
    inputs = Input(shape = (img_height, img_width, img_channl))
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    x = resnet_layer(inputs = inputs, num_filters = num_filters, conv_first = True)
    
    #resNetBlock 2層
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            #判斷是否是第一階段或第一層
            if stack > 0 and res_block == 0:  
                strides = 2  
            y = resnet_layer(inputs = x,
                             num_filters = num_filters,
                             strides = strides)
            y = resnet_layer(inputs = y,
                             num_filters = num_filters,
                             activation = None)
            #判斷是否是第一階段或第一層
            if stack > 0 and res_block == 0: 
                x = resnet_layer(inputs = x,
                                 num_filters = num_filters,
                                 kernel_size = 1,
                                 strides = strides,
                                 activation = None,
                                 batch_normalization = False)
            #合併resNetBlock(shortcut)
            x = add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    x = AveragePooling2D(pool_size = 8)(x)
    # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation = 'softmax',
                    kernel_initializer = 'he_normal')(y)
    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    model.compile(loss = categorical_crossentropy,
            optimizer = Adam(lr = 0.001),
            metrics = ['accuracy'])
    return model

def buildResNetModel_v2(img_height, img_width, img_channl,
                        num_classes, num_GPU, depth = 16):

    inputs = Input(shape = (img_height, img_width, img_channl))
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    x = resnet_layer(inputs = inputs, num_filters = num_filters_in, conv_first = True)
          
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
            y = resnet_layer(inputs = x, num_filters = num_filters_in, kernel_size = 1,
                             strides = strides, activation = activation,
                             batch_normalization = batch_normalization,
                             conv_first = False)

            y = resnet_layer(inputs = y, num_filters = num_filters_in, conv_first = False)
            y = resnet_layer(inputs = y, num_filters = num_filters_out, kernel_size = 1, conv_first = False)
            
            #合併resNetBlock(shortcut)
            if res_block == 0:
                x = resnet_layer(inputs = x, num_filters = num_filters_out, kernel_size = 1,
                                 strides = strides, activation = None,
                                 batch_normalization = False)
            x = add([x, y])

        num_filters_in = num_filters_out

    # final layer
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = AveragePooling2D(pool_size=8)(x)
    x = MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation = 'softmax',
                    kernel_initializer = 'he_normal')(y)

    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    
    model = multi_gpu_model(model, gpus = num_GPU)

    model.compile(loss = categorical_crossentropy,
              optimizer = Adam(lr = 0.001),#0.001
              metrics = ['accuracy'])
    return model
