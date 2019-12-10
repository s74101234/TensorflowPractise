import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense, BatchNormalization, Dropout
# 299x299,top-1 0.790, top-5 0.945, Parameters 22,910,480
from tensorflow.keras.applications.xception import Xception
# 224x224,top-1 0.713, top-5 0.901, Parameters 138,357,544	
from tensorflow.keras.applications.vgg16 import VGG16 
# 224x224,top-1 0.713, top-5 0.900, Parameters 143,667,240
from tensorflow.keras.applications.vgg19 import VGG19 
# 224x224,top-1 0.749, top-5 0.921, Parameters 25,636,712
# 224x224,top-1 0.764, top-5 0.928, Parameters 44,707,176
# 224x224,top-1 0.766, top-5 0.931, Parameters 60,419,944
from tensorflow.keras.applications.resnet import ResNet50, ResNet101, ResNet152
# 224x224,top-1 0.760, top-5 0.930, Parameters 25,613,800
# 224x224,top-1 0.772, top-5 0.938, Parameters 44,675,560
# 224x224,top-1 0.780, top-5 0.942, Parameters 60,380,648
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
# 299x299,top-1 0.779, top-5 0.937, Parameters 23,851,784
from tensorflow.keras.applications.inception_v3 import InceptionV3
# 299x299,top-1 0.803, top-5 0.953, Parameters 55,873,736
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
# 224x224,top-1 0.704, top-5 0.895, Parameters 4,253,864
from tensorflow.keras.applications.mobilenet import MobileNet
# 224x224,top-1 0.713, top-5 0.901, Parameters 3,538,984
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# 224x224,top-1 0.750, top-5 0.923, Parameters 8,062,504
# 224x224,top-1 0.762, top-5 0.932, Parameters 14,307,880
# 224x224,top-1 0.773, top-5 0.936, Parameters 20,242,984
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
# 331x331,top-1 0.744, top-5 0.919, Parameters 5,326,716
# 224x224,top-1 0.825, top-5 0.960, Parameters 88,949,818
from tensorflow.keras.applications.nasnet import NASNetLarge, NASNetMobile

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model

def buildKerasAppModel(img_height, img_width, img_channl, num_classes, num_GPU):
    FreezeNum = 0
    # inputs = Input(shape = (img_height, img_width, img_channl))
    # AppModel = MobileNetV2(include_top=False, pooling='avg', weights='imagenet', input_shape=inputs)
    AppModel = MobileNetV2(include_top = False, pooling = 'avg', weights = 'imagenet')

    # Freeze
    for idx, layer in enumerate(AppModel.layers):
        layer.trainable = False
        if(idx == StopNum):
            break

    x = AppModel.layers[-1].output
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation = 'softmax', name = 'softmax')(x)
    model = Model(inputs = AppModel.input, outputs = output_layer)

    model.summary()
    if(num_GPU > 1):
        model = multi_gpu_model(model, gpus = num_GPU)
    #categorical_crossentropy , sparse_categorical_crossentropy
    model.compile(loss = 'categorical_crossentropy',
            optimizer = Adam(lr = 1e-5),
            metrics = ['accuracy'])
    return model