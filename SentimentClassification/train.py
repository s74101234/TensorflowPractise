import glob
import os
import numpy as np
import pandas as pd
import re

import keras
from core.main import readText, saveTrainModels, ConvertEmbeddingMatrix
from core.Model.NLPModel import buildRNNModel, buildLSTMModel, buildGRUModel
from core.Model.NLPAppModel import buildAppModel

if __name__ == "__main__":
    readDataPath = './Data/train.csv'
    readDataPath2 = './Data/test.csv'
    usingWeight = False
    readGlovePath = "./glove/glove.6B.300d.txt"
    embed_dim = 300
    saveModelPath = "./Model/Keras_LSTM"
    saveTensorBoardPath = "./Model/Tensorboard/"
    max_fatures = 2000
    num_classes = 3
    batch_size = 256
    epochs = 1000
    dataSplitRatio = 0.8
    num_GPU = 1

    # 讀取資料
    data, label, tokenizer = readText(readDataPath, readDataPath2, max_fatures)
    print('data shape:', data.shape)
    print('label shape:', label.shape)
    print(data.shape[0], 'train samples')

    #順序隨機
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]
    
    #切割資料
    s = np.int(num_example * dataSplitRatio)
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]
    
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_val shape:', x_val.shape)
    print('y_val shape:', y_val.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'val samples')

    # 讀取GloVe
    # embedding_matrix, vocab_size = ConvertEmbeddingMatrix(readGlovePath, embed_dim, tokenizer)

    # 建構模型
    model = buildAppModel(max_fatures, x_train.shape[1], embed_dim, num_classes, num_GPU, embedding_matrix=None, usingWeight=usingWeight)
    # model = buildLSTMModel(max_fatures, x_train.shape[1], embed_dim, num_classes, num_GPU, usingWeight=usingWeight)
    # model = buildLSTMModel(vocab_size, x_train.shape[1], embed_dim, num_classes, num_GPU, embedding_matrix=embedding_matrix, usingWeight=usingWeight)

    # 訓練並儲存模型
    saveTrainModels(model, saveModelPath, saveTensorBoardPath, epochs, batch_size, x_train, y_train, x_val, y_val)