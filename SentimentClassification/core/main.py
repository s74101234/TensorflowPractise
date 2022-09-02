import re
import pandas as pd
import numpy as np

import keras
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#讀取文字
def readText(path, path2, max_fatures):
    Data = pd.read_csv(path , names = ['text', 'sentiment'], skiprows = 1)
    Data2 = pd.read_csv(path2 , names = ['text'], skiprows = 1)
    TotalData = pd.concat([Data.drop(columns = ['sentiment']), Data2], axis=0)

    # one-hot encoding 
    Label = pd.get_dummies(Data['sentiment']).values

    # 將大寫轉成小寫
    Data['text'] = Data['text'].apply(lambda x: x.lower())
    TotalData['text'] = TotalData['text'].apply(lambda x: x.lower())

    # 過濾一些其他符號
    Data['text'] = Data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
    TotalData['text'] = TotalData['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

    # Tokenizer 實際上只是生成了一個字典，並且統計了詞頻等資訊，並沒有把文本轉成需要的向量表示。
    tokenizer = Tokenizer(num_words = max_fatures, split = ' ')
    # tokenizer.fit_on_texts(texts) <-- texts 是要訓練的text列表，此方法會轉成所需的文本字典。
    # tokenizer.fit_on_texts(Data['text'].values)
    tokenizer.fit_on_texts(TotalData['text'].values)
    # texts_to_sequences <-- 我們將訓練好的Text轉成向量
    Data = tokenizer.texts_to_sequences(Data['text'].values)
    # pad_sequences <-- 將多個序列截斷或補齊為相同長度
    Data = pad_sequences(Data)
    
    return Data, Label, tokenizer

def ConvertEmbeddingMatrix(readGlovePath, embed_dim, tokenizer):
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open(readGlovePath, encoding = "utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    # create a weight matrix for words in training docs
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embed_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix, vocab_size

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
    Reduce = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.9, patience = 10, cooldown = 1, verbose = 1)

    #設置EarlyStopping
    Early = EarlyStopping(monitor = 'val_accuracy', patience = 30, verbose = 1)

    callbacks_list = [checkpoint, tbCallBack, Reduce, Early]

    #訓練模型
    model.fit(x_train, y_train,
                batch_size = batch_size,
                nb_epoch = epochs,
                verbose = 1,
                shuffle = True,
                validation_data = (x_val, y_val),
                callbacks = callbacks_list)
