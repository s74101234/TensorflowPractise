import re
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

#讀取文字
def readText(path, path2, max_fatures):
    Data = pd.read_csv(path , names = ['text'], skiprows = 1)
    Data2 = pd.read_csv(path2 , names = ['text', 'sentiment'], skiprows = 1)
    TotalData = pd.concat([Data2.drop(columns = ['sentiment']), Data], axis=0)

    # 將大寫轉成小寫
    Data['text'] = Data['text'].apply(lambda x: x.lower())
    TotalData['text'] = TotalData['text'].apply(lambda x: x.lower())
    # 過濾一些其他符號
    Data['text'] = Data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
    TotalData['text'] = TotalData['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
    # Tokenizer 實際上只是生成了一個字典，並且統計了詞頻等資訊，並沒有把文本轉成需要的向量表示。
    tokenizer = Tokenizer(num_words = max_fatures, split = ' ')
    # tokenizer.fit_on_texts(texts) <-- texts 是要訓練的text列表，此方法會轉成所需的文本字典。
    tokenizer.fit_on_texts(TotalData['text'].values)
    # texts_to_sequences <-- 我們將訓練好的Text轉成向量
    Data = tokenizer.texts_to_sequences(Data['text'].values)
    # pad_sequences <-- 將多個序列截斷或補齊為相同長度
    Data = pad_sequences(Data)
    
    return Data

if __name__ == "__main__":
    #參數設定
    readDataPath = './Data/test.csv'
    readDataPath2 = './Data/train.csv'
    loadModelPath = "./Model/Keras_LSTM_05_0.8704_0.8105.h5"
    writePath = './result_Keras_LSTM_05_0.8704_0.8105.csv'
    max_fatures = 2000

    #載入資料
    data = readText(readDataPath, readDataPath2, max_fatures)
    print('data shape:', data.shape)
    print(data.shape[0], 'test samples')

    #載入模型
    model = load_model(loadModelPath)

    fw = open(writePath, "w")
    fw.write('%s,%s\n'%('Id', 'Category'))
    result = model.predict(data)
    for i in range(0, result.shape[0], 1):
        print('%s,%s'%(str(i), str(np.argmax(result[i]))))
        fw.write('%s,%s\n'%(str(i), str(np.argmax(result[i]))))
    fw.close()