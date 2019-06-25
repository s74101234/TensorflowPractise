from skimage import io,transform
import glob
import os
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import sys
import re

import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.losses import categorical_crossentropy
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from keras.models import load_model

#讀取CSV
def readCSV(path):
    f = open(path)
    readText = pd.read_csv(f)
    f.close()
    print("readCSV finish.")
    return readText

#Clean Missing Data 移除有缺失的資料
def CleanMissingData(csvData):
    dropindex=[]
    for i in range(0,len(csvData.values),1):
        for j in range(0,len(csvData.values[0]),1):
            if(csvData.values[i][j] == "-"):
                dropindex.append(i)
                break
    csvData = csvData.drop(dropindex)
    print("CleanMissingData finish.")
    return csvData

#資料增強-日期切割
def augFeatures(csvData):
    csvData["Date"] = pd.to_datetime(csvData["Date"])
    csvData["year"] = csvData["Date"].dt.year
    csvData["month"] = csvData["Date"].dt.month
    csvData["date"] = csvData["Date"].dt.day
    csvData["week"] = csvData["Date"].dt.dayofweek
    print("augFeatures finish.")
    return csvData

#日期移除,資料正規化
def normalize(csvData,readNormalizePath):
    csvData = csvData.drop(["Date"], axis=1)
    result = csvData.copy()
    csvDataMax = []
    csvDataMin = []
    #讀取正規化最大與最小
    readMax,readMin = readNormalize(readNormalizePath)
    # print(len(csvData.columns))
    index = 0
    for feature_name in csvData.columns:
        max_value = csvData[feature_name].max()
        if(max_value < float(readMax[index])):
            max_value = float(readMax[index])
        min_value = csvData[feature_name].min()
        if(min_value < float(readMin[index])):
            min_value = float(readMin[index])
        csvDataMax.append(max_value)
        csvDataMin.append(min_value)
        if(max_value == min_value):
            result[feature_name] = 0.0
        else:
            result[feature_name] = (csvData[feature_name] - min_value) / (max_value - min_value)
        index += 1
    # print(csvDataMax)
    # print(csvDataMin)
    return result,csvDataMax,csvDataMin

#讀檔(資料正規化,MAX,MIN)
def readNormalize(readPath):
    f = open(readPath, "r")
    readText = f.read()
    result = re.split('\n',readText)
    csvDataMax = re.split(',',result[0])
    csvDataMin = re.split(',',result[1])
    f.close()
    return csvDataMax,csvDataMin

#還原正規化結果
def revertNormalize(PredictValue,csvDataMax,csvDataMin,index):
    for i in range(0,len(PredictValue),1):
        PredictValue[i] = PredictValue[i] * (csvDataMax[index] - csvDataMin[index]) + csvDataMin[index]
    
    return PredictValue

#移除不必要的資料
def csvdataSelect(csvData,dropLabel):
    csvData = csvData.loc[:, ~csvData.columns.str.match('Unnamed')]
    for i in range(0,len(dropLabel),1):
        csvData = csvData.drop([dropLabel[i]], axis=1)
    return csvData

#資料切割train與label - 過去與未來日期
def csvDataSplit(csvData, pastDay, futureDay):
    data = []
    for i in range(csvData.shape[0]-pastDay+1):
        data.append(np.array(csvData.iloc[i:i+pastDay]))
    return np.array(data)

if __name__ == "__main__":
    #參數設定
    predictValue = "Close"
    pastDay,futureDay = 60,1
    readDataPath = "./../../../Input/Data.csv"
    loadModelPath = "./../../../Model/Keras_LSTM_"+predictValue+".h5"
    readNormalizePath = "./../../../Model/Keras_LSTM_Normalize.txt"

    #載入資料
    csvData = readCSV(readDataPath)

    #Clean Missing Data 移除有缺失的資料
    csvData = CleanMissingData(csvData)

    #資料增強
    csvData = augFeatures(csvData)
    # print(csvData)

    #日期移除,資料正規化
    csvData,csvDataMax,csvDataMin = normalize(csvData,readNormalizePath)

    #移除不必要資料
    csvData = csvdataSelect(csvData,["MarketCap","Volume"])
    print(csvData)

    #資料切割train與label - 過去與未來日期
    data = csvDataSplit(csvData, pastDay, futureDay)

    #讀取模型
    model = load_model(loadModelPath)

    #Predict
    result = model.predict(data)
    print(str(result))

    #還原正規化
    index = csvData.columns.get_loc(predictValue)
    result = revertNormalize(result,csvDataMax,csvDataMin,index)
    print(str(result))