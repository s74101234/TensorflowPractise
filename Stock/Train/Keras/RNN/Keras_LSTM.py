from skimage import io,transform
import glob
import os
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
from pandas import DataFrame
import calendar

import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.losses import categorical_crossentropy
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import multi_gpu_model

#參考 shorturl.at/joruY
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
def normalize(csvData,writePath):
    csvData = csvData.drop(["Date"], axis=1)

    csvDataMax = []
    csvDataMin = []
    result = csvData.copy()
    for feature_name in csvData.columns:
        max_value = csvData[feature_name].max()
        min_value = csvData[feature_name].min()
        csvDataMax.append(max_value)
        csvDataMin.append(min_value)
        if(max_value == min_value):
            result[feature_name] = 0.0
        else:
            result[feature_name] = (csvData[feature_name] - min_value) / (max_value - min_value)

    writeNormalize(writePath,csvDataMax,csvDataMin)
    return result

#寫檔(資料正規化,MAX,MIN)
def writeNormalize(writePath,csvDataMax,csvDataMin):
    f = open(writePath, "w")
    for i in range(0,len(csvDataMax),1):
        f.write(str(csvDataMax[i]) + ",")
    f.write("\n")
    for i in range(0,len(csvDataMin),1):
        f.write(str(csvDataMin[i]) + ",")    
    f.close()

#移除不必要的資料
def csvdataSelect(csvData,dropLabel):
    csvData = csvData.loc[:, ~csvData.columns.str.match('Unnamed')]
    for i in range(0,len(dropLabel),1):
        csvData = csvData.drop([dropLabel[i]], axis=1)
    return csvData

# Insert_row
def Insert_row(row_number, df, row_value): 
    start_upper = 0   
    end_upper = row_number    
    start_lower = row_number    
    end_lower = df.shape[0]    
    upper_half = [*range(start_upper, end_upper, 1)]    
    lower_half = [*range(start_lower, end_lower, 1)]    
    lower_half = [x.__add__(1) for x in lower_half] 
    index_ = upper_half + lower_half 
    df.index = index_ 
    df.loc[row_number] = row_value 
    df = df.sort_index() 
    return df

#修復缺失資料(Hour)
def RepairMissingData(csvData):
    # csvData = csvData.values.tolist()
    Count = len(csvData.values)
    i = 1
    while(i<Count):
        if(csvData.values[i-1][0] == csvData.values[i][0]):
            if((int(csvData.values[i-1][1]) + 1) != int(csvData.values[i][1])):
                # print(str((csvData.values[i][1])))
                csvData = Insert_row((i-1),csvData,csvData.values[i-1])
                csvData.loc[i,'Hour'] = int(csvData.values[i-1][1]) + 1
                csvData.sort_index(inplace=True)
                Count += 1
        else:
            if(int(csvData.values[i-1][1]) != 24):
                csvData = Insert_row((i-1),csvData,csvData.values[i-1])
                csvData.loc[i,'Hour'] = int(csvData.values[i-1][1]) + 1
                csvData.sort_index(inplace=True)
                Count += 1   
            elif(int(csvData.values[i][1]) != 1):
                csvData = Insert_row((i-1),csvData,csvData.values[i-1])
                csvData.loc[i,'Hour'] = 1
                csvData.loc[i,'Date'] = csvData.loc[i+1,'Date']
                csvData.loc[i,'date'] = csvData.loc[i+1,'date']
                csvData.sort_index(inplace=True)
                Count += 1  

        i += 1

    Count = len(csvData.values)
    i = 1
    while(i<Count):
        if(csvData.values[i-1][0] != csvData.values[i][0]):
            if(csvData.loc[i-1,'month'] == csvData.loc[i,'month']):
                MissingDayTemp = csvData.loc[i-1,'date'] - csvData.loc[i,'date']
                if(MissingDayTemp > 1):
                    dayTemp = csvData.loc[i-1,'date'] - 1
                    csvData["Date"].dt.day
                    for j in range(1,25,1):
                        csvData = Insert_row((i-1),csvData,csvData.values[i-1])
                        csvData.loc[i,'Hour'] = 25-j
                        csvData.loc[i,'date'] = dayTemp
                        DateTemp  = pd.to_datetime(csvData.loc[i,'Date'])
                        DateTemp = DateTemp.replace(day=dayTemp)
                        csvData.loc[i,'Date'] = DateTemp
                        csvData.sort_index(inplace=True)
                        Count += 1
            else:
                totalDayTemp = calendar.monthrange(csvData.loc[i,'year'],csvData.loc[i,'month'])[1]
                if(csvData.loc[i,'date'] != totalDayTemp):
                    MissingDayTemp = totalDayTemp - csvData.loc[i,'date']
                    dayTemp = csvData.loc[i,'date'] + 1
                    monthTemp = csvData.loc[i,'month']
                    for j in range(1,25,1):
                        csvData = Insert_row((i-1),csvData,csvData.values[i-1])
                        csvData.loc[i,'Hour'] = 25-j
                        csvData.loc[i,'month'] = monthTemp
                        csvData.loc[i,'date'] = dayTemp
                        DateTemp  = pd.to_datetime(csvData.loc[i,'Date'])
                        DateTemp = DateTemp.replace(month=monthTemp,day=dayTemp)
                        csvData.loc[i,'Date'] = DateTemp
                        csvData.sort_index(inplace=True)
                        Count += 1
                    i-=1
        i += 1

    # csvData = DataFrame(csvData)
    return csvData


#修復缺失資料(Date)
def RepairMissingData2(csvData):
    csvData = csvData.reset_index(drop=True)
    Count = len(csvData.values)
    i = 1
    while(i<Count):
        if(csvData.loc[i-1,'month'] == csvData.loc[i,'month']):
            MissingDayTemp = csvData.loc[i-1,'date'] - csvData.loc[i,'date']
            if(MissingDayTemp > 1):
                dayTemp = csvData.loc[i-1,'date'] - 1
                csvData = Insert_row((i-1),csvData,csvData.values[i-1])
                csvData.loc[i,'date'] = dayTemp
                DateTemp  = pd.to_datetime(csvData.loc[i,'Date'])
                DateTemp = DateTemp.replace(day=dayTemp)
                csvData.loc[i,'Date'] = DateTemp
                csvData.sort_index(inplace=True)
                Count += 1
        else:
            totalDayTemp = calendar.monthrange(csvData.loc[i,'year'],csvData.loc[i,'month'])[1]
            if(csvData.loc[i,'date'] != totalDayTemp):
                MissingDayTemp = totalDayTemp - csvData.loc[i,'date']
                dayTemp = csvData.loc[i,'date'] + 1
                monthTemp = csvData.loc[i,'month']
                csvData = Insert_row((i-1),csvData,csvData.values[i-1])
                csvData.loc[i,'month'] = monthTemp
                csvData.loc[i,'date'] = dayTemp
                DateTemp  = pd.to_datetime(csvData.loc[i,'Date'])
                DateTemp = DateTemp.replace(month=monthTemp,day=dayTemp)
                csvData.loc[i,'Date'] = DateTemp
                csvData.sort_index(inplace=True)
                Count += 1
                i-=1
        i += 1

    # csvData = DataFrame(csvData)
    return csvData


#資料切割train與label - 過去與未來日期
def csvDataSplit(csvData, pastDay, futureDay , predictValue):
    data, label = [], []
    for i in range(csvData.shape[0]-futureDay-pastDay):
        data.append(np.array(csvData.iloc[i:i+pastDay]))
        label.append(np.array(csvData.iloc[i+pastDay:i+pastDay+futureDay][predictValue]))
    return np.array(data), np.array(label)

def saveTrainModels(model,saveModelPath,epochs,batch_size,x_train,y_train,x_test,y_test):
    #設置checkpoint
    checkpoint = ModelCheckpoint(
            monitor='val_loss', verbose=0, 
            save_best_only=True, mode='auto',
            filepath=('%s_{epoch:02d}_{val_loss:.4f}.h5' %(saveModelPath)))
    callbacks_list = [checkpoint]

    #訓練模型
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              shuffle = True,
              validation_data =(x_test, y_test),
              callbacks=callbacks_list)
    
def buildLSTMModel(num_units,height,width,num_classes):
    #建立模型,(LSTM架構)
    model = Sequential()

    model.add(LSTM(
        batch_input_shape=(None,height, width), 
        units= num_units,
        unroll=True,
        activation='relu'
    ))     
    # model.add(Dropout(0.2))
    model.add(Dense(units=num_classes)) 

    # mse mean_squared_error
    model.compile(loss="mse",
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
    
    model.summary()    
    return model

if __name__ == "__main__":
    # 參數設定
    predictValue = "Close"
    pastDay,futureDay = 60,1
    num_units = 200
    batch_size = 64
    epochs = 100
    dataSplitRatio=0.8
    readDataPath = "./../../../Data/Data.csv"
    saveModelPath = "./../../../Model/Keras_LSTM_"+predictValue
    writeNormalizePath = "./../../../Model/Keras_LSTM_Normalize.txt"

    # 載入資料
    csvData = readCSV(readDataPath)

    # Clean Missing Data 移除有缺失的資料
    csvData = CleanMissingData(csvData)

    #資料增強
    csvData = augFeatures(csvData)

    #修復缺失資料
    csvData = RepairMissingData2(csvData)

    #日期移除,資料正規化
    csvData = normalize(csvData,writeNormalizePath)

    #移除不必要資料
    csvData = csvdataSelect(csvData,["MarketCap","Volume"])
    print(csvData)

    #資料切割train與label - 過去與未來日期
    data, label = csvDataSplit(csvData, pastDay, futureDay , predictValue)

    #順序隨機
    num_example=data.shape[0]
    arr=np.arange(num_example)
    np.random.shuffle(arr)
    data=data[arr]
    label=label[arr]
    
    #切割資料
    s=np.int(num_example*dataSplitRatio)
    x_train=data[:s]
    y_train=label[:s]
    x_val=data[s:]
    y_val=label[s:]

    print(x_train)
    print('x_train shape:', x_train.shape)
    print('x_val shape:', x_val.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')

    model = buildLSTMModel(num_units,x_train.shape[1],x_train.shape[2],futureDay)
    
    #訓練及保存模型
    saveTrainModels(model,saveModelPath,epochs,batch_size,x_train,y_train,x_val,y_val)
    