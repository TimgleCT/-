import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import get_data
from datetime import datetime
import time


def getTarget(df,postKey):
    postKeyList = [postKey]
    getTarget = df.loc[np.in1d(df['post_key'], postKeyList)]
    time = getTarget.loc[:,"created_at_hour"].values
    count = getTarget.loc[:,"count"].values

    return time,count


def createFeature(count,time,createTime,row,data,featureNum,indexErrorCount,postKey):
    if time.shape[0] != 0:
            timestap = ((time - createTime[0]).astype('timedelta64[h]')).astype('int16')
            for rowNum in range(timestap.shape[0]):
                index = timestap[rowNum]
                npToDate = datetime.fromtimestamp((time[rowNum]-np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
                postTime = npToDate.hour
                if postTime == 1:
                    postTime ==25
                elif postTime == 2:
                    postTime ==26
                postTime = postTime/26
                try:
                    data[row][featureNum][index] = count[rowNum] * postTime
                except IndexError:
                    indexErrorCount += 1
                    print(postKey)
                    print("錯誤index："+str(index))
                    print("出現錯誤次數："+str(indexErrorCount))
                    continue
            for plus in range(1,11):
                data[row][featureNum][plus] += data[row][featureNum][plus - 1]

def preprocessingNPForTest(host):
    print("進來資料預處理")
    test, getPSTS, getPCCTS, getPLTS, getPCTS = get_data.getTestFromServer(host)
    print(test)

    testPostKeyDict = postKeyDict(test)
    getPSTS['post_key'] =  changePostKeyToIndex(getPSTS,testPostKeyDict)
    getPCCTS['post_key'] =  changePostKeyToIndex(getPCCTS,testPostKeyDict)
    getPLTS['post_key'] =  changePostKeyToIndex(getPLTS,testPostKeyDict)
    getPCTS['post_key'] =  changePostKeyToIndex(getPCTS,testPostKeyDict)


    scaler = MinMaxScaler()
    getPSTS['count'] = scaler.fit_transform(getPSTS[['count']])
    getPCCTS['count'] = scaler.fit_transform(getPCCTS[['count']])
    getPLTS['count'] = scaler.fit_transform(getPLTS[['count']])
    getPCTS['count'] = scaler.fit_transform(getPCTS[['count']])

    test['created_at_hour'] = pd.to_datetime(test['created_at_hour'], format = "%Y-%m-%d %H")
    getPSTS['created_at_hour'] = pd.to_datetime(getPSTS['created_at_hour'], format = "%Y-%m-%d %H")
    getPCCTS['created_at_hour'] = pd.to_datetime(getPCCTS['created_at_hour'], format = "%Y-%m-%d %H")
    getPLTS['created_at_hour'] = pd.to_datetime(getPLTS['created_at_hour'], format = "%Y-%m-%d %H")
    getPCTS['created_at_hour'] = pd.to_datetime(getPCTS['created_at_hour'], format = "%Y-%m-%d %H")

    testCount = test.shape[0]
    print(testCount)
    testData = np.zeros((testCount,4,11))

    indexErrorCount = 0

    for testRow in range(testCount):
        postKey = testRow
        createTime = test.loc[[testRow],"created_at_hour"].values

        PSTTime,PST = getTarget(getPSTS,postKey)
        PCCTTime,PCCT = getTarget(getPCCTS,postKey)
        PLTTime,PLT = getTarget(getPLTS,postKey)
        PCTTime,PCT = getTarget(getPCTS,postKey)
        
        createFeature(PST,PSTTime,createTime,testRow,testData,0,indexErrorCount,postKey)
        createFeature(PCCT,PCCTTime,createTime,testRow,testData,1,indexErrorCount,postKey)
        createFeature(PLT,PLTTime,createTime,testRow,testData,2,indexErrorCount,postKey)
        createFeature(PCT,PCTTime,createTime,testRow,testData,3,indexErrorCount,postKey)
                           
        print("Test：已完成"+str(testRow)+"筆資料建構\r",end = "")

    return testData,test['post_key']

    

def postKeyDict(df):
    dataDict = df.to_dict()
    reverseDict = {v: k for k, v in dataDict["post_key"].items()}
    return reverseDict

def changePostKeyToIndex(df,reverseDict):
    df["post_key"] = df["post_key"].map(reverseDict)
    return df["post_key"]



def preprocessingNP(host):
    if os.path.isfile("train.csv"):
        train = pd.read_csv('train.csv')
        getPST = pd.read_csv('PST.csv')
        getPCCT = pd.read_csv('PCCT.csv')
        getPLT = pd.read_csv('PLT.csv')
        getPCT = pd.read_csv('PCT.csv')
    else:
        train, getPST, getPCCT, getPLT,getPCT = get_data.createTrainCSV(host)


    if os.path.isfile("test.csv"):
        test = pd.read_csv('test.csv')
        getPSTS = pd.read_csv('PSTS.csv')
        getPCCTS = pd.read_csv('PCCTS.csv')
        getPLTS = pd.read_csv('PLTS.csv')
        getPCTS = pd.read_csv('PCTS.csv')
    else:
        test, getPSTS, getPCCTS, getPLTS, getPCTS = get_data.createTestCSV(host)

    trainPostKeyDict = postKeyDict(train)
    testPostKeyDict = postKeyDict(test)

    getPST['post_key'] =  changePostKeyToIndex(getPST,trainPostKeyDict)
    getPCCT['post_key'] =  changePostKeyToIndex(getPCCT,trainPostKeyDict)
    getPLT['post_key'] =  changePostKeyToIndex(getPLT,trainPostKeyDict)
    getPCT['post_key'] =  changePostKeyToIndex(getPCT,trainPostKeyDict)

    getPSTS['post_key'] =  changePostKeyToIndex(getPSTS,testPostKeyDict)
    getPCCTS['post_key'] =  changePostKeyToIndex(getPCCTS,testPostKeyDict)
    getPLTS['post_key'] =  changePostKeyToIndex(getPLTS,testPostKeyDict)
    getPCTS['post_key'] =  changePostKeyToIndex(getPCTS,testPostKeyDict)

    scaler = MinMaxScaler()
    getPST['count'] = scaler.fit_transform(getPST[['count']])
    getPCCT['count'] = scaler.fit_transform(getPCCT[['count']])
    getPLT['count'] = scaler.fit_transform(getPLT[['count']])
    getPCT['count'] = scaler.fit_transform(getPCT[['count']])

    getPSTS['count'] = scaler.fit_transform(getPSTS[['count']])
    getPCCTS['count'] = scaler.fit_transform(getPCCTS[['count']])
    getPLTS['count'] = scaler.fit_transform(getPLTS[['count']])
    getPCTS['count'] = scaler.fit_transform(getPCTS[['count']])

    train['created_at_hour'] = pd.to_datetime(train['created_at_hour'], format = "%Y-%m-%d %H")
    test['created_at_hour'] = pd.to_datetime(test['created_at_hour'], format = "%Y-%m-%d %H")

    getPST['created_at_hour'] = pd.to_datetime(getPST['created_at_hour'], format = "%Y-%m-%d %H")
    getPCCT['created_at_hour'] = pd.to_datetime(getPCCT['created_at_hour'], format = "%Y-%m-%d %H")
    getPLT['created_at_hour'] = pd.to_datetime(getPLT['created_at_hour'], format = "%Y-%m-%d %H")
    getPCT['created_at_hour'] = pd.to_datetime(getPCT['created_at_hour'], format = "%Y-%m-%d %H")

    getPSTS['created_at_hour'] = pd.to_datetime(getPSTS['created_at_hour'], format = "%Y-%m-%d %H")
    getPCCTS['created_at_hour'] = pd.to_datetime(getPCCTS['created_at_hour'], format = "%Y-%m-%d %H")
    getPLTS['created_at_hour'] = pd.to_datetime(getPLTS['created_at_hour'], format = "%Y-%m-%d %H")
    getPCTS['created_at_hour'] = pd.to_datetime(getPCTS['created_at_hour'], format = "%Y-%m-%d %H")


    trainCount = train.shape[0]
    testCount = test.shape[0]

    trainData = np.zeros((trainCount,3,11))
    trainLabel = np.zeros(trainCount)
    testData = np.zeros((testCount,3,11))
    testLabel = np.zeros(testCount)

    trainLabel = (train['like_count_36_hour']>=1000)
    testLabel = (test['like_count_36_hour']>=1000)


    print(trainLabel)
    print(testLabel)

    indexErrorCount = 0
    errorKey = []
    startLoading = int(round(time.time() * 1000))

    for trainRow in range(trainCount):
        postKey = trainRow
        createTime = train.loc[[trainRow],"created_at_hour"].values

        PSTTime,PST = getTarget(getPST,postKey)
        PCCTTime,PCCT = getTarget(getPCCT,postKey)
        PLTTime,PLT = getTarget(getPLT,postKey)
        PCTTime,PCT = getTarget(getPCT,postKey)

        createFeature(PST,PSTTime,createTime,trainRow,trainData,0,indexErrorCount,postKey)
        createFeature(PCCT,PCCTTime,createTime,trainRow,trainData,1,indexErrorCount,postKey)
        createFeature(PLT,PLTTime,createTime,trainRow,trainData,2,indexErrorCount,postKey)
        createFeature(PCT,PCTTime,createTime,trainRow,trainData,3,indexErrorCount,postKey)

        # print(trainData[trainRow])
        print("Train：已完成"+str(trainRow)+"筆資料建構\r",end = "")

    finishLoading = int(round(time.time() * 1000))
    loadingTime = (finishLoading - startLoading)/1000
    print("訓練資料建構完畢!共花費了"+str(int(loadingTime / 60))+"分"+str(loadingTime % 60)+"秒")


    startLoading = int(round(time.time() * 1000))
    for testRow in range(testCount):
        postKey = testRow
        createTime = test.loc[[testRow],"created_at_hour"].values

        PSTTime,PST = getTarget(getPSTS,postKey)
        PCCTTime,PCCT = getTarget(getPCCTS,postKey)
        PLTTime,PLT = getTarget(getPLTS,postKey)
        PCTTime,PCT = getTarget(getPCTS,postKey)

        createFeature(PST,PSTTime,createTime,testRow,testData,0,indexErrorCount,postKey)
        createFeature(PCCT,PCCTTime,createTime,testRow,testData,1,indexErrorCount,postKey)
        createFeature(PLT,PLTTime,createTime,testRow,testData,2,indexErrorCount,postKey)
        createFeature(PCT,PCTTime,createTime,testRow,testData,3,indexErrorCount,postKey)
                            
        print("Test：已完成"+str(testRow)+"筆資料建構\r",end = "")
    
    finishLoading = int(round(time.time() * 1000))
    loadingTime = (finishLoading - startLoading)/1000
    print("測試資料建構完畢!共花費了"+str(int(loadingTime / 60))+"分"+str(loadingTime % 60)+"秒")

    np.save('trainData', trainData)
    np.save('trainLabel', trainLabel)
    np.save('testData', testData)
    np.save('testLabel', testLabel)

    return trainData,trainLabel,testData,testLabel