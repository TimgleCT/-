import os
import pandas as pd
import numpy as np
from preprocess_data import preprocessingNPForTest
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation ,Reshape
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
import keras.backend as K
import sys

def getRecall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def getPrecision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def getF1(y_true, y_pred):
    precision = getPrecision(y_true, y_pred)
    recall = getRecall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



host = sys.argv[1]
modelFilePath = sys.argv[2]
outputFilePath = sys.argv[3]

np.random.seed(10)
testDataArray , postKey = preprocessingNPForTest(host)
print("測試資料已匯入!")
testDataArrayReshape = testDataArray.reshape(testDataArray.shape[0],44,1).astype('float32')


model = Sequential()
model.add(Reshape((11, 4), input_shape=(44,1)))
model.add(Conv1D(70, 5, activation='relu', input_shape=(11, 4),padding='same'))
model.add(Conv1D(70, 5, activation='relu',padding='same'))
model.add(MaxPooling1D(7))
model.add(Conv1D(100, 3, activation='relu',padding='same'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(units=33,kernel_initializer='normal', activation='relu')) 
model.add(Dropout(rate=0.5)) 
model.add(Dense(units=22,kernel_initializer='normal', activation='relu')) 
model.add(Dropout(rate=0.5))
model.add(Dense(units=10,kernel_initializer='normal', activation='relu')) 
model.add(Dropout(rate=0.5))                
model.add(Dense(units=2,kernel_initializer='normal',activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam', 
                metrics=['accuracy',getF1])

try:
    model.load_weights(modelFilePath)
    print("載入模型成功!")
    prediction = model.predict_classes(testDataArrayReshape)
    output = pd.DataFrame({ 'post_key':postKey, 'is_trending':prediction})
    output.set_index('post_key', inplace = True)
    output.to_csv(outputFilePath)
    print("預測完成!")
except :    
    print("載入模型失敗!")

