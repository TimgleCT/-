# -*- coding: UTF-8 -*-
import numpy
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
import pandas as pd
from PIL import Image
import glob
import os
import timekeeper_single_data_matric
import sys

np.random.seed(10)


test = sys.argv[1]
timekeeper_single_data_matric.create_data_matric(test)
matric_list = timekeeper_single_data_matric.data_matric_list
label_list = timekeeper_single_data_matric.label_list

num_train = len(matric_list)*0.8
train_label_list = []
train_matric_list = []
test_label_list = []
test_matric_list = []  

#print("將資料分為訓練資料與測試資料......") 
for matric in matric_list:
    test_matric_list.append(matric)
    
for label in label_list:
    test_label_list.append(label)
    
test_matric_array = np.array(test_matric_list)
test_label_array = np.array(test_label_list)



#print(test_matric_array.shape)
#print(test_label_array.shape)
#print(train_matric_array[0])

#print("將矩陣轉為一維中....")



x_matric_test_normalize = test_matric_array.reshape(test_matric_array.shape[0],11,11,5).astype('float32')
#y_label_test_OneHot = np_utils.to_categorical(test_label_array,num_classes=2)

model = Sequential()
# 卷基層1
model.add(Conv2D(filters=32,kernel_size=(3,3),
                 input_shape=(11,11,5), 
                 activation='relu', 
                 padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
# 卷積層2與池化層2
model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                 activation='relu', 
                 padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 平坦層、隱藏層、輸出層
model.add(Flatten())
model.add(Dropout(rate=0.25))
model.add(Dense(605, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(121, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(11, activation='relu'))
model.add(Dense(2, activation='softmax'))

#print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam', 
              metrics=['accuracy'])

model.load_weights("SaveModel/TimeKeeper_AI_matric.h5")
# scores = model.evaluate(x_matric_test_normalize, y_label_test_OneHot)
# print('準確度=', scores[1])

test = test_label_array.reshape(-1)
prediction=model.predict_classes(x_matric_test_normalize)
df=pd.DataFrame({ 'label':test, 'predict':prediction})
#print(df)
if(prediction == 1):
    print(1)
else:
    print(0)