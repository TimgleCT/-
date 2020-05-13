import matplotlib as mpl
mpl.use('Agg')
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
import timekeeper_data_matric

np.random.seed(10)
label_dict={0:"睡著", 1:"醒著"}

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.show()

#timekeeper_data_matric.create_data_matric()
matric_list = timekeeper_data_matric.data_matric_list
label_list = timekeeper_data_matric.label_list

num_train = len(matric_list)*0.8
train_label_list = []
train_matric_list = []
test_label_list = []
test_matric_list = []  
i = 0
print("將資料分為訓練資料與測試資料......") 
for matric in matric_list:
    if(i < num_train):
        train_matric_list.append(matric)
        i=i+1
    else:
        test_matric_list.append(matric)
        i=i+1
i = 0
for label in label_list:
    if(i < num_train):
        train_label_list.append(label)
        i=i+1
    else:
        test_label_list.append(label)
        i=i+1

train_matric_array = np.array(train_matric_list)
train_label_array = np.array(train_label_list)
test_matric_array = np.array(test_matric_list)
test_label_array = np.array(test_label_list)


print(train_matric_array.shape)
print(test_matric_array.shape)
print(train_label_array.shape)
print(test_label_array.shape)
#print(train_matric_array[0])

print("將矩陣轉為一維中....")


x_matric_train_normalize = train_matric_array.reshape(train_matric_array.shape[0],11,11,5).astype('float32') 
x_matric_test_normalize = test_matric_array.reshape(test_matric_array.shape[0],11,11,5).astype('float32')


y_label_train_OneHot = np_utils.to_categorical(train_label_array,num_classes=2)
y_label_test_OneHot = np_utils.to_categorical(test_label_array,num_classes=2)


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
model.add(Dense(2, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='adam', 
              metrics=['accuracy'])
# 模型的載入
#try:
#     model.load_weights("SaveModel/TimeKeeper_AI_matric.h5")
#     print("載入模型成功!繼續訓練模型")
#except :    
#     print("載入模型失敗!開始訓練一個新模型")


train_history=model.fit(x_matric_train_normalize,           
                       y_label_train_OneHot,
                       validation_split=0.2,
                       epochs=20,
                       batch_size=8, 
                       verbose=2)
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

 # 進行預測
scores = model.evaluate(x_matric_test_normalize, y_label_test_OneHot)
print('Accuracy=', scores[1])

prediction=model.predict_classes(x_matric_test_normalize)
print(prediction)

test = test_label_array.reshape(-1)
df=pd.DataFrame({ 'label':test, 'predict':prediction})
print(df)

print("Save Model")
model.save('SaveModel/TimeKeeper_AI_matric.h5')   # HDF5 file, you have to pip3 install h5py if don't have it
del model  # deletes the existing model