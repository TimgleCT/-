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
import matplotlib.pyplot as plt
np.random.seed(10)
label_dict={0:"十", 1:"一", 2:"二", 3:"三", 4:"四", 5:"五", 6:"六", 7:"七", 8:"八", 9:"九"}
def plot_images_labels_prediction(images, labels, prediction, idx, num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25: num = 25 
    for i in range(0, num):
        ax=plt.subplot(5, 5, 1 + i)
        ax.imshow(images[idx], cmap = 'binary')
        title= "label=" + str(labels[idx])
        if len(prediction) > 0:
            title = str(i) + ',' + label_dict[labels[i][0]]
            title += '=>' + label_dict[prediction[i]]
        ax.set_title(title, fontsize = 10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx += 1 
    plt.show()
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.show()

train_label_list = []
train_img_list = []
test_label_list = []
test_img_list = []  
i = 0
list = glob.glob('processed_img/*jpg')
#print(list)
print("跑迴圈把圖片資料與標籤存進矩陣......") 
for jpg in list:
    im  =  Image.open(jpg)
    data = np.array(im)
    route = os.path.splitext(jpg)[0]
    route1 = route.split("\\")
    route2 = route1[1].split("-")
    label = route2[2]
    if(i<15999):
        train_img_list.append(data)
        train_img_array = np.array(train_img_list)
        train_label_list.append(label)
        train_label_array = np.array(train_label_list)
        i=i+1
    else:
        test_img_list.append(data)
        test_img_array = np.array(test_img_list)
        test_label_list.append(label)
        test_label_array = np.array(test_label_list)
        i=i+1
    


    #print(im)
    #print(testimg[0])



plot_images_labels_prediction(train_img_array, train_label_array, [], 0)

print("將圖片二維矩陣轉為一維中....")

print(train_img_array.shape)
print(test_img_array.shape)
print(train_label_array.shape)
print(test_label_array.shape)
print(train_img_array[0])


x_img_train_normalize = train_img_array.reshape(train_img_array.shape[0],28,28,1).astype('float32') / 255.0
x_img_test_normalize = test_img_array.reshape(test_img_array.shape[0],28,28,1).astype('float32') / 255.0
y_label_train_OneHot = np_utils.to_categorical(train_label_array)
y_label_test_OneHot = np_utils.to_categorical(test_label_array)


model = Sequential()
# 卷基層1
model.add(Conv2D(filters=16,kernel_size=(3,3),
                 input_shape=(28, 28,1), 
                 activation='relu', 
                 padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
# 卷積層2與池化層2
model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                 activation='relu', 
                 padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 平坦層、隱藏層、輸出層
model.add(Flatten())
model.add(Dropout(rate=0.25))
model.add(Dense(392, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(392, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(10, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam', 
              metrics=['accuracy'])
# 模型的載入
try:
    model.load_weights("SaveModel/cifarCnnModelnew.h5")
    print("載入模型成功!繼續訓練模型")
except :    
    print("載入模型失敗!開始訓練一個新模型")


train_history=model.fit(x_img_train_normalize,           
                        y_label_train_OneHot,
                        validation_split=0.2,
                        epochs=20,
                        batch_size=128, 
                        verbose=2)
show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

# # 進行預測
scores = model.evaluate(x_img_test_normalize, y_label_test_OneHot)
print('準確度=', scores[1])

prediction=model.predict_classes(x_img_test_normalize)
print(prediction)



#plot_images_labels_prediction(test_img_array, test_label_array, prediction, 0,10)

# #印出混淆矩陣
con = pd.crosstab(test_label_array.reshape(-1),prediction,
            rownames=['label'],
            colnames=['predict'])

print(con)
test = test_label_array.reshape(-1)
df=pd.DataFrame({ 'label':test, 'predict':prediction})
print(df)
get_label = df.label==5  
get_predict = df.predict==3
print(df[get_label])
print(df[get_predict])
#print(df[get_label & get_predict])

#print(df[(df.label==5)&(df.predict==3)])

#plot_images_labels_prediction([test_img_array[i] for i in df[(df.label==5)&(df.predict==3)].index],[test_label_array[i] for i in df[(df.label==5)&(df.predict==3)].index],prediction,idx=0)