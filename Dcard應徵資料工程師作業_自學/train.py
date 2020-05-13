import os
import pandas as pd
import numpy as np
from preprocess_data import preprocessingNP
from collections import Counter
from keras.callbacks import Callback,EarlyStopping
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation ,Reshape
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D,Flatten
import keras.backend as K
from keras.utils import np_utils
import matplotlib.pyplot as plt
from datetime import datetime
from imblearn.over_sampling import ADASYN
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,roc_curve,auc
from sklearn.utils import class_weight
import sys
import tensorflow as tf

host = sys.argv[1]
modelFilePath = sys.argv[2]
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss
    
    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    
        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed

def showTrainHistory(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.show()

def showROC(y,prob):
    fpr,tpr,threshold = roc_curve(y,prob)
    roc_auc = auc(fpr,tpr) 
    print('ROC分數 =',roc_auc)
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC曲線')
    plt.legend(loc="lower right")
    plt.show()

def showPSTDisrtibution(trainDataArray):
    plt.figure()
    for dataNum in range(trainDataArray.shape[0]):
        if(trainLabelArray[dataNum] == False):
            plt.plot(trainDataArray[dataNum][0],  label='0', color='black',linewidth=0.5)
        else:
            plt.plot(trainDataArray[dataNum][0],  label='1', color='red',linewidth=0.5)
    plt.show()

def showPCCTDisrtibution(trainDataArray):
    plt.figure()
    for dataNum in range(trainDataArray.shape[0]):
        if(trainLabelArray[dataNum] == False):
            plt.plot(trainDataArray[dataNum][1],  label='0', color='black',linewidth=0.5)
        else:
            plt.plot(trainDataArray[dataNum][1],  label='1', color='red',linewidth=0.5)
    plt.show()

def showPLTDisrtibution(trainDataArray):
    plt.figure()
    for dataNum in range(trainDataArray.shape[0]):
        if(trainLabelArray[dataNum] == False):
            plt.plot(trainDataArray[dataNum][2],  label='0', color='black',linewidth=0.5)
        else:
            plt.plot(trainDataArray[dataNum][2],  label='1', color='red',linewidth=0.5)
    plt.show()

def showPCTDisrtibution(trainDataArray):
    plt.figure()
    for dataNum in range(trainDataArray.shape[0]):
        if(trainLabelArray[dataNum] == False):
            plt.plot(trainDataArray[dataNum][3], label='0', color='black',linewidth=0.5)
        else:
            plt.plot(trainDataArray[dataNum][3], label='1', color='red',linewidth=0.5)
    plt.show()


if os.path.isfile("trainData.npy") == False:
    trainDataArray, trainLabelArray, testDataArray, testLabelArray = preprocessingNP(host)
else:
    trainDataArray = np.load('trainData.npy')
    trainLabelArray = np.load('trainLabel.npy')
    testDataArray = np.load('testData.npy')
    testLabelArray = np.load('testLabel.npy')

# showPSTDisrtibution(trainDataArray)
# showPCCTDisrtibution(trainDataArray)
# showPLTDisrtibution(trainDataArray)
# showPCTDisrtibution(trainDataArray)



print(trainDataArray.shape)
print(testDataArray.shape)
print(trainLabelArray.shape)
print(testLabelArray.shape)

np.random.seed(10)

trainDataArrayReshape = trainDataArray.reshape(trainDataArray.shape[0],44).astype('float32')
testDataArrayReshape = testDataArray.reshape(testDataArray.shape[0],44).astype('float32')

# print(Counter(trainLabelArray))
# trainDataArrayReshape, trainLabelArray = ADASYN().fit_sample(trainDataArrayReshape, trainLabelArray)
# print(Counter(trainLabelArray).items())

trainDataArrayReshape = trainDataArrayReshape.reshape(trainDataArrayReshape.shape[0],44,1).astype('float32')
testDataArrayReshape = testDataArrayReshape.reshape(testDataArrayReshape.shape[0],44,1).astype('float32')

# newTrainDataArray = trainDataArrayReshape.reshape(trainDataArrayReshape.shape[0],4,11).astype('float32')
# showPSTDisrtibution(newTrainDataArray)
# showPCCTDisrtibution(newTrainDataArray)
# showPLTDisrtibution(newTrainDataArray)
# showPCTDisrtibution(newTrainDataArray)

trainLabelArrayOneHot = np_utils.to_categorical(trainLabelArray,num_classes=2)
testLabelArrayOneHot = np_utils.to_categorical(testLabelArray,num_classes=2)

model = Sequential()
model.add(Reshape((11, 4), input_shape=(44,1)))
model.add(Conv1D(80, 5, activation='relu', input_shape=(11, 4),padding='same'))#原本是90
model.add(Conv1D(90, 5, activation='relu',padding='same'))#原本是110
# model.add(MaxPooling1D(5))#5
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(units=22,kernel_initializer='normal', activation='relu')) #原本是22
model.add(Dropout(rate=0.25))
model.add(Dense(units=10,kernel_initializer='normal', activation='relu')) 
model.add(Dropout(rate=0.25))                
model.add(Dense(units=2,kernel_initializer='normal',activation='sigmoid'))
# model.compile(loss='binary_crossentropy',
model.compile(loss=[binary_focal_loss(alpha=.23, gamma=2)],#alpha=.2, gamma=2
              optimizer='adamax', #adam
                metrics=['accuracy'])


earlyStopping = EarlyStopping(monitor='loss', patience=5, verbose=2,mode='min')
classWeights = {0:6.08,1:10} #{0:5.3,1:10}
print(model.summary())

# model.load_weights(modelFilePath)

trainHistory=model.fit(trainDataArrayReshape,           
                       trainLabelArrayOneHot,
                       validation_split=0.3,
                       epochs=50,
                       batch_size=30, 
                       verbose=1,
                       class_weight=classWeights,
                       callbacks=[earlyStopping])

showTrainHistory(trainHistory,'accuracy','val_accuracy')
showTrainHistory(trainHistory,'loss','val_loss')

score = model.evaluate(testDataArrayReshape, testLabelArrayOneHot)
print('accuracy =', score[1])


prediction = model.predict_classes(testDataArrayReshape)

accuracy = accuracy_score(testLabelArray, prediction)
precision = precision_score(testLabelArray, prediction)
recall = recall_score(testLabelArray, prediction)
f1 = f1_score(testLabelArray, prediction)
print('accuracy =', accuracy)
print('precision =', precision)
print('recall =', recall)
print('f1 =', f1)

showROC(testLabelArray,prediction.ravel())

print(prediction)
df=pd.DataFrame({ 'label':testLabelArray, 'predict':prediction})
print(df)

con = pd.crosstab(testLabelArray,prediction,
            rownames=['label'],
            colnames=['predict'])
print(con)

print("儲存模型")
# print(model.summary())
model.save(modelFilePath)
