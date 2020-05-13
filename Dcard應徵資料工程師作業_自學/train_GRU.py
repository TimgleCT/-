import os
import pandas as pd
import numpy as np
from preprocess_data import preprocessingNP
from keras.callbacks import Callback,EarlyStopping
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation,GRU,TimeDistributed,Flatten
from keras.utils import np_utils
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,roc_curve,auc
from sklearn.utils import class_weight
import sys

host = sys.argv[1]
modelFilePath = sys.argv[2]

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

if os.path.isfile("trainData.npy") == False:
    trainDataArray, trainLabelArray, testDataArray, testLabelArray = preprocessingNP(host)
else:
    trainDataArray = np.load('trainData.npy')
    trainLabelArray = np.load('trainLabel.npy')
    testDataArray = np.load('testData.npy')
    testLabelArray = np.load('testLabel.npy')

print(trainDataArray.shape)
print(testDataArray.shape)
print(trainLabelArray.shape)
print(testLabelArray.shape)

np.random.seed(10)

trainDataArrayReshape = trainDataArray.reshape(trainDataArray.shape[0],11,4).astype('float32')
testDataArrayReshape = testDataArray.reshape(testDataArray.shape[0],11,4).astype('float32')


model = Sequential()
model.add(GRU(units=11, input_shape=(11, 4), return_sequences=True, dropout=0.2))
model.add(GRU(units=11,return_sequences=True, dropout=0.2))
model.add(TimeDistributed(Dense(1)))
model.add(Flatten())
model.add(Dense(units=6,kernel_initializer='normal', activation='relu')) #unit = 6
model.add(Dropout(rate=0.2)) 
model.add(Dense(units=3,kernel_initializer='normal', activation='relu')) #unit = none
model.add(Dropout(rate=0.2))               
model.add(Dense(units=1,kernel_initializer='normal',activation='sigmoid'))
model.compile(loss='binary_crossentropy',
                optimizer='adam', 
                metrics=['accuracy'])


# earlyStopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=2,mode='max')
classWeights = {0:5.3,1:10}

trainHistory=model.fit(trainDataArrayReshape,           
                       trainLabelArray,
                       validation_split=0.3,
                       epochs=100,
                       batch_size=30, 
                       verbose=1,
                       class_weight=classWeights)
                    #    callbacks=[earlyStopping])

showTrainHistory(trainHistory,'accuracy','val_accuracy')
showTrainHistory(trainHistory,'loss','val_loss')

score = model.evaluate(testDataArrayReshape, testLabelArray)
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
prediction = prediction.reshape(prediction.shape[0])
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
