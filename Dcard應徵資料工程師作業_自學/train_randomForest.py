import os
import pandas as pd
import numpy as np
from preprocess_data import preprocessingNP
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,roc_curve,auc
from sklearn import ensemble
from sklearn.externals import joblib
import sys

host = sys.argv[1]

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

trainDataArray = trainDataArray.reshape(trainDataArray.shape[0],44).astype('float32')
testDataArray = testDataArray.reshape(testDataArray.shape[0],44).astype('float32')

forest = ensemble.RandomForestClassifier(n_estimators = 375)
forest_fit = forest.fit(trainDataArray, trainLabelArray)
output = forest_fit.predict(testDataArray)


con = pd.crosstab(testLabelArray,output,
        rownames=['label'],
        colnames=['predict'])
print(con)
accuracy = accuracy_score(testLabelArray, output)
precision = precision_score(testLabelArray, output)
recall = recall_score(testLabelArray, output)
f1 = f1_score(testLabelArray, output)
print('accuracy =', accuracy)
print('precision =', precision)
print('recall =', recall)
print('f1 =', f1)

joblib.dump(forest_fit, 'random_forest.pkl')

showROC(testLabelArray,output.ravel())