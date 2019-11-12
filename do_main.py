from svm_sgd import SVMSGD,MultiClassSVMSGD
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import StratifiedKFold,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier

import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import joblib
import os
import sklearn.metrics as skl_met
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import pandas as pd
svm =MultiClassSVMSGD(10,0.001,0.01)#SVMSGD()#OneVsRestClassifier(SVMSGD())

from sgd import AdalineSGD
sgd = OneVsRestClassifier(AdalineSGD())
from mlxtend.data import loadlocal_mnist
X,y = loadlocal_mnist(
        images_path='../train-images-idx3-ubyte', 
        labels_path='../train-labels-idx1-ubyte'
        )
test_x,test_y = loadlocal_mnist(
            images_path='../t10k-images-idx3-ubyte', 
            labels_path='../t10k-labels-idx1-ubyte'
            )
#test_x = StandardScaler().fit_transform(test_x)
l = 10000
X = X[:l]
y = y[:l]
nx = []
ny = []


svc = sgd #LinearSVC(loss='hinge') #SVC(kernel='linear')
X = StandardScaler().fit_transform(X)

common_param = {"C":[0.001,0.01,0.1,1,10,100,1000],"eta":[0.00001,0.0001,0.001,0.01,0.1]}
pipe_model = Pipeline([('regressor',svm)])
param_grid = [{"regressor__C":common_param["C"]
    ,
    "regressor__eta":common_param["eta"]

    #'regressor__penalty':common_param["penalty"]
    },]

gs = GridSearchCV(estimator=pipe_model,param_grid=param_grid,scoring="accuracy",cv=4,n_jobs=-1,refit=True)
gs.fit(X,y)
mini_test_x = test_x
mini_test_y = test_y
ovr1_pred = gs.predict(mini_test_x)
print("\tacc:",skl_met.accuracy_score(ovr1_pred,mini_test_y))
print("\tF1 Score:",skl_met.f1_score(mini_test_y,ovr1_pred,average="macro"))
print("\tRecall:",skl_met.recall_score(mini_test_y,ovr1_pred,average="macro"))
print("\tPrecision:",skl_met.precision_score(mini_test_y,ovr1_pred,average="macro"))
print("\tclassification report:")
print("\t\t",skl_met.classification_report(mini_test_y,ovr1_pred))
print("\tconfussion matrix:")
print("\t\t",skl_met.confusion_matrix(mini_test_y,ovr1_pred))
print("\tGS Best Score: ",gs.best_score_)
print("\tGS Best Param: ",gs.best_params_)
print("\tGS results_:",gs.cv_results_)
rst_pd = pd.DataFrame(gs.cv_results_)
rst_pd.to_csv("./result.csv",header=True)
