# -*- coding: utf-8 -*-
"""
For Python 3

Sklearn 15+ is required for BaggingClassifier
@author: kunclvad

"""
#imports
from walkforest.walkforest_new import WalkForestClassifier,WalkForestHyperLearner
import pandas as pd
import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix
from sklearn import cross_validation
import scipy.sparse as sp
import pickle
from sklearn.ensemble import BaggingClassifier
from sklearn import svm


import itertools
from sklearn.utils import check_random_state #, check_X_y, check_array, column_or_1d
import numbers
from sklearn.ensemble.base import BaseEnsemble, _partition_estimators
from sklearn.ensemble.bagging import _parallel_build_estimators
from sklearn.externals.joblib import Parallel, delayed

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.dummy import DummyClassifier

from networkclassif import GeneNetwork, NetworkBaggingClassifier, random_feature_sets
gn = GeneNetwork()
print('n of causgenes')
print(len(gn.causgenes))
fs = gn.get_n_feature_sets(5,5)
print('fs')
print(fs)
print('random')
fs2 = random_feature_sets(5,5,gn.causgenes)
print(fs2)
#learning
folds = cross_validation.StratifiedKFold(gn.classes,n_folds=min(10,min(sum(gn.classes==1),sum(gn.classes==0))))

learner = NetworkBaggingClassifier(gnetwork=gn,
                                   n_estimators = 1000,max_features=100,bootstrap=False,
                                   choosing_function=gn.get_n_feature_sets,
                                   choosing_function_kwargs=[1000,100])
pa=[]
for itrain, itest in folds:
    #print('Trenuji')
    learner.fit(gn.data[itrain,:],gn.classes[itrain])
    preds=learner.predict(gn.data[itest,:])
    pa.append(accuracy_score(gn.classes[itest], preds) )
    #print(confusion_matrix(classes[itest], preds))
print('prediction accuracy of Network Bagging Classifier - decision tree:')
print(pa)
print(np.mean(pa))

base_estim = svm.SVC()
learner = NetworkBaggingClassifier(gnetwork=gn,
                                   base_estimator=base_estim,
                                   n_estimators = 1000,max_features=100,bootstrap=False,
                                   choosing_function=gn.get_n_feature_sets,
                                   choosing_function_kwargs=[1000,1000])
pa=[]
for itrain, itest in folds:
    #print('Trenuji')
    learner.fit(gn.data[itrain,:],gn.classes[itrain])
    preds=learner.predict(gn.data[itest,:])
    pa.append(accuracy_score(gn.classes[itest], preds) )
    print(confusion_matrix(gn.classes[itest], preds))
    print(preds)
print('prediction accuracy of Network Bagging Classifier - linSVM):')
print(pa)
print(np.mean(pa))
# raise SystemExit(0)
# learner = BaggingClassifier(n_estimators = 1000,max_features=100,bootstrap=False)
# pa=[]
# for itrain, itest in folds:
#     #print('Trenuji')
#     learner.fit(gn.data[itrain,:],gn.classes[itrain])
#     preds=learner.predict(gn.data[itest,:])
#     pa.append(accuracy_score(gn.classes[itest], preds) )
#     #print(confusion_matrix(classes[itest], preds))
# print('prediction accuracy of randomly Bagging Classifier (all features):')
# print(pa)
# print(np.mean(pa))
#
# learner = BaggingClassifier(n_estimators = 1000,max_features=100,bootstrap=False)
# pa=[]
# for itrain, itest in folds:
#     #print('Trenuji')
#     X_train = gn.data[itrain,:]
#     y_train = gn.classes[itrain]
#     y_test = gn.classes[itest]
#     X_test = gn.data[itest,:]
#     learner.fit(X_train[:,gn.causgenes],y_train)
#     preds=learner.predict(X_test[:,gn.causgenes])
#     pa.append(accuracy_score(y_test, preds) )
#     #print(confusion_matrix(classes[itest], preds))
# print('prediction accuracy of randomly Bagging Classifier using Decision Trees (only causgenes):')
# print(pa)
# print(np.mean(pa))
#
# base_estim = DecisionTreeClassifier(criterion='gini',max_depth=2,min_samples_leaf=2)
# learner = BaggingClassifier(base_estimator=base_estim,n_estimators = 1000,max_features=100,bootstrap=False)
# pa=[]
# for itrain, itest in folds:
#     #print('Trenuji')
#     X_train = gn.data[itrain,:]
#     y_train = gn.classes[itrain]
#     y_test = gn.classes[itest]
#     X_test = gn.data[itest,:]
#     learner.fit(X_train[:,gn.causgenes],y_train)
#     preds=learner.predict(X_test[:,gn.causgenes])
#     pa.append(accuracy_score(y_test, preds) )
#     #print(confusion_matrix(classes[itest], preds))
# print('prediction accuracy of randomly Bagging Classifier using Decision Trees similar to WFC (only causgenes):')
# print(pa)
# print(np.mean(pa))
#
# base_estim = ExtraTreeClassifier(criterion='gini',max_depth=2,min_samples_leaf=2)
# learner = BaggingClassifier(base_estimator=base_estim,n_estimators = 1000,max_features=100,bootstrap=False)
# pa=[]
# for itrain, itest in folds:
#     #print('Trenuji')
#     X_train = gn.data[itrain,:]
#     y_train = gn.classes[itrain]
#     y_test = gn.classes[itest]
#     X_test = gn.data[itest,:]
#     learner.fit(X_train[:,gn.causgenes],y_train)
#     preds=learner.predict(X_test[:,gn.causgenes])
#     pa.append(accuracy_score(y_test, preds) )
#     #print(confusion_matrix(classes[itest], preds))
# print('prediction accuracy of randomly Bagging Classifier using Extra Trees similar to WFC (only causgenes):')
# print(pa)
# print(np.mean(pa))
#
base_estim = DummyClassifier(strategy='most_frequent')
learner = BaggingClassifier(base_estimator=base_estim,n_estimators = 1000,max_features=100,bootstrap=False)
pa=[]
for itrain, itest in folds:
    #print('Trenuji')
    X_train = gn.data[itrain,:]
    y_train = gn.classes[itrain]
    y_test = gn.classes[itest]
    X_test = gn.data[itest,:]
    learner.fit(X_train[:,gn.causgenes],y_train)
    preds=learner.predict(X_test[:,gn.causgenes])
    pa.append(accuracy_score(y_test, preds) )
    print(confusion_matrix(gn.classes[itest], preds))
    print(preds)
print('prediction accuracy of randomly Bagging Classifier using Dummy Classifier (most frequent)(only causgenes):')
print(pa)
print(np.mean(pa))

pa=[]
learner=WalkForestClassifier(gn.gene2gene,miRNA2gene=gn.miRNA2gene,\
        K=1,n_estimators=1000,max_features=100,\
        max_depth=2,min_samples_leaf=2,bootstrap=False)
for itrain, itest in folds:
    #print('Trenuji')
    learner.fit(gn.data[itrain,:],gn.classes[itrain])
    preds=learner.predict(gn.data[itest,:])
    pa.append(accuracy_score(gn.classes[itest], preds) )
    #print(confusion_matrix(classes[itest], preds))
print('prediction accuracy of original WalkForestClassifier:')
print(pa)
print(np.mean(pa))


