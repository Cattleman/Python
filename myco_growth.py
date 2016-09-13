# -*- coding: utf-8 -*-
"""
Created on Fri Sep 09 13:06:02 2016

@author: Hensel
"""
from sklearn.decomposition import FastICA, PCA, NMF  
from sklearn.ensemble import GradientBoostingClassifier  
from sklearn.svm import SVC  
import numpy as np  
import pandas as pd  
from sklearn.linear_model import LogisticRegression  
from sklearn.naive_bayes import GaussianNB  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.ensemble import VotingClassifier  
from sklearn.neural_network import BernoulliRBM  
from sklearn.pipeline import Pipeline  
from sklearn import preprocessing  
from sklearn.calibration import CalibratedClassifierCV  
from sklearn.cluster import MiniBatchKMeans

dip = pd.read_csv('numerai_training_data.csv')

training = dip.values[:, :21]  
classes = dip.values[:, -1]  
training = preprocessing.scale(training)  
kmeans = MiniBatchKMeans(n_clusters=500, init_size=6000).fit(training)  
labels = kmeans.predict(training)

clusters = {}  
for i in range(0, np.shape(training)[0]):  
    label = labels[i]  
    if label not in clusters:  
        clusters[label] = training[i, :]  
    else:  
        clusters[label] = np.vstack((clusters[label], training[i, :]))
#depth defalt is 3, random state is 3 learning rate is 0.01
params = {'n_estimators': 1000, 'max_depth': 3, 'subsample': 0.5,  
          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}  
gbm = GradientBoostingClassifier(**params)  
ica = FastICA(10)

icas = {}  
for label in clusters:  
    icas[label] = ica.fit(clusters[label])

factors = np.zeros((np.shape(training)[0], 10))

for i in range(0, np.shape(training)[0]):  
    factors[i, :] = icas[labels[i]].transform(training[i, :].reshape(1, -1))

gbm = gbm.fit(factors, classes)    

tf = pd.read_csv("numerai_tournament_data.csv")  
forecast = tf.values[:, 1:]  
forecast = preprocessing.scale(forecast)  
labels = kmeans.predict(forecast)

factors = np.zeros((np.shape(labels)[0], 10))  
for i, label in enumerate(labels):  
    factors[i, :] = icas[label].transform(forecast[i, :].reshape(1, -1))

proba = gbm.predict_proba(factors)

of = pd.Series(proba[:, 1], index=tf.values[:, 0])

of.to_csv("myco_growth.csv", header=['probability'], index_label='t_id')  