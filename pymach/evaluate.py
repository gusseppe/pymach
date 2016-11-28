#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module provides ideas for evaluating some machine learning algorithms.

"""
from __future__ import print_function
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

#Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClasifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Ensembles algorithms
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


class Evaluate():
    """ A class for resampling and evaluation """

    bestAlgorithms = None
    X_test = None
    y_test = None


    def __init__(self, definer, preparer, featurer):
        self.definer = self.definer 
        self.preparer = self.preparer
        self.featurer = self.featurer


    def pipeline(self):
        """ This function chooses the best way to find features"""


        return Evaluate.bestAlgorithms

    def defineAlgorithms(self):
        models = []
        models.append(('SVC', SVC()))
        models.append(('GNB', GaussianNB()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('DT', DecisionTreeClasifier()))
        models.append(('LR', LogisticRegression()))

        return models

    def defineTrainingData(self):
        """ Need to fill """
        pass
         
    #X_train, X_test, Y_train, Y_test =  cross_validation.train_test_split(definer.X, definer.y,


    def buildPipelines(self, models):
        pipelines = []

        for m in models:
            pipelines.append(
                Pipeline([
                    ('preparer', self.preparer),
                    ('featurer', self.featurer),
                    m,
                ])
            )

        return pipelines

    def evaluatePipelines(self):
        pipelines = self.buildPipelines(self.defineAlgorithms)

        num_folds = 10
        seed = 7
        scoring = 'accuracy'
        results = []
        names = []

        for name, model in pipelines:
            kfold = KFold(n_splits=num_folds, random_state=seed)
            """ Need to fill """

        #Choose the two best ones.


    class CustomFeature(TransformerMixin):
        """ A custome class for modeling """

        def transform(self, X, **transform_params):
            #X = pd.DataFrame(X)
            return X

        def fit(self, X, y=None, **fit_params):
            return self
