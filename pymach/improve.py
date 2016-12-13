#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module provides ideas for improving some machine learning algorithms.

"""
from __future__ import print_function
import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt
import warnings
#sklearn warning
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV



class Improve():
    """ A class for resampling and evaluation """

    bestAlgorithms = {}
    pipelines = None
    X_train = None
    y_train = None
    X_test = None
    y_test = None


    def __init__(self, definer, preparer, featurer):
        self.definer = definer 
        self.preparer = preparer
        self.featurer = featurer


    def pipeline(self):

        #evaluators = []
        #evaluators.append(self.evaluatePipelines())
        #[m() for m in evaluators]

        Evaluate.pipelines = self.buildPipelines(self.defineAlgorithms())


        return self


    def evaluatePipelines(self):

        test_size = 0.33
        num_folds = 10
        seed = 7
        scoring = 'accuracy'

        pipelines = self.buildPipelines(self.defineAlgorithms())
        X_train, X_test, Y_train, Y_test = self.defineTrainingData(test_size, seed)


        #report = {}
        #report_element = {}
        report = [["Model", "Mean", "STD"]]
        results = []
        names = []

        for name, model in pipelines:
            kfold = KFold(n_splits=num_folds, random_state=seed)
            cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, 
                    scoring=scoring)
            results.append(cv_results)
            names.append(name)

            mean = cv_results.mean()
            std = cv_results.std()
            #report_element[name] = {'mean':mean, 'std':std}
            #report.update(report_element)

            #report_print = "Model: {}, mean: {}, std: {}".format(name, 
                    #mean, std)
            report.append([name, mean, std])
            #print(report_print)

        headers = report.pop(0)
        df_report = pd.DataFrame(report, columns=headers)
        #print(df_report)

        #print(report)
        #self.chooseTopRanked(report)
        self.chooseTopRanked(df_report)
        #self.plotModels(results, names)

        
    def chooseTopRanked(self, report):
        """" Choose the best two algorithms"""

        #sorted_t = sorted(report.items(), key=operator.itemgetter(1))
        report.sort_values(['Mean'], ascending=[False], inplace=True)
        #Evaluate.bestAlgorithms = sorted_t[-2:]
        Evaluate.bestAlgorithms = report

        print(Evaluate.bestAlgorithms)

    def plotModels(self, results, names):
        """" Plot the best two algorithms by using box plots"""

        fig = plt.figure()
        fig.suptitle("Model Comparison")
        ax = fig.add_subplot(111) 
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show()

    class CustomFeature(TransformerMixin):
        """ A custome class for modeling """

        def transform(self, X, **transform_params):
            #X = pd.DataFrame(X)
            return X

        def fit(self, X, y=None, **fit_params):
            return self
