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
    """ A class for improving """

    bestConfiguration = None


    def __init__(self, evaluator):
        self.pipeline = evaluator.pipelines 


    def pipeline(self):

        self.evaluatePipelines()

        return self


    def evaluatePipelines(self):
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier()),
        ])

        parameters = {
            '__max_df': (0.5, 0.75, 1.0),
            #'vect__max_features': (None, 5000, 10000, 50000),
            'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
            #'tfidf__use_idf': (True, False),
            #'tfidf__norm': ('l1', 'l2'),
            'clf__alpha': (0.00001, 0.000001),
            'clf__penalty': ('l2', 'elasticnet'),
            #'clf__n_iter': (10, 50, 80),
        }

        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        pprint(parameters)
        t0 = time()
        grid_search.fit(data.data, data.target)
        print("done in %0.3fs" % (time() - t0))
        print()

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))


        
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
