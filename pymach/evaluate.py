#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module provides ideas for evaluating some machine learning algorithms.

"""
from __future__ import print_function
import operator
import warnings
import pickle

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
# import plotly.plotly as py
import plotly.graph_objs as go
import cufflinks as cf # Needed
#sklearn warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

from collections import OrderedDict
from plotly.offline.offline import _plot_html

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import FunctionTransformer

#Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Ensembles algorithms
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier


class Evaluate():
    """ A class for resampling and evaluation """



    def __init__(self, definer, preparer, selector):
        self.definer = definer
        self.preparer = preparer
        self.selector = selector
        self.plot_html = None

        self.report = None
        self.raw_report = None
        self.best_pipelines = None
        self.pipelines = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.test_size = 0.2
        self.num_folds = 10
        self.seed = 7

    def pipeline(self):

        #evaluators = []
        self.build_pipelines()
        self.split_data(self.test_size, self.seed)
        self.evaluate_pipelines()
        self.set_best_pipelines()

        #[m() for m in evaluators]
        return self

    def set_models(self):

        models = []
        # LDA : Warning(Variables are collinear)
        models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
        models.append(('SVC', SVC()))
        models.append(('GaussianNB', GaussianNB()))
        models.append(('KNeighborsClassifier', KNeighborsClassifier()))
        models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
        models.append(('LogisticRegression', LogisticRegression()))

        # Bagging and Boosting
        # models.append(('ExtraTreesClassifier', ExtraTreesClassifier(n_estimators=150)))
        models.append(('ExtraTreesClassifier', ExtraTreesClassifier()))
        models.append(('AdaBoostClassifier', AdaBoostClassifier(DecisionTreeClassifier())))
        # models.append(('AdaBoostClassifier', AdaBoostClassifier(DecisionTreeClassifier())))
        models.append(('RandomForestClassifier', RandomForestClassifier()))
        models.append(('GradientBoostingClassifier',
                       GradientBoostingClassifier(n_estimators=150, learning_rate=0.2)))
        # models.append(('GradientBoostingClassifier', GradientBoostingClassifier()))

        # Voting
        estimators = []
        estimators.append(("Voting_GradientBoostingClassifier", GradientBoostingClassifier()))
        estimators.append(("Voting_ExtraTreesClassifier", ExtraTreesClassifier()))
        voting = VotingClassifier(estimators)
        models.append(('VotingClassifier', voting))

        return models

    def split_data(self, test_size=0.20, seed=7):
        """ Need to fill """

        X_train, X_test, y_train, y_test =  train_test_split(
                self.definer.X, self.definer.y, test_size=test_size, random_state=seed)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # return X_train, X_test, y_train, y_test


    def build_pipelines(self):
        pipelines = []
        models = self.set_models()

        for m in models:
            pipelines.append((m[0],
                Pipeline([
                    #('preparer', FunctionTransformer(self.preparer)),
                    ('preparer', self.preparer),
                    ('selector', self.selector),
                    m,
                ])
            ))

        self.pipelines = pipelines

        return pipelines

    def evaluate_pipelines(self, ax=None):

        test_size = self.test_size
        num_folds = self.num_folds
        seed = self.seed
        scoring = 'accuracy'

        #pipelines = self.build_pipelines(self.set_models())
        #pipelines = self.pipelines


        #self.report = {}
        #report_element = {}
        self.report = [["Model", "Mean", "STD"]]
        results = []
        names = []

        for name, model in self.pipelines:
            print("Performing grid search...", name)

            kfold = KFold(n_splits=num_folds, random_state=seed)
            #cv_results = cross_val_score(model, self.definer.data.ix[:,:-1], self.definer.data.ix[:,-1], cv=kfold, \
                    #scoring=scoring)
            cv_results = cross_val_score(model, self.X_train, self.y_train, cv=kfold, \
                    scoring=scoring)

            # save the model to disk
            #filename = name+'.ml'
            #pickle.dump(model, open('./models/'+filename, 'wb'))

            #results.append(cv_results)
            mean = cv_results.mean()
            std = cv_results.std()

            d = {'name': name, 'values': cv_results, 'mean': round(mean, 3), 'std': round(std, 3)}
            results.append(d)
            #results['result'] = cv_results
            #names.append(name)
            #report_element[name] = {'mean':mean, 'std':std}
            #self.report.update(report_element)

            #report_print = "Model: {}, mean: {}, std: {}".format(name,
                    #mean, std)
            self.report.append([name, round(mean,3), round(std,3)])
            #print(report_print)

        self.raw_report = sorted(results, key=lambda k: k['mean'], reverse=True)
        #print(self.raw_report)
        headers = self.report.pop(0)
        df_report = pd.DataFrame(self.report, columns=headers)
        #print(df_report)

        #print(self.report)
        #self.sort_report(self.report)
        self.sort_report(df_report)
        #self.plotModels(results, names)


    def sort_report(self, report):
        """" Choose the best two algorithms"""

        #sorted_t = sorted(report.items(), key=operator.itemgetter(1))
        report.sort_values(['Mean'], ascending=[False], inplace=True)
        #self.bestAlgorithms = sorted_t[-2:]
        self.report = report.copy()

        #print(self.report)

    def set_best_pipelines(self):
        alg = list(self.report.Model)[0:2]
        best_pipelines = []

        for p in self.pipelines:
            if p[0] in alg:
                best_pipelines.append(p)

        self.best_pipelines = best_pipelines

        #print(self.best_pipelines)

    def plot_to_html(self, fig):
        plotly_html_div, plotdivid, width, height = _plot_html(
                figure_or_data=fig,
                config="",
                validate=True,
                default_width='90%',
                default_height="100%",
                global_requirejs=False)

        return plotly_html_div

    def plot_models(self):
        """" Plot the algorithms by using box plots"""
        #df = pd.DataFrame.from_dict(self.raw_report)
        #print(df)

        results = self.raw_report
        data = []
        N = len(results)
        c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 270, N)]

        for i, d in enumerate(results):
            trace = go.Box(
                y=d['values'],
                name=d['name'],
                marker=dict(
                    color=c[i],
                ),
                boxmean='sd'
            )
            data.append(trace)

        text_scatter = go.Scatter(
                x=[d['name'] for d in results],
                y=[d['mean'] for d in results],
                name='score',
                mode='markers',
                text=['Explanation' for _ in results]
        )
        data.append(text_scatter)
        layout = go.Layout(
            #showlegend=False,
            title='Hover over the bars to see the details',
            annotations=[
                dict(
                    x=results[0]['name'],
                    y=results[0]['mean'],
                    xref='x',
                    yref='y',
                    text='Best model',
                    showarrow=True,
                    arrowhead=7,
                    ax=0,
                    ay=-40
                ),
                dict(
                    x=results[-1]['name'],
                    y=results[-1]['mean'],
                    xref='x',
                    yref='y',
                    text='Worst model',
                    showarrow=True,
                    arrowhead=7,
                    ax=0,
                    ay=-40
                )
            ]
        )

        fig = go.Figure(data=data, layout=layout)

        self.plot_html = self.plot_to_html(fig)
        return self.plot_html

    def save_plot(self, path):
        with open(path, "w") as plot:
            plot.write(self.plot_html)

    def save_report(self, path):
        # with open(path, "w") as plot:
        self.report.to_csv(path, index=False)
        # plot.write(valuate.report.to_csv())

    class CustomFeature(TransformerMixin):
        """ A custome class for modeling """

        def transform(self, X, **transform_params):
            #X = pd.DataFrame(X)
            return X

        def fit(self, X, y=None, **fit_params):
            return self
