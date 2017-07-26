#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module provides ideas for improving some machine learning algorithms.

"""
from __future__ import print_function
import tools

import warnings
import sys
import pandas as pd
import numpy as np
import plotly.graph_objs as go
warnings.filterwarnings("ignore", category=DeprecationWarning)

from collections import OrderedDict
from time import time
from plotly.offline.offline import _plot_html
from scipy.stats import randint
from scipy.stats import expon

# from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



class Improve():
    """ A class for improving """

    bestConfiguration = None


    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.pipelines = evaluator.build_pipelines()
        self.search = None
        self.score_report = None
        self.full_report = None
        self.best_search = None
        self.best_model = None

    def pipeline(self):

        self.improve_grid_search()

        return self

    # @property
    # def gradientboosting_param(self, method='grid'):
    #
    #     parameters = {
    #         'selector__extraTC__n_estimators': [10, 15, 20, 25],
    #         'selector__extraTC__criterion': ['gini', 'entropy'],
    #         'selector__extraTC__n_jobs': [-1],
    #         'selector__pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
    #         'selector__pca__whiten': [True,False],
    #         'GradientBoostingClassifier__n_estimators': [100, 150, 200],
    #         'GradientBoostingClassifier__learning_rate': [0.1, 0.2, 0.4, 0.8, 1.0]
    #     }
    #
    #     if method == 'random':
    #         parameters['GradientBoostingClassifier__learning_rate'] = expon(0,1)
    #
    #     return parameters

    def adaboost_param(self, method='grid'):

        parameters = {
            'selector__extraTC__n_estimators': [10],
            'selector__extraTC__criterion': ['entropy'],
            'selector__extraTC__n_jobs': [-1],
            'selector__pca__svd_solver': ['randomized'],
            'selector__pca__whiten': [True],
            'AdaBoostClassifier__n_estimators': [50, 100],
            'AdaBoostClassifier__learning_rate': [1.0, 2.0]
        }

        if method == 'random':
            pass

        return parameters

    def voting_param(self, method='grid'):

        parameters = {
            'selector__extraTC__n_estimators': [10],
            'selector__extraTC__criterion': ['entropy'],
            'selector__extraTC__n_jobs': [-1],
            'selector__pca__svd_solver': ['randomized'],
            'selector__pca__whiten': [True],
            'VotingClassifier__voting': ['hard', 'soft']
        }

        if method == 'random':
            pass

        return parameters

    def gradientboosting_param(self, method='grid'):

        parameters = {
            # 'selector__extraTC__n_estimators': [10],
            # 'selector__extraTC__criterion': ['entropy'],
            # 'selector__extraTC__n_jobs': [-1],
            # 'selector__pca__svd_solver': ['randomized'],
            # 'selector__pca__whiten': [True],
            # 'GradientBoostingClassifier__n_estimators': [200],
            'GradientBoostingClassifier__max_depth': [3,6,9]
            # 'GradientBoostingClassifier__learning_rate': [0.2]
        }

        if method == 'random':
            parameters['GradientBoostingClassifier__learning_rate'] = expon(0,1)

        return parameters

    def extratrees_param(self, method='grid'):
        parameters = {
            'selector__extraTC__n_estimators': [10],
            # 'selector__extraTC__n_estimators': [10, 15, 20, 25],
            # 'selector__extraTC__criterion': ['gini', 'entropy'],
            'selector__extraTC__criterion': ['entropy'],
            'selector__extraTC__n_jobs': [-1],
            'selector__pca__svd_solver': ['randomized'],
            # 'selector__pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True],
            # 'selector__pca__whiten': [True,False],
            'ExtraTreesClassifier__n_estimators': [10, 15, 20, 25],
            'ExtraTreesClassifier__criterion': ['gini', 'entropy']
            # 'ExtraTreesClassifier__min_samples_leaf': range(200,1001,200),
            # 'ExtraTreesClassifier__max_leaf_nodes': [2,3,4,5],
            # 'ExtraTreesClassifier__max_depth': [2,3,4,5],
        }

        if method == 'random':
            parameters['ExtraTreesClassifier__min_samples_leaf'] = randint(200,1001)
            # parameters['ExtraTreesClassifier__max_leaf_nodes'] = randint(2,20)
            # parameters['ExtraTreesClassifier__max_depth'] = randint(1,20)
            pass

        return parameters

    def randomforest_param(self, method='grid'):
        parameters = {
            'selector__extraTC__n_estimators': [10],
            # 'selector__extraTC__n_estimators': [10, 15, 20, 25],
            'selector__extraTC__criterion': ['entropy'],
            # 'selector__extraTC__criterion': ['gini', 'entropy'],
            'selector__extraTC__n_jobs': [-1],
            'selector__pca__svd_solver': ['randomized'],
            # 'selector__pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True],
            # 'selector__pca__whiten': [True,False],
            'RandomForestClassifier__n_estimators': [10, 15, 20, 25],
            'RandomForestClassifier__criterion': ['gini', 'entropy'],
            'RandomForestClassifier__min_samples_leaf': [1,2,3,4,5],
            'RandomForestClassifier__max_leaf_nodes': [2,3,4,5],
            'RandomForestClassifier__max_depth': [2,3,4,5],
            'RandomForestClassifier__warm_start': [True,False]
        }
        if method == 'random':
            parameters['RandomForestClassifier__min_samples_leaf'] = randint(1,20)
            parameters['RandomForestClassifier__max_leaf_nodes'] = randint(2,20)
            parameters['RandomForestClassifier__max_depth'] = randint(1,20)

        return parameters

    def decisiontree_param(self, method='grid'):
        parameters = {
            'selector__extraTC__n_estimators':  [10],
            # 'selector__extraTC__n_estimators':  [10, 15, 20, 25],
            'selector__extraTC__criterion': ['entropy'],
            # 'selector__extraTC__criterion': ['gini','entropy'],
            'selector__extraTC__n_jobs': [-1],
            'selector__pca__svd_solver': ['randomized'],
            # 'selector__pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True],
            # 'selector__pca__whiten': [True,False],
            'DecisionTreeClassifier__max_features': ['sqrt','log2', None],
            'DecisionTreeClassifier__max_leaf_nodes': [2,3,4,5],
            'DecisionTreeClassifier__max_depth': [2,3,4,5],
            'DecisionTreeClassifier__min_samples_leaf': [1,2,3,4,5,10,15,20]

        }
        if method == 'random':
            parameters['DecisionTreeClassifier__min_samples_leaf'] = randint(1,20)
            parameters['DecisionTreeClassifier__max_leaf_nodes'] = randint(2,20)
            parameters['DecisionTreeClassifier__max_depth'] = randint(1,20)

        return parameters

    def lda_param(self, method='grid'):
        parameters = {
            'selector__extraTC__n_estimators':  [10],
            # 'selector__extraTC__n_estimators':  [10, 15, 20, 25],
            'selector__extraTC__criterion': ['entropy'],
            # 'selector__extraTC__criterion': ['gini','entropy'],
            'selector__extraTC__n_jobs': [-1],
            'selector__pca__svd_solver': ['randomized'],
            # 'selector__pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True],
            # 'selector__pca__whiten': [True,False],
            'LinearDiscriminantAnalysis__solver': ['svd','lsqr', 'eigen']

        }
        if method == 'random':
            pass

        return parameters

    def svc_param(self, method='grid'):
        parameters = {
            'selector__extraTC__n_estimators':  [10],
            # 'selector__extraTC__n_estimators':  [10, 15, 20, 25],
            'selector__extraTC__criterion': ['entropy'],
            # 'selector__extraTC__criterion': ['gini','entropy'],
            'selector__extraTC__n_jobs': [-1],
            # 'selector__pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
            'selector__pca__svd_solver': ['randomized'],
            'selector__pca__whiten': [True],
            # 'selector__pca__whiten': [True,False],
            # 'SVC__kernel': ['linear','poly', 'rbf','sigmoid','precomputed'],
            'SVC__kernel': ['rbf'],
            'SVC__C': [1, 10, 100],
            # 'SVC__decision_function_shape': ['ovo','ovr']
            'SVC__decision_function_shape': ['ovr']

        }

        if method == 'random':
            pass

        return parameters

    def knn_param(self, method='grid'):

        parameters = {
            # 'selector__extraTC__n_estimators':  [10, 15, 20, 25],
            'selector__extraTC__n_estimators':  [10],
            'selector__extraTC__criterion': ['gini','entropy'],
            'selector__extraTC__n_jobs': [-1],
            'selector__pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True],
            # 'selector__pca__whiten': [True,False],
            'KNeighborsClassifier__n_neighbors': [5,6,7,8,9,10],
            'KNeighborsClassifier__weights': ['uniform','distance'],
            # 'KNeighborsClassifier__algorithm': ['auto','ball_tree','kd_tree','brute']
            'KNeighborsClassifier__algorithm': ['auto']

        }

        if method == 'random':
            parameters['KNeighborsClassifier__n_neighbors'] = randint(5,10)

        return parameters

    def logistic_param(self, method='grid'):

        parameters = {
            'selector__extraTC__n_estimators':  [10],
            # 'selector__extraTC__n_estimators':  [10, 15, 20, 25],
            'selector__extraTC__criterion': ['gini','entropy'],
            'selector__extraTC__n_jobs': [-1],
            'selector__pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True],
            # 'selector__pca__whiten': [True,False],
            # 'LogisticRegression__penalty': ['l2'],
            # 'LogisticRegression__solver': ['newton-cg','lbfgs','liblinear','sag'],
            'LogisticRegression__solver': ['newton-cg','lbfgs'],
            'LogisticRegression__warm_start': [True,False]
        }
        if method == 'random':
            pass


        return parameters


    def get_params(self, model, method):
        if model == 'AdaBoostClassifier':
            return self.adaboost_param(method)
        elif model == 'VotingClassifier':
            return self.voting_param(method)
        elif model == 'GradientBoostingClassifier':
            return self.gradientboosting_param(method)
        elif model == 'ExtraTreesClassifier':
            return self.extratrees_param(method)
        elif model == 'RandomForestClassifier':
            return self.randomforest_param(method)
        elif model == 'DecisionTreeClassifier':
            return self.decisiontree_param(method)
        elif model == 'LinearDiscriminantAnalysis':
            return self.lda_param(method)
        elif model == 'SVC':
            return self.svc_param(method)
        elif model == 'KNeighborsClassifier':
            return self.knn_param(method)
        elif model == 'LogisticRegression':
            return self.logistic_param(method)
        return None

    def improve_grid_search(self):
        dic_pipeline = dict(self.pipelines)
        models = ['GradientBoostingClassifier', 'ExtraTreesClassifier',
                  'RandomForestClassifier', 'DecisionTreeClassifier',
                  'LinearDiscriminantAnalysis', 'SVC', 'KNeighborsClassifier',
                  'LogisticRegression']

        models = ['GradientBoostingClassifier']
        report = []
        grid_search = {}

        self.evaluator.split_data()
        for m in models:
            pipeline = dic_pipeline[m]
            parameters = self.get_params(m, 'grid')

            grid_search_t = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

            print("Performing grid search...", m)
            try:
                start = time()
                grid_search_t.fit(self.evaluator.X_train, self.evaluator.y_train)
                end = time()

                dict_report = OrderedDict()
                dict_report['name'] = m
                dict_report['best_score'] = round(grid_search_t.best_score_, 3)


                model_t = grid_search_t.best_estimator_
                y_pred = model_t.predict(self.evaluator.X_test)
                y_real = self.evaluator.y_test.values
                dict_report['mean_error'] = str(round(tools.mean_error_localization(y_pred, y_real), 3))+'m'


                dict_report['time'] = str(round((end-start)/60.0, 3))+'min'
                dict_report.update(grid_search_t.best_params_)
        #         dict_report['best_params'] = grid_search.best_params_

                report.append(dict_report)
                grid_search[m] = grid_search_t
        #         print("done in %0.3fs" % (t)
        #         print()

                print("Best score: %0.3f" % grid_search_t.best_score_)
        #         print("Best parameters: ", grid)
            except:
                print("Unexpected error:", sys.exc_info()[0])


        score_r, full_r = self.make_report(report)
        self.score_report = score_r
        self.full_report = full_r
        self.search = grid_search
        best_model = self.score_report['Model'].head(1).values[0]
        self.best_search = self.search[best_model]
        self.best_model = self.best_search.best_estimator_


    def improve_random_search(self):
        dic_pipeline = dict(self.pipelines)
        models = ['GradientBoostingClassifier', 'ExtraTreesClassifier',
                  'RandomForestClassifier', 'DecisionTreeClassifier',
                  'LinearDiscriminantAnalysis', 'SVC', 'KNeighborsClassifier',
                  'LogisticRegression']

        models = ['GradientBoostingClassifier', 'SVC']
        report = []
        random_search = {}

        self.evaluator.split_data()
        for m in models:
            pipeline = dic_pipeline[m]
            parameters = self.get_params(m, 'random')

            # random_search_t = RandomizedSearchCV(pipeline, parameters, n_iter=1000, n_jobs=-1, verbose=1)
            random_search_t = RandomizedSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

            print("Performing grid search...", m)
            # try:
            start = time()
            random_search_t.fit(self.evaluator.X_train, self.evaluator.y_train)
            end = time()

            dict_report = OrderedDict()
            dict_report['name'] = m
            dict_report['best_score'] = round(random_search_t.best_score_, 3)


            model_t = random_search_t.best_estimator_
            y_pred = model_t.predict(self.evaluator.X_test)
            y_real = self.evaluator.y_test.values
            dict_report['mean_error'] = str(round(tools.mean_error_localization(y_pred, y_real), 3))+'m'


            dict_report['time'] = str(round((end-start)/60.0, 3))+'min'
            dict_report.update(random_search_t.best_params_)
            #         dict_report['best_params'] = random_search.best_params_

            report.append(dict_report)
            random_search[m] = random_search_t
            #         print("done in %0.3fs" % (t)
            #         print()

            print("Best score: %0.3f" % random_search_t.best_score_)
            #         print("Best parameters: ", grid)
            # except:
            #     print("Unexpected error:", sys.exc_info()[0])
            #     continue


        score_r, full_r = self.make_report(report)
        self.score_report = score_r
        self.full_report = full_r
        self.search = random_search
        best_model = self.score_report['Model'].head(1).values[0]
        self.best_search = self.search[best_model]
        self.best_model = self.best_search.best_estimator_

    def make_report(self, report):
        score_report = []
        full_report = []

        for r in report:
            full_report.append(pd.DataFrame(list(r.items()), columns=['Topic', "Value"]))
            score_report.append([r['name'], r['best_score']])

        score_report = pd.DataFrame(score_report, columns=['Model', "Score"])
        score_report = self.sort_report(score_report)


        return score_report, full_report

    def sort_report(self, report):
        """" Choose the best two algorithms"""

        #sorted_t = sorted(report.items(), key=operator.itemgetter(1))
        report.sort_values(['Score'], ascending=[False], inplace=True)
        #self.bestAlgorithms = sorted_t[-2:]
        # self.report = report.copy()
        return report

    # def chooseTopRanked(self, report):
    #     """" Choose the best two algorithms"""
    #
    #     #sorted_t = sorted(report.items(), key=operator.itemgetter(1))
    #     report.sort_values(['Mean'], ascending=[False], inplace=True)
    #     #self.bestAlgorithms = sorted_t[-2:]
    #     self.bestAlgorithms = report
    #
    #     print(self.bestAlgorithms)

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
        #df = pd.DataFrame.from_dict(self.raw_results)
        #print(df)

        results = self.score_report
        data = []
        N = len(results)
        c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 270, N)]

        for i, d in enumerate(results):
            trace = go.Box(
                y=d['values'],
                name=d['Model'],
                marker=dict(
                    color=c[i],
                ),
                boxmean='sd'
            )
            data.append(trace)

        text_scatter = go.Scatter(
                x=[d['Model'] for d in results],
                y=[d['Score'] for d in results],
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
                    x=results[0]['Model'],
                    y=results[0]['Score'],
                    xref='x',
                    yref='y',
                    text='Best model',
                    showarrow=True,
                    arrowhead=7,
                    ax=0,
                    ay=-40
                ),
                dict(
                    x=results[-1]['Model'],
                    y=results[-1]['Score'],
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

        return self.plot_to_html(fig)

    def save_plot(self, path):
        with open(path, "w") as plot:
            plot.write(self.plot_html)

    def save_full_report(self, path):

        for index, elem in enumerate(self.full_report):
            self.elem.to_csv(path+str(index), index=False)


    def save_score_report(self, path):

        self.score_report.to_csv(path, index=False)

    class CustomFeature(TransformerMixin):
        """ A custome class for modeling """

        def transform(self, X, **transform_params):
            #X = pd.DataFrame(X)
            return X

        def fit(self, X, y=None, **fit_params):
            return self
