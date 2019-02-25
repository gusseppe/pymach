#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module provides ideas for improving some machine learning algorithms.
"""
from __future__ import print_function
from core import tools

import matplotlib.pyplot as plt
import plotly.graph_objs as go
import multiprocessing as mp
import multiprocessing
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import sys

from plotly.offline.offline import _plot_html
from scipy.stats import randint
from scipy.stats import uniform
from scipy.stats import expon
from collections import OrderedDict
from time import time

# from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from core.searchers import GeneticSearchCV, EdasSearch
from core.methods import getModelAccuracy


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


class Improve:
    """ A class for improving """
    bestConfiguration = None

    def __init__(self, evaluator, optimizer, modelos):
        self.evaluator = evaluator
        self.pipelines = evaluator.build_pipelines(modelos)
        self.modelos = modelos
        self.optimizer = optimizer
        self.problem_type = evaluator.problem_type
        self.search = None
        self.score_report = None
        self.full_report = None
        self.best_search = None
        self.best_model = None
        self.cv = 10

    def pipeline(self):
        self.improve_grid_search()
        return self

    ############################# Classification ###################################
    def adaboost_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'AdaBoostClassifier__n_estimators': [50,75,100],
                'AdaBoostClassifier__learning_rate': [0.5,1.0,1.5],
                'AdaBoostClassifier__algorithm' : ['SAMME', 'SAMME.R']
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'AdaBoostClassifier__n_estimators': randint(25,100),
                # 'AdaBoostClassifier__learning_rate': [0.5,1.0,1.5],
                'AdaBoostClassifier__learning_rate': expon(0,1),
                'AdaBoostClassifier__algorithm' : ['SAMME', 'SAMME.R']
            }
        else:
            pass
        return parameters

    def gradientboosting_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'GradientBoostingClassifier__n_estimators': [200, 250],
            'GradientBoostingClassifier__max_depth': [3,6,9],
            'GradientBoostingClassifier__learning_rate': [0.1, 0.2, 0.3]
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'GradientBoostingClassifier__n_estimators': randint(200,250),
            'GradientBoostingClassifier__max_depth': randint(3,9),
            'GradientBoostingClassifier__learning_rate': expon(0,1)
            }
        else:
            pass
        return parameters

    def bagging_paramC(self,method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'BaggingClassifier__n_estimators': [50, 100],
            'BaggingClassifier__warm_start': [True,False]
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'BaggingClassifier__n_estimators': randint(50,100),
            'BaggingClassifier__warm_start': [True,False]
            }
        else:
            pass
        return parameters

    def randomforest_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'RandomForestClassifier__n_estimators': [10, 15],
            'RandomForestClassifier__criterion': ['gini', 'entropy'],
            'RandomForestClassifier__warm_start': [True,False]
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'RandomForestClassifier__n_estimators': randint(10,15),
            'RandomForestClassifier__criterion': ['gini', 'entropy'],
            'RandomForestClassifier__warm_start': [True,False]
            }
        else:
            pass
        return parameters

    def extratrees_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'ExtraTreesClassifier__n_estimators': [10, 15, 20],
            'ExtraTreesClassifier__criterion': ['gini', 'entropy'],
            'ExtraTreesClassifier__warm_start': [True, False]
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'ExtraTreesClassifier__n_estimators': randint(10,20),
            'ExtraTreesClassifier__criterion': ['gini', 'entropy'],
            'ExtraTreesClassifier__warm_start': [True, False]
            }
        else:
            pass
        return parameters

    def knn_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'KNeighborsClassifier__n_neighbors': [5,10,15],
            'KNeighborsClassifier__weights': ['uniform','distance'],
            'KNeighborsClassifier__algorithm': ['ball_tree','kd_tree','brute']
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'KNeighborsClassifier__n_neighbors': randint(5,15),
            'KNeighborsClassifier__weights': ['uniform','distance'],
            'KNeighborsClassifier__algorithm': ['ball_tree','kd_tree','brute']
            }
        else:
            pass
        return parameters

    def decisiontree_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'DecisionTreeClassifier__criterion': ['gini','entropy'],
                'DecisionTreeClassifier__splitter': ['best','random'],
                'DecisionTreeClassifier__max_features': ['sqrt','log2', None]
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'DecisionTreeClassifier__criterion': ['gini','entropy'],
                'DecisionTreeClassifier__splitter': ['best','random'],
                'DecisionTreeClassifier__max_features': ['sqrt','log2', None]
            }
        else:
            pass
        return parameters

    def mlperceptron_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'MLPClassifier__hidden_layer_sizes': [50,100],
            'MLPClassifier__activation': ['identity', 'logistic', 'tanh', 'relu']
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'MLPClassifier__hidden_layer_sizes': randint(50,100),
            'MLPClassifier__activation': ['identity', 'logistic', 'tanh', 'relu']
            }
        else:
            pass
        return parameters

    def svc_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'SVC__kernel': ['linear','poly', 'rbf','sigmoid'],
            'SVC__C': [1, 10, 100],
            'SVC__decision_function_shape': ['ovo','ovr']
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'SVC__kernel': ['linear','poly', 'rbf','sigmoid'],
            'SVC__C': randint(1,100),
            'SVC__decision_function_shape': ['ovo','ovr']
            }
        else:
            pass
        return parameters

    def lda_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'LinearDiscriminantAnalysis__solver': ['svd']
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'LinearDiscriminantAnalysis__solver': ['svd']
            }
        else:
            pass
        return parameters

    '''
    def voting_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'VotingClassifier__voting': ['hard', 'soft']
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'VotingClassifier__voting': ['hard', 'soft']
            }
        else:
            pass
            return parameters
    '''

    def logistic_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'LogisticRegression__penalty': ['l2'],
                # 'LogisticRegression__solver': ['newton-cg','lbfgs','liblinear','sag'],
                'LogisticRegression__solver': ['newton-cg','lbfgs', 'sag'],
                'LogisticRegression__warm_start': [True,False]
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'LogisticRegression__penalty': ['l2'],
                # 'LogisticRegression__solver': ['newton-cg','lbfgs','liblinear','sag'],
                'LogisticRegression__solver': ['newton-cg','lbfgs', 'sag'],
                'LogisticRegression__warm_start': [True,False]
            }
        else:
            pass
        return parameters

    def naivebayes_paramC(self, method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False]
                # 'GaussianNB__priors': [None]
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False]
                # 'GaussianNB__priors': [None]
            }
        else:
            pass
        return parameters


    ############################# Regression ###################################
    def adaboost_paramR(self,method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'AdaBoostRegressor__n_estimators': [50,75,100],
            'AdaBoostRegressor__learning_rate': [0.5,1.0,1.5,2.0],
            'AdaBoostRegressor__loss' : ['linear', 'square', 'exponential']
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],# n_components=3 must be stricly less than n_features=3 with svd_solver='arpack'
            'selector__pca__whiten': [True,False],
            'AdaBoostRegressor__n_estimators': randint(50,100),
            'AdaBoostRegressor__learning_rate': expon(0,5),
            'AdaBoostRegressor__loss' : ['linear', 'square', 'exponential']
            }
        else:
            pass
        return parameters

    def gradientboosting_paramR(self,method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'GradientBoostingRegressor__loss': ['ls','lad','huber','quantile'],
            'GradientBoostingRegressor__n_estimators': [100, 200, 250],
            'GradientBoostingRegressor__max_depth': [3,6,9],
            'GradientBoostingRegressor__learning_rate': [0.1, 0.2, 0.3]
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'GradientBoostingRegressor__loss': ['ls','lad','huber','quantile'],
            'GradientBoostingRegressor__n_estimators': randint(200,250),
            'GradientBoostingRegressor__max_depth': randint(3,9),
            'GradientBoostingRegressor__learning_rate': expon(0,1)
            }
        else:
            pass
        return parameters

    def bagging_paramR(self,method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'BaggingRegressor__n_estimators': [50, 100],
            'BaggingRegressor__warm_start': [True,False]
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'BaggingRegressor__n_estimators': randint(50,100),
            'BaggingRegressor__warm_start': [True,False]
            }
        else:
            pass
        return parameters

    def randomforest_paramR(self,method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'RandomForestRegressor__n_estimators': [10, 15],
            'RandomForestRegressor__criterion': ['mse', 'mae'],
            'RandomForestRegressor__warm_start': [True,False]
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'RandomForestRegressor__n_estimators': randint(10,15),
            'RandomForestRegressor__criterion': ['mse', 'mae'],
            'RandomForestRegressor__warm_start': [True,False]
            }
        else:
            pass
        return parameters

    def extratrees_paramR(self,method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'ExtraTreesRegressor__n_estimators': [10, 15, 20],
            'ExtraTreesRegressor__criterion': ['mse', 'mae'],
            'ExtraTreesRegressor__warm_start': [True, False]
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'ExtraTreesRegressor__n_estimators': randint(10,20),
            'ExtraTreesRegressor__criterion': ['mse', 'mae'],
            'ExtraTreesRegressor__warm_start': [True, False]
            }
        else:
            pass
        return parameters

    def knn_paramR(self,method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'KNeighborsRegressor__n_neighbors': [5,10,15],
                'KNeighborsRegressor__weights': ['uniform','distance'],
                'KNeighborsRegressor__algorithm': ['ball_tree','kd_tree','brute']
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'KNeighborsRegressor__n_neighbors': randint(5,15),
                'KNeighborsRegressor__weights': ['uniform','distance'],
                'KNeighborsRegressor__algorithm': ['ball_tree','kd_tree','brute']
            }
        else:
            pass
        return parameters

    def decisiontree_paramR(self,method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'DecisionTreeRegressor__criterion': ['mse','friedman_mse','mae'],
                'DecisionTreeRegressor__splitter': ['best','random'],
                'DecisionTreeRegressor__max_features': ['sqrt','log2', None]
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'DecisionTreeRegressor__criterion': ['mse','friedman_mse','mae'],
                'DecisionTreeRegressor__splitter': ['best','random'],
                'DecisionTreeRegressor__max_features': ['sqrt','log2', None]
            }
        else:
            pass
        return parameters

    def mlperceptron_paramR(self,method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'MLPRegressor__hidden_layer_sizes': [50,100],
            'MLPRegressor__activation': ['identity', 'logistic', 'tanh', 'relu']
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
            'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
            'selector__pca__whiten': [True,False],
            'MLPRegressor__hidden_layer_sizes': randint(50,100),
            'MLPRegressor__activation': ['identity', 'logistic', 'tanh', 'relu']
            }
        else:
            pass
        return parameters

    def svc_paramR(self,method='GridSearchCV'):
        if method == 'GridSearchCV' or method == 'GeneticSearchCV' or method == 'EdasSearch':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'SVR__kernel': ['linear','poly', 'rbf','sigmoid'],
                'SVR__C': [1, 10, 100]
            }
        elif  method == 'RandomizedSearchCV':
            parameters = {
                'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
                'selector__pca__whiten': [True,False],
                'SVR__kernel': ['linear','poly', 'rbf','sigmoid'],
                'SVR__C': randint(1,100)
            }
        else:
            pass
        return parameters

    ############################################################################

    def get_params(self, model, method):
        if self.problem_type == 'Classification':
            if model == 'AdaBoostClassifier':
                return self.adaboost_paramC(method)
            elif model == 'GradientBoostingClassifier':
                return self.gradientboosting_paramC(method)
            elif model == 'BaggingClassifier':
                return self.bagging_paramC(method)
            elif model == 'RandomForestClassifier':
                return self.randomforest_paramC(method)
            elif model == 'ExtraTreesClassifier':
                return self.extratrees_paramC(method)
            elif model == 'KNeighborsClassifier':
                return self.knn_paramC(method)
            elif model == 'DecisionTreeClassifier':
                return self.decisiontree_paramC(method)
            elif model == 'MLPClassifier':
                return self.mlperceptron_paramC(method)
            elif model == 'VotingClassifier':
                return self.voting_paramC(method)
            elif model == 'SVC':
                return self.svc_paramC(method)
            elif model == 'LinearDiscriminantAnalysis':
                return self.lda_paramC(method)
            elif model == 'LogisticRegression':
                return self.logistic_paramC(method)
            elif model == 'GaussianNB':
                return self.naivebayes_paramC(method)
        elif self.problem_type=='Regression':
            if model == 'AdaBoostRegressor':
                return self.adaboost_paramR(method)
            elif model == 'GradientBoostingRegressor':
                return self.gradientboosting_paramR(method)
            elif model == 'BaggingRegressor':
                return self.bagging_paramR(method)
            elif model == 'RandomForestRegressor':
                return self.randomforest_paramR(method)
            elif model == 'ExtraTreesRegressor':
                return self.extratrees_paramR(method)
            elif model == 'KNeighborsRegressor':
                return self.knn_paramR(method)
            elif model == 'DecisionTreeRegressor':
                return self.decisiontree_paramR(method)
            elif model == 'MLPRegressor':
                return self.mlperceptron_paramR(method)
            elif model == 'SVR':
                return self.svc_paramR(method)

        return None

    def evaluate_model(self, pipelines):
        n,m = pipelines
        parameters = self.get_params(n, self.optimizer)
        if self.optimizer == 'GridSearchCV':
            print("Performing GridSearchCV...", n)
            grid_search_t = GridSearchCV(m, parameters, verbose=1)
            grid_search_t.fit(self.evaluator.X_train, self.evaluator.y_train)
            return [grid_search_t.best_score_,grid_search_t.best_params_]
        elif self.optimizer == 'RandomizedSearchCV':
            print("Performing RandomizedSearchCV...", n)
            random_search_t = RandomizedSearchCV(m, parameters, verbose=1)
            random_search_t.fit(self.evaluator.X_train, self.evaluator.y_train)
            return [random_search_t.best_score_,random_search_t.best_params_]
        elif self.optimizer == 'GeneticSearchCV':
            print("Performing GeneticSearchCV...", n)
            genetic_search_t = GeneticSearchCV(m, parameters, scoring=None, cv=KFold(n_splits=5), n_jobs=1, verbose=1, refit=False, population_size=50, gene_mutation_prob=0.10, gene_crossover_prob=0.5, tournament_size=3, generations_number=10)
            genetic_search_t.fit(self.evaluator.X_train, self.evaluator.y_train)
            return [genetic_search_t.best_score_,genetic_search_t.best_params_]
        elif self.optimizer == 'EdasSearch':
            print("Performing EdasSearch...", n)
            eda_search_t = EdasSearch(getModelAccuracy, parameters, m,iterations=2, sample_size=15, select_ratio=0.3, debug=False, n_jobs=1)
            eda_search_t.fit()
            return [eda_search_t.best_score_,eda_search_t.best_params_]

    def improve_grid_search(self):
        self.evaluator.split_data()
        self.report = [["Model", "Best_score", "Parameters"]]
        results = []
        if self.optimizer == 'GridSearchCV' or self.optimizer == 'RandomizedSearchCV':
            num_cores=mp.cpu_count()
            print("****************** num_cores: ",num_cores," *********************")
            pool = mp.Pool(processes=num_cores)
            r = pool.map(self.evaluate_model,self.pipelines)
        elif self.optimizer == 'GeneticSearchCV' or self.optimizer == 'EdasSearch':
            pool = MyPool(1)
            r = pool.map(self.evaluate_model,self.pipelines)

        pool.close()
        pool.join()
        i=0

        for cv_results in r:
            print("Modeling ...", self.modelos[i])
            d = {'name': self.modelos[i], 'best_score': round(cv_results[0], 5), 'parameters': cv_results[1]}
            results.append(d)
            self.report.append([self.modelos[i], round(cv_results[0], 5), cv_results[1]])
            print("Best score: %0.3f" % cv_results[0])
            i = i+1
        print("*****************************************************************")

        self.score_report = sorted(results, key=lambda k: k['best_score'], reverse=True)
        headers = self.report.pop(0)
        df_report = pd.DataFrame(self.report, columns=headers)
        print(df_report)
        self.sort_report(df_report)


    def sort_report(self, report):
        """" Choose the best two algorithms"""
        report.sort_values(['Best_score'], ascending=[False], inplace=True)
        self.report = report.copy()
        #print(self.report)

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
        results = self.score_report
        data = []
        N = len(results)
        c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 270, N)]

        for i, d in enumerate(results):
            trace = go.Box(
                y=d['best_score'],
                name=d['name'],
                marker=dict(
                    color=c[i],
                ),
                boxmean='sd'
            )
            data.append(trace)

        text_scatter = go.Scatter(
                x=[d['name'] for d in results],
                y=[d['best_score'] for d in results],
                name='best score',
                mode='markers',
                text=['Explanation' for _ in results]
        )
        data.append(text_scatter)
        if len(self.score_report) == 1:
            layout = go.Layout(
                #showlegend=False,
                title='Hover over the bars to see the details',
                annotations=[
                    dict(
                        x=results[0]['name'],
                        y=results[0]['best_score'],
                        xref='x',
                        yref='y',
                        text='Best model',
                        showarrow=True,
                        arrowhead=7,
                        ax=0,
                        ay=-40
                    )
                ]
            )
        else:
            layout = go.Layout(
                #showlegend=False,
                title='Hover over the bars to see the details',
                annotations=[
                    dict(
                        x=results[0]['name'],
                        y=results[0]['best_score'],
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
                        y=results[-1]['best_score'],
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
            elem.to_csv(path+'_model'+str(index+1)+'.csv', index=False)

    def save_score_report(self, path):

        self.score_report.to_csv(path+'_score'+'.csv', index=False)
