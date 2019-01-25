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
import time
import pickle

import json
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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Clasification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
#Ensembles algorithms
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Regression algorithms
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
#Ensembles algorithms
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


class Evaluate:
    """ A class for resampling and evaluation """

    def __init__(self, definer=None, preparer=None, selector=None):
        self.definer = definer
        self.preparer = preparer
        self.selector = selector
        if definer is not None:
            self.problem_type = definer.problem_type
        self.plot_html = None

        self.report = None
        self.raw_report = None
        self.best_pipelines = None
        self.pipelines = None
        self.estimators = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None

        self.metrics = dict()
        self.feature_importance = dict()

        self.test_size = 0.3
        self.num_folds = 10
        self.seed = 7

    def pipeline(self, list_models):

        self.build_pipelines(list_models)
        self.split_data(self.test_size, self.seed)
        self.evaluate_pipelines()
        self.set_best_pipelines()
        # self.plot_metrics('AdaBoostClassifier')

        #[m() for m in evaluators]
        return self

    def set_models(self, list_models=None):

        models = []
        rs = 1

        if self.problem_type == "classification":

            # Ensemble Methods
            if 'AdaBoostClassifier' in list_models:
                models.append( ('AdaBoostClassifier', AdaBoostClassifier(random_state=rs)) )
            if 'GradientBoostingClassifier' in list_models:
                models.append( ('GradientBoostingClassifier', GradientBoostingClassifier(random_state=rs)) )
            if 'BaggingClassifier' in list_models:
                models.append( ('BaggingClassifier', BaggingClassifier(random_state=rs)))
            if 'RandomForestClassifier' in list_models:
                models.append( ('RandomForestClassifier', RandomForestClassifier(random_state=rs)) )
            if 'ExtraTreesClassifier' in list_models:
                models.append( ('ExtraTreesClassifier', ExtraTreesClassifier(random_state=rs)) )
            # Non linear Methods
            if 'KNeighborsClassifier' in list_models:
                models.append( ('KNeighborsClassifier', KNeighborsClassifier()) )
            if 'DecisionTreeClassifier' in list_models:
                models.append( ('DecisionTreeClassifier', DecisionTreeClassifier(random_state=rs)) )
            if 'MLPClassifier' in list_models:
                models.append( ('MLPClassifier', MLPClassifier(max_iter=1000,random_state=rs)) )
            if 'SVC' in list_models:
                models.append( ('SVC', SVC(random_state=rs)) )
            # Linear Methods
            if 'LinearDiscriminantAnalysis' in list_models:
                models.append( ('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()) )
            if 'GaussianNB' in list_models:
                models.append( ('GaussianNB', GaussianNB()) )
            if 'LogisticRegression' in list_models:
                models.append( ('LogisticRegression', LogisticRegression()) )
            if 'XGBoostClassifier' in list_models:
                models.append( ('XGBoostClassifier', XGBClassifier()) )
            if 'LGBMClassifier' in list_models:
                models.append( ('LGBMClassifier', LGBMClassifier()) )
            # Voting
            estimators = list()
            estimators.append( ("Voting_GradientBoostingClassifier", GradientBoostingClassifier(random_state=rs)) )
            estimators.append( ("Voting_ExtraTreesClassifier", ExtraTreesClassifier(random_state=rs)) )
            voting = VotingClassifier(estimators)
            if 'VotingClassifier' in list_models:
                models.append( ('VotingClassifier', voting) )

        elif self.problem_type == "regression":
            # Ensemble Methods
            if 'AdaBoostRegressor' in list_models:
                models.append( ('AdaBoostRegressor', AdaBoostRegressor(random_state=rs)))
            if 'GradientBoostingRegressor' in list_models:
                models.append( ('GradientBoostingRegressor', GradientBoostingRegressor(random_state=rs)) )
            if 'BaggingRegressor' in list_models:
                models.append( ('BaggingRegressor', BaggingRegressor(random_state=rs)))
            if 'RandomForestRegressor' in list_models:
                models.append( ('RandomForestRegressor',RandomForestRegressor(random_state=rs))  )
            if 'ExtraTreesRegressor' in list_models:
                models.append( ('ExtraTreesRegressor', ExtraTreesRegressor(random_state=rs)) )
            # Non linear Methods
            if 'KNeighborsRegressor' in list_models:
                models.append( ('KNeighborsRegressor', KNeighborsRegressor()) )
            if 'DecisionTreeRegressor' in list_models:
                models.append( ('DecisionTreeRegressor', DecisionTreeRegressor(random_state=rs)) )
            if 'MLPRegressor' in list_models:
                models.append( ('MLPRegressor', MLPRegressor(max_iter=1000, random_state=rs)) )
            if 'SVR' in list_models:
                models.append( ('SVR', SVR()) )
            # Linear Methods
            if 'LinearRegression' in list_models:
                models.append( ('LinearRegression', LinearRegression()) )
            if 'BayesianRidge' in list_models:
                models.append( ('BayesianRidge', BayesianRidge()) )
            if 'XGBoostRegressor' in list_models:
                models.append( ('XGBoostRegressor', XGBRegressor()) )
            if 'LGBMRegressor' in list_models:
                models.append( ('LGBMRegressor', LGBMRegressor()) )

        return models

    def split_data(self, test_size=0.30, seed=7):
        """ Need to fill """

        X_train, X_test, y_train, y_test =  train_test_split(
                self.definer.X, self.definer.y, test_size=test_size, random_state=seed)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # return X_train, X_test, y_train, y_test

    def build_pipelines(self, list_models=None):
        pipelines = []
        models = self.set_models(list_models)

        if self.definer.n_features > 20:
            for m in models:
                pipelines.append((m[0],
                    Pipeline([
                        #('preparer', FunctionTransformer(self.preparer)),
                        ('preparer', self.preparer),
                        ('selector', self.selector),
                        m,
                    ])
                ))
        else:
            for m in models:
                pipelines.append((m[0],
                      Pipeline([
                          #('preparer', FunctionTransformer(self.preparer)),
                          ('preparer', self.preparer),
                          # ('selector', self.selector),
                          m,
                      ])
                 ))


        self.pipelines = pipelines

        return pipelines

    def evaluate_pipelines(self, ax=None):

        test_size = self.test_size
        num_folds = self.num_folds
        seed = self.seed
        if self.definer.problem_type == 'classification':
            scoring = 'accuracy'
        else:
            scoring = 'r2'

        #pipelines = self.build_pipelines(self.set_models())
        #pipelines = self.pipelines


        #self.report = {}
        #report_element = {}
        self.report = [["Model", "Mean", "STD", "Time"]]
        results = []
        names = []
        grid_search = dict()

        for name, model in self.pipelines:
            print("Modeling...", name)

            kfold = KFold(n_splits=num_folds, random_state=seed)
            start = time.time()
            # cv_results = cross_val_score(model, self.X_train, self.y_train, cv=kfold, \
            #         scoring=scoring)

            params = dict()
            # name = 'LogisticRegression'
            for k, v in model.get_params().items():
                # params[name+'__'+k] = [v]
                params[k] = [v]
            grid_search_t = GridSearchCV(model, params, n_jobs=-1,
                             verbose=1, cv=kfold, return_train_score=True,
                                         scoring=scoring)
            grid_search_t.fit(self.X_train, self.y_train)

            end = time.time()
            duration = end - start


            # save the model to disk
            #filename = name+'.ml'
            #pickle.dump(model, open('./models/'+filename, 'wb'))
            # print(cv_results)

            #results.append(cv_results)

            mean = grid_search_t.cv_results_['mean_test_score'][0]
            std = grid_search_t.cv_results_['std_test_score'][0]
            # mean = cv_results.mean()
            # std = cv_results.std()
            cv_results = []
            for i in range(num_folds):
                name_t = 'split' + str(i) + '_test_score'
                cv_results.append(grid_search_t.cv_results_[name_t][0])

            d = {'name': name, 'values': cv_results, 'mean': round(mean, 3),
                 'std': round(std, 3)}
            results.append(d)
            grid_search[name] = grid_search_t.best_estimator_
            #results['result'] = cv_results
            #names.append(name)
            #report_element[name] = {'mean':mean, 'std':std}
            #self.report.update(report_element)

            #report_print = "Model: {}, mean: {}, std: {}".format(name,
                    #mean, std)
            self.report.append([name, round(mean, 3), round(std, 3),
                                round(duration, 3)])
            print("Score ", mean)
            print("---------------------")
            #print(report_print)

        self.raw_report = sorted(results, key=lambda k: k['mean'], reverse=True)
        self.estimators = grid_search
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

    def get_metrics(self):
        if self.problem_type == 'classification':
            models = self.estimators.keys()
            for name_model in models:

                metric_model = dict()
                estimator = self.estimators[name_model]
                y_pred = estimator.predict(self.X_test.values)


                # print(f'The accuracy of the {name_model} is:', accuracy_score(y_pred, self.y_test))
                cm = confusion_matrix(list(self.y_test.reset_index(drop=True)), list(y_pred))
                cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

                metric_model['confusion_matrix_normalized'] = cm_n.tolist()
                metric_model['confusion_matrix'] = cm.tolist()
                metric_model['accuracy'] = accuracy_score(y_pred, self.y_test)
                metric_model['estimator_classes'] = estimator.classes_.tolist()
                self.metrics[name_model] = metric_model
        # print(self.metrics)

    def get_feature_importance(self):
        non_tree_based_models = ['KNeighborsClassifier', 'MLPClassifier', 'SVC',
                                 'LinearDiscriminantAnalysis', 'GaussianNB',
                                 'LogisticRegression', 'KNeighborsRegressor',
                                 'MLPRegressor', 'SVR', 'LinearRegression',
                                 'BayesianRidge']

        models = self.estimators.keys()
        if self.problem_type == 'classification':
            for name_model in models:

                feature_imp = {'feature': [], 'importance':[]}
                if name_model in non_tree_based_models:
                    # estimator = self.estimators[name_model]
                    # y_pred = estimator.predict(self.X_test.values)

                    kbest = SelectKBest(score_func=chi2, k=self.X_train.shape[1])
                    kbest = kbest.fit(self.X_train, self.y_train)

                    print(kbest.scores_)
                    feature_importance = kbest.scores_
                    feature_names = list(self.X_train.columns)
                    for score, name in sorted(zip(feature_importance, feature_names), reverse=True):
                        feature_imp['feature'].append(name)
                        feature_imp['importance'].append(score)

                    df_fi =  pd.DataFrame(feature_imp)
                    df_fi['importance'] = df_fi['importance'] / df_fi['importance'].sum()
                    self.feature_importance[name_model] = df_fi
                else:
                    # Tree based models
                    estimator = self.estimators[name_model].named_steps[name_model]
                    if not hasattr(estimator, 'feature_importances_'):
                        feature_importance = np.mean([
                            tree.feature_importances_ for tree in estimator.estimators_], axis=0)
                        feature_names = list(self.X_train.columns)
                        for score, name in sorted(zip(feature_importance, feature_names), reverse=True):
                            feature_imp['feature'].append(name)
                            feature_imp['importance'].append(score)

                        self.feature_importance[name_model] = pd.DataFrame(feature_imp)
                    else:
                        feature_importance = estimator.feature_importances_
                        feature_names = list(self.X_train.columns)
                        for score, name in sorted(zip(feature_importance, feature_names), reverse=True):
                            feature_imp['feature'].append(name)
                            feature_imp['importance'].append(score)

                        df_fi =  pd.DataFrame(feature_imp)
                        if name_model == 'LGBMClassifier':
                            df_fi['importance'] = df_fi['importance'] / df_fi['importance'].sum()

                        self.feature_importance[name_model] = pd.DataFrame(df_fi)


        else:
            # Here fill with regression
            self.feature_importance = None
        # print(self.metrics)

    def plot_feature_importance(self, name_model):
        if self.problem_type == 'classification':
            feature_importance = self.feature_importance[name_model]

            sorted_idx = np.argsort(feature_importance['importance'].values)

            x = feature_importance['importance'].values[sorted_idx]
            y = feature_importance['feature'].values[sorted_idx]
            text = [str(round(e, 2)*100) +'%' for e in x]
            trace = go.Bar(x=x,
                           y=y,
                           orientation = 'h', text=text,
                           textposition = 'auto',
                           marker=dict(
                            color='rgb(58,200,225)',
                            line=dict(
                                color='rgb(8,48,107)',
                                width=1.5),
                            ),
                            opacity=0.6)

            layout = go.Layout(xaxis=dict(title='Importance'),
                               yaxis=dict(title='Features'),
                               title=f'Feature Importance - {name_model}',
                               annotations=[
                                   dict(
                                       x=x[0],
                                       y=y[0],
                                       xref='x',
                                       yref='y',
                                       text='Least important feature',
                                       showarrow=True,
                                       arrowhead=3,
                                       ax=40,
                                       ay=40
                                   ),
                                   dict(
                                       x=x[-1],
                                       y=y[-1],
                                       xref='x',
                                       yref='y',
                                       text='Most important feature',
                                       showarrow=True,
                                       arrowhead=3,
                                       ax=40,
                                       ay=40
                                   )
                               ])
            fig = go.Figure(data=[trace], layout=layout)

        else:
            fig = None

        return fig

    def plot_metrics(self, name_model):
        if self.problem_type == 'classification':
            acc = self.metrics[name_model]['accuracy']
            cm = np.array(self.metrics[name_model]['confusion_matrix_normalized'])
            estimator_classes = self.metrics[name_model]['estimator_classes']

            annot = list(zip(cm.diagonal(), estimator_classes))
            sort_annot = sorted(annot, key=lambda x: x[0])
            trace = go.Heatmap(z=cm,
                               x=estimator_classes,
                               y=estimator_classes,
                               colorscale='Viridis')
            data=[trace]

            layout = go.Layout(xaxis=dict(title='Predicted label'),
                               yaxis=dict(title='True label'),
                               title=f"Confusion Matrix - {name_model} -  Acc: {round(acc, 2)} - Test dataset",
                               annotations=[
                                   dict(
                                       x=sort_annot[0][1],
                                       y=sort_annot[0][1],
                                       xref='x',
                                       yref='y',
                                       text='Least precision',
                                       showarrow=True,
#                                        font=dict(
#                                         family='Courier New, monospace',
# #                                         size=16,
#                                         color='#ffffff'
#                                        ),
                                       arrowhead=3,
#                                        arrowcolor='#ffffff',
                                       ax=30,
                                       ay=30
                                   ),
                                   dict(
                                       x=sort_annot[-1][1],
                                       y=sort_annot[-1][1],
                                       xref='x',
                                       yref='y',
                                       text='Perfect precision',
                                       showarrow=True,
                                       arrowhead=3,
                                       ax=30,
                                       ay=30
                                   )
                               ])

            fig = go.Figure(data=[trace], layout=layout)

        else:
            fig = None

        return fig

    # def plot_metrics(self, name_model):
    #     if self.problem_type == 'classification':
    #         acc = self.metrics[name_model]['accuracy']
    #         cm = self.metrics[name_model]['confusion_matrix_normalized']
    #         estimator_classes = self.metrics[name_model]['estimator_classes']
    #         cm_df = pd.DataFrame(cm,
    #                              index=estimator_classes,
    #                              columns=estimator_classes)
    #         fig = cm_df.iplot(
    #             kind="heatmap",
    #             asFigure=True,
    #             xTitle="Predicted label",
    #             colorscale='ylorrd',
    #             yTitle="True label",
    #             # title="Confusion Matrix - "+name_model+" - Acc: ",
    #             title=f"Confusion Matrix - {name_model} -  Acc: {round(acc, 2)} - Test dataset",
    #             theme="white")
    #
    #     else:
    #         fig = None
    #
    #     return fig

    def set_best_pipelines(self):
        best_models = list(self.report.Model)[0:2]
        # print('evaluate>>>', best_models)
        best_pipelines = []

        for m in best_models:
            for p in self.pipelines:
                if m == p[0]:
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
            title="Models' performance ranking",
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

        # self.plot_html = self.plot_to_html(fig)
        # return self.plot_html
        return fig

    def save_plot(self, path):
        with open(path, "w") as plot:
            plot.write(self.plot_html)

    def save_report(self, path):
        # with open(path, "w") as plot:
        self.report.to_csv(path, index=False)
        # plot.write(valuate.report.to_csv())

    def save_raw_report(self, path):
        # with open(path, "w") as plot:
        # self.raw_report.to_csv(path, index=False)
        # plot.write(valuate.report.to_csv())
        with open(path, 'w') as file:
            json.dump(self.raw_report, file)

    def save_metrics(self, path):
        # with open(path, "w") as plot:
        # self.raw_report.to_csv(path, index=False)
        # plot.write(valuate.report.to_csv())
        with open(path, 'w') as file:
            json.dump(self.metrics, file)

    def save_feature_importance(self, path):
        jstr = json.dumps(self.feature_importance,
                          default=lambda df: json.loads(df.to_json()))
        result = json.loads(jstr)
        with open(path, 'w') as file:
            json.dump(result, file)

    def save_model(self, model, path):
        # for k, v in self.estimators.items():
            # estimator = self.estimators[name_model]
        with open(path, 'wb') as f:
            pickle.dump(model, f)

            # pickle.dump(model, open(filename, 'wb'))
            # with open(path, "w") as plot:
            #     plot.write(self.plot_html)

    class CustomFeature(TransformerMixin):
        """ A custome class for modeling """

        def transform(self, X, **transform_params):
            #X = pd.DataFrame(X)
            return X

        def fit(self, X, y=None, **fit_params):
            return self
