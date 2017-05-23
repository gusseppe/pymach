#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module provides ideas for improving some machine learning algorithms.

"""
from __future__ import print_function
import warnings
import pandas as pd
import plotly.graph_objs as go
warnings.filterwarnings("ignore", category=DeprecationWarning)

from collections import OrderedDict
from time import time
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.grid_search import GridSearchCV



class Improve():
    """ A class for improving """

    bestConfiguration = None


    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.pipeline = evaluator.buildPipelines() 
        self.gridsearch = None
        self.score_report = None
        self.full_report = None

    def pipeline(self):

        self.improve_pipelines()

        return self

    def gradientboosting_param(self):

        parameters = {
            'featurer__extraTC__n_estimators':  [10, 16, 32],
            'featurer__extraTC__criterion': ['gini','entropy'],
            'featurer__extraTC__n_jobs': [-1],
            'featurer__pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
            'featurer__pca__whiten': [True],
            'GradientBoostingClassifier__n_estimators': [100, 150, 200],
            'GradientBoostingClassifier__learning_rate': [0.1, 0.2, 0.4, 0.8, 1.0]    
        }
    
        return parameters
    

    def extratrees_param(self):
        parameters = { 
            'featurer__extraTC__n_estimators':  [10, 16, 32],
            'featurer__extraTC__criterion': ['gini','entropy'],
            'featurer__extraTC__n_jobs': [-1],
            'featurer__pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
            'featurer__pca__whiten': [True],
            'ExtraTreesClassifier__n_estimators': [100, 150, 200],
            'ExtraTreesClassifier__criterion': ['gini','entropy']    
        }
        
        return parameters


    def randomforest_param(self):
        parameters = { 
            'featurer__extraTC__n_estimators':  [10, 16, 32],
            'featurer__extraTC__criterion': ['gini','entropy'],
            'featurer__extraTC__n_jobs': [-1],
            'featurer__pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
            'featurer__pca__whiten': [True],
            'RandomForestClassifier__n_estimators': [100, 150, 200],
            'RandomForestClassifier__criterion': ['gini','entropy']    
        }
        return parameters

    def decisiontree_param(self):
        parameters = { 
            'featurer__extraTC__n_estimators':  [10, 16, 32],
            'featurer__extraTC__criterion': ['gini','entropy'],
            'featurer__extraTC__n_jobs': [-1],
            'featurer__pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
            'featurer__pca__whiten': [True],
            'DecisionTreeClassifier__max_features': ['sqrt','log2', None],
            'DecisionTreeClassifier__criterion': ['gini','entropy']    
        }
        return parameters

    def get_params(self, model):
        if model == 'GradientBoostingClassifier':
            return self.gradientboosting_param()
        elif model == 'ExtraTreesClassifier':
            return self.extratrees_param()
        elif model == 'RandomForestClassifier':
            return self.randomforest_param()
        elif model == 'DecisionTreeClassifier':
            return self.decisiontree_param() 
        
        return None

    def improve_pipelines(self):
        dic_pipeline = dict(self.pipeline)
        models = ['GradientBoostingClassifier', 'ExtraTreesClassifier',
                  'RandomForestClassifier', 'DecisionTreeClassifier']
        report = []
        for m in models:        
            pipeline = dic_pipeline[m]
            parameters = self.get_params(m)

            grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

            print("Performing grid search...", m)
            start = time()
            grid_search.fit(self.evaluator.definer.X, self.evaluator.definer.y)
            end = time()
            
            dict_report = OrderedDict()
            dict_report['name'] = m
            dict_report['best_score'] = round(grid_search.best_score_, 3)
            dict_report['time'] = round((end-start)/60.0, 3)
            dict_report.update(grid_search.best_params_)
    #         dict_report['best_params'] = grid_search.best_params_
                          
            report.append(dict_report)
    #         print("done in %0.3fs" % (t)
    #         print()

            print("Best score: %0.3f" % grid_search.best_score_)
    #         print("Best parameters: ", grid)
        
        score_r, full_r = self.make_report(report)
        self.score_report = score_r
        self.full_report = full_r

        # return report

    def make_report(self, report):
        score_report = [] 
        full_report = []

        for r in report:
            full_report.append(pd.DataFrame(list(r.items()), columns=['Topic', "Value"]))
            score_report.append([r['name'], r['best_score']])
        
        score_report = pd.DataFrame(score_report, columns=['Model', "Score"])
                                 

        return score_report, full_report

    def chooseTopRanked(self, report):
        """" Choose the best two algorithms"""

        #sorted_t = sorted(report.items(), key=operator.itemgetter(1))
        report.sort_values(['Mean'], ascending=[False], inplace=True)
        #Evaluate.bestAlgorithms = sorted_t[-2:]
        Evaluate.bestAlgorithms = report

        print(Evaluate.bestAlgorithms)

    def plot_to_html(self, fig):
        plotly_html_div, plotdivid, width, height = _plot_html(
                figure_or_data=fig, 
                config="", 
                validate=True,
                default_width='75%', 
                default_height="100%", 
                global_requirejs=False)

        return plotly_html_div

    def plot_models(self):
        """" Plot the algorithms by using box plots"""
        #df = pd.DataFrame.from_dict(Evaluate.raw_results)
        #print(df)

        results = Evaluate.raw_results
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
        return self.plot_to_html(fig)

    class CustomFeature(TransformerMixin):
        """ A custome class for modeling """

        def transform(self, X, **transform_params):
            #X = pd.DataFrame(X)
            return X

        def fit(self, X, y=None, **fit_params):
            return self
