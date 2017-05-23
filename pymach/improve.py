#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module provides ideas for improving some machine learning algorithms.

"""
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from time import time
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.grid_search import GridSearchCV



class Improve():
    """ A class for improving """

    bestConfiguration = None


    def __init__(self, evaluator):
        self.pipeline = evaluator.buildPipelines() 
        self.gridsearch = None


    def pipeline(self):

        self.improve()

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

    def adaboost_param(self):
        parameters = {
            'featurer__extraTC__n_estimators':  [10, 16, 32],
            'featurer__extraTC__criterion': ['gini', 'entropy'],
            'featurer__extraTC__n_jobs': [-1],
            'featurer__pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
            'featurer__pca__whiten': [True],
            'AdaBoostClassifier__base_estimator__criterion': ['gini', 'entropy'],
            'AdaBoostClassifier__learning_rate': [0.1, 0.2, 0.4, 0.8, 1.0],
            'AdaBoostClassifier__n_estimators': [50, 100, 150, 200]
        }

        return parameters

    def improve(self):
        dic_pipeline = dict(self.pipeline)
        pipeline = dic_pipeline['GradientBoostingClassifier']
        parameters = self.gradientboosting_param()

        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

        print("Performing grid search...")
        # print("pipeline:", [name for name, _ in pipeline.steps])
        # print("parameters:")
        # print(parameters)
        t0 = time()
        grid_search.fit(self.definer.X, self.definer.y)
        print("done in %0.3fs" % (time() - t0))
        print()

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters: %0.3f" % grid_search.best_params_)

        self.gridsearch = grid_search

        # return grid_search

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
