#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module provides a few of useful functions (actually, methods)
for describing the dataset which is to be studied.

"""
from __future__ import print_function
import plotly.graph_objs as go
import pandas as pd
import hickle

import core.tools as tools
from plotly.offline.offline import _plot_html
from collections import namedtuple
from sklearn.preprocessing import LabelEncoder

# cf.set_config_file(world_readable=True,theme='pearl')

# __all__ = [
#     'read', 'description', 'classBalance', 'hist', 'density']


class Analyze():
    """ A class for data analysis """

    FigureStruct = namedtuple("FigureStruct", "figure explanation")

    def __init__(self, definer):
        """The init class.

        Parameters
        ----------
        typeModel : string
            String that indicates if the model will be trained for clasification
            or regression.
        className : string
            String that indicates which column in the dataset is the class.

        """
        self.problem_type = definer.metadata['problem_type']
        # self.infer_algorithm = definer.infer_algorithm
        self.response = definer.response
        # self.data_path = definer.data_path
        self.data = definer.data
        self.plot_html = None
        self.fig = None

    def pipeline(self):

        analyzers = []
        analyzers.append(self.hist)
        analyzers.append(self.density)
        analyzers.append(self.corr)
        analyzers.append(self.scatter)

        [m() for m in analyzers]

        return self

    def description(self):
        """Shows a basic data description .

        Returns
        -------
        out : ndarray

        """
        #return self.data.describe()
        return pd.DataFrame(self.data.describe())

    def classBalance(self):
        """Shows how balanced the class values are.

        Returns
        -------
        out : pandas.core.series.Series
        Serie showing the count of classes.

        """
        return self.data.groupby(self.response).size()

    #def init_plot(self):
        #cf.set_config_file(offline=True, world_readable=True,
                #theme='pearl')

    def plot_to_html(self, fig):
        plotly_html_div, plotdivid, width, height = _plot_html(
                figure_or_data=fig,
                config="",
                validate=True,
                default_width='90%',
                default_height="100%",
                global_requirejs=False)

        return plotly_html_div

    def histogram(self):
        fig = self.data.iplot(
                kind="histogram",
                asFigure=True,
                xTitle="Features",
                yTitle="Frequency",
                theme="white")

        # self.plot_html = self.plot_to_html(fig)
        self.fig = fig
        return fig

    def boxplot(self):
        fig = self.data.iplot(
                kind="box",
                asFigure=True,
                xTitle="Features",
                yTitle="Values",
                boxpoints="outliers",
                mode='markers',
                text=['Text A']*(len(self.data.columns)-1),
                theme="white")

        # self.plot_html = self.plot_to_html(fig)
        self.fig = fig
        return fig

    def correlation(self):
        corr_data = self.data.corr()
        fig = corr_data.iplot(
                kind="heatmap",
                asFigure=True,
                xTitle="Features",
                colorscale='ylorrd',
                yTitle="Features",
                theme="white")

        # self.plot_html = self.plot_to_html(fig)
        self.fig = fig
        return fig

    def scatter(self):
        columns = [x for x in self.data.columns if x != self.response]
        columns = columns[:4]

        fig = self.data[columns].scatter_matrix(
            asFigure=True,
            title="Scatter Matrix",
            xTitle="Features",
            yTitle="Features",
            mode='markers',
            columns=columns,
            subplots=True,
            theme="white")
        # fig = ff.create_scatterplotmatrix(
        #         self.data,
        #         diag='box',
        #         # height=800,
        #         width=1000,
        #         index=self.response)

        # self.plot_html = self.plot_to_html(fig)
        self.fig = fig
        return fig

    def parallel(self):
        columns = [x for x in self.data.columns if x != self.response]

        min_range = self.data[columns].min().values.min()
        max_range = self.data[columns].max().values.max()
        dimensions = list([dict(range=[min_range, max_range], label=col,
                       values=self.data[col]) for col in columns])

        encoder = LabelEncoder()
        color_labels = encoder.fit_transform(self.data[self.response])
        data = [
            go.Parcoords(
                line=dict(color=color_labels,
                           colorscale = [[0,'#D7C16B'],[0.5,'#23D8C3'],[1,'#F3F10F']]),
                dimensions=dimensions
            )
        ]
        # layout = go.Layout(
        #     plot_bgcolor = '#E5E5E5',
        #     paper_bgcolor = '#E5E5E5'
        # )
        fig = go.Figure(data=data)

        # self.plot_html = self.plot_to_html(fig)
        self.fig = fig
        return fig

    def plot(self, name):
        if name == "hist":
            return self.histogram()
        elif name == "box":
            return self.boxplot()
        elif name == "corr":
            return self.correlation()
        elif name == "scatter":
            return self.scatter()
        elif name == "parallel":
            return self.parallel()

    def save(self, path):
        # hickle.dump(self.fig, path, mode='w')
        tools.plotlyfig2json(self.fig, path)
        # with shelve.open(path, protocol=3) as file:
        #     file['plot'] = self.fig
        #     plot.write(self.fig)
