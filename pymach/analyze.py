#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module provides a few of useful functions (actually, methods)
for describing the dataset which is to be studied.

"""
from __future__ import print_function
# from pandas.tools.plotting import scatter_matrix
# import numpy as np
import tools
# import matplotlib.pyplot as plt
import pandas as pd
# import plotly.plotly as py
import cufflinks as cf # Needed
from plotly.offline.offline import _plot_html
from collections import namedtuple

__all__ = [
    'read', 'description', 'classBalance', 'hist', 'density']


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
        self.problem_type = definer.problem_type
        # self.infer_algorithm = definer.infer_algorithm
        self.response = definer.response
        # self.data_path = definer.data_path
        self.data = definer.data
        self.plot_html = None

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
                default_width='75%',
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

        self.plot_html = self.plot_to_html(fig)
        return self.plot_html

    def boxplot(self):
        fig = self.data.iplot(
                kind="box",
                asFigure=True,
                xTitle="Features",
                yTitle="Values",
                boxpoints="outliers",
                mode='markers',
                text=['Text A', 'Text B', 'Text C'],
                theme="white")

        self.plot_html = self.plot_to_html(fig)
        return self.plot_html

    def correlation(self):
        corr_data = self.data.corr()
        fig = corr_data.iplot(
                kind="heatmap",
                asFigure=True,
                xTitle="Features",
                yTitle="Features",
                theme="white")

        self.plot_html = self.plot_to_html(fig)
        return self.plot_html

    def scatter(self):
        fig = self.data.scatter_matrix(
                asFigure=True,
                # title="Correlation Matrix",
                xTitle="Features",
                yTitle="Features",
                theme="white")

        self.plot_html = self.plot_to_html(fig)
        return self.plot_html

    def plot(self, name):
        if name == "histogram":
            return self.histogram()
        elif name == "box":
            return self.boxplot()
        elif name == "corr":
            return self.correlation()
        elif name == "scatter":
            return self.scatter()

    def save_plot(self, name):
        with open(name, "w") as plot:
            plot.write(self.plot_html)
