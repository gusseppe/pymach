#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module provides a few of useful functions (actually, methods)
for describing the dataset which is to be studied.

"""
from __future__ import print_function
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.plotly as py
import cufflinks as cf
from plotly.offline.offline import _plot_html

__all__ = [
    'read', 'description', 'classBalance', 'hist', 'density']


class Analyze():
    """ A class for data analysis """

    def __init__(self, definer):
        """The init class.

        Parameters
        ----------
        typeModel : string
            String that indicates if the model will be train for clasification
            or regression.
        className : string
            String that indicates which column in the dataset is the class.

        """
        self.typeModel = definer.typeModel
        self.typeAlgorithm = definer.typeAlgorithm
        self.className = definer.className
        self.nameData = definer.nameData
        self.data = definer.data

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
        return self.data.groupby(self.className).size()

    #def hist(self, ax=None):
        ##plt.figure(figsize=(10.8, 3.6))
        ##for column in df:
            ##df[column].hist(color=[(0.196, 0.694, 0.823)], ax=ax, align='left', label = 'Frequency bar of subsectors') 
        #self.data.hist(color=[(0.196, 0.694, 0.823)], ax=ax, label='frecuencia') 
        #plt.legend(loc='best')
        #if ax is None:
            #plt.show()

    def init_plot(self):
        cf.set_config_file(offline=True, world_readable=True, 
                theme='pearl')

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

        return self.plot_to_html(fig)

    def boxplot(self):
        fig = self.data.iplot(
                kind="box",
                asFigure=True, 
                xTitle="Features",
                yTitle="Values",
                boxpoints="outliers",
                theme="white")

        return self.plot_to_html(fig)

    def correlation(self):
        corr_data = self.data.corr()
        fig = corr_data.iplot(
                kind="heatmap",
                asFigure=True,
                xTitle="Features",
                yTitle="Features",
                theme="white")

        return self.plot_to_html(fig)

    def scatter(self):
        corr_data = self.data.corr()
        fig = corr_data.iplot(
                kind="scatter",
                asFigure=True,
                #title="Correlation Matrix",
                xTitle="Features",
                yTitle="Features",
                theme="white")

        return self.plot_to_html(fig)

    def plot(self, name):
        if name == "histogram":
            return self.histogram()
        elif name == "box":
            return self.boxplot()
        elif name == "corr":
            return self.correlation()
        elif name == "scatter":
            return self.scatter()

    #def density(self, ax=None):
        ##Analyze.data.plot(color=[(0.196, 0.694, 0.823)], kind='density', 
                ##subplots=True, layout=(3,3), sharex=False, figsize = (10, 10)) 
        #self.data.plot(kind='density', 
                #subplots=True, layout=(3,3), sharex=False, ax=ax) 
        #if ax is None:
            #plt.show()

    #def corr(self, ax=None):
        #corr = self.data.corr()
        #names = list(self.data.columns.values)
        #fig, ax1 = plt.subplots()

        #if ax is not None:
            #bar = ax.matshow(corr, vmin=-1, vmax=1)
        #else:
            #bar = ax1.matshow(corr, vmin=-1, vmax=1)

        #fig.colorbar(bar)
        ##plt.xticks(range(len(corr.columns)), corr.columns)
        ##plt.yticks(range(len(corr.columns)), corr.columns)
        #ax.set_xticks(range(len(corr.columns)))
        #ax.set_yticks(range(len(corr.columns)))
        #ax.set_xticklabels(names)
        #ax.set_yticklabels(names)

        #if ax is None:
            #plt.show()

    #def scatter(self, ax=None):
        #scatter_matrix(self.data, alpha=0.7, figsize=(6, 6), diagonal='kde', ax=ax)
        #if ax is None:
            #plt.show()
        
    #def box(self, ax=None):
        #self.data.plot(kind="box" , subplots=True, layout=(3,3), sharex=False, sharey=False, ax=ax)
        #if ax is None:
            #plt.show()

