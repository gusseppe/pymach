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

__all__ = [
    'read', 'description', 'classBalance', 'hist', 'density']


class Analyze():
    """ A class for data analysis """

    data = None

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

    def pipeline(self):

        analyzers = []
        analyzers.append(self.read)
        analyzers.append(self.hist)
        analyzers.append(self.density)
        analyzers.append(self.corr)
        analyzers.append(self.scatter)

        [m() for m in analyzers]


    def read(self):
        """Read the dataset.

        Returns
        -------
        out : ndarray

        """
        data = pd.read_csv(self.nameData)
        Analyze.data = data.ix[:, data.columns != self.className]

        return Analyze.data

    def description(self):
        """Shows a basic data description .

        Returns
        -------
        out : ndarray

        """
        return Analyze.data.describe()

    def classBalance(self):
        """Shows how balanced the class values are.

        Returns
        -------
        out : pandas.core.series.Series
        Serie showing the count of classes.

        """
        return Analyze.data.groupby(self.className).size()

    def hist(self):
        Analyze.data.hist(color=[(0.196, 0.694, 0.823)]) 
        plt.show()

    def density(self):
        #Analyze.data.plot(color=[(0.196, 0.694, 0.823)], kind='density', 
                #subplots=True, layout=(3,3), sharex=False, figsize = (10, 10)) 
        Analyze.data.plot(kind='density', 
                subplots=True, layout=(3,3), sharex=False) 
        plt.show()

    def corr(self):
        corr = Analyze.data.corr()
        fig, ax = plt.subplots()
        bar = ax.matshow(corr, vmin=-1, vmax=1)
        fig.colorbar(bar)
        plt.xticks(range(len(corr.columns)), corr.columns);
        plt.yticks(range(len(corr.columns)), corr.columns);
        plt.show()

    def scatter(self):
        scatter_matrix(Analyze.data)
        plt.show()
