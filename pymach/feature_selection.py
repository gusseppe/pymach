#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module provides a few of useful functions (actually, methods)
for feature selection the dataset which is to be studied.

"""
from __future__ import print_function
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
import numpy as np
import pandas as pd

__all__ = [
    'read', 'clean', 'reescale', 'standardize', 'normalize', 'binarize']


class FeatureSelection():
    """ A class for feature selection """

    data = None

    def __init__(self, typeModel='clasification', className=''):
        self.typeModel = typeModel
        self.className = className

    def read(self, name):
        data = pd.read_csv(name)
        Prepare.data = data

    def univariateSelection(self):
        if typeModel == 'clasification':
            pass


    def recursiveFeature(self):
        pass


    def pca(self):
        X = Prepare.data.values[:, 0:len(Prepare.data.columns)-1]
        #Y = Prepare.data.values[:, len(data.columns)-1]

        scaler = StandardScaler()
        rescaledX = scaler.fit_transform(X)

        return rescaledX, scaler

    def featureImportance(self):
        X = Prepare.data.values[:, 0:len(Prepare.data.columns)-1]
        #Y = Prepare.data.values[:, len(data.columns)-1]

        normalizer = Normalizer()
        normalizedX = normalizer.fit_transform(X)

        return normalizedX, normalizer

    def binarize(self):
        X = Prepare.data.values[:, 0:len(Prepare.data.columns)-1]
        #Y = Prepare.data.values[:, len(data.columns)-1]

        binarizer = Binarizer()
        binaryX = binarizer.fit_transform(X)

        return binaryX, binarizer
