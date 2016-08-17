#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module provides a few of useful functions (actually, methods)
for preparing the dataset which is to be studied.

"""
from __future__ import print_function
import numpy as np
import pandas as pd

__all__ = [
    'read', 'clean', 'reescale', 'standardize', 'normalize', 'binarize']


class Prepare():
    """ A class for data preparation """

    data = None

    def __init__(self, typeModel='class', className=''):
        self.typeModel = typeModel
        self.className = className

    def read(self, name):
        data = pd.read_csv(name)
        Prepare.data = data

    def clean(self):
        Prepare.data.dropna()

    def reescale(self):
        X = Prepare.data.values[:, 0:len(Prepare.data.columns)-1]
        #Y = Prepare.data.values[:, len(data.columns)-1]

        scaler = MinMaxScaler(feature_range=(0,1))
        rescaledX = scaler.fit_transform(X)

        return rescaledX, scaler

    def standardize(self):
        X = Prepare.data.values[:, 0:len(Prepare.data.columns)-1]
        #Y = Prepare.data.values[:, len(data.columns)-1]

        scaler = StandardScaler()
        rescaledX = scaler.fit_transform(X)

        return rescaledX, scaler

    def normalize(self):
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

    def labelEncoder(self):
        """If a dataset has categorical variables, change it"""
