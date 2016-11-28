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
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, Normalizer,\
StandardScaler


__all__ = [
    'pipeline']


class Prepare():
    """ A class for data preparation """

    data = None

    def __init__(self, definer):
        self.typeModel = definer.typeModel
        self.typeAlgorithm = definer.typeAlgorithm
        self.className = definer.className
        self.nameData = definer.nameData

    def pipeline(self):
        """ This function chooses the best way to scale a data"""

        transformers = []

        clean = self.Clean()
        transformers.append(('clean', clean))

        if self.typeAlgorithm in ["NeuralN", "K-N"]:
            minmax = MinMaxScaler(feature_range=(0,1))
            normalizer = Normalizer()
            transformers.append(('minmax', minmax))
            transformers.append(('normalizer', normalizer))
        elif self.typeAlgorithm in ["LinearR", "LogisticR"]:
            #print('hola')
            scaler = StandardScaler()
            transformers.append(('scaler', scaler))
        else:
            scaler = StandardScaler()
            transformers.append(('scaler', scaler))

        #scaler = StandardScaler()
        #transformers.append(('scaler', scaler))
        #binarizer = Binarizer()
        return FeatureUnion(transformers)

    class Clean(TransformerMixin):
        """ A class for removing NAN values """

        def transform(self, X, **transform_params):
            #X = pd.DataFrame(X)
            return X.dropna()

        def fit(self, X, y=None, **fit_params):
            return self


    #def binarize(self):
        #X = Prepare.data.values[:, 0:len(Prepare.data.columns)-1]
        ##Y = Prepare.data.values[:, len(data.columns)-1]

        #binarizer = Binarizer()
        #binaryX = binarizer.fit_transform(X)

        #return binaryX, binarizer

    def labelEncoder(self):
        """If a dataset has categorical variables, change it"""
        pass
