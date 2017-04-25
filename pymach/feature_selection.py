#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module provides a few of useful functions (actually, methods)
for feature selection the dataset which is to be studied.

"""
from __future__ import print_function
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    'pipeline']


class FeatureSelection():
    """ A class for feature selection """

    data = None

    def __init__(self, definer):
        self.problem_type = definer.problem_type
        self.infer_algorithm = definer.infer_algorithm
        self.class_name = definer.class_name
        self.data_name = definer.data_name
        self.n_features = definer.n_features

    def pipeline(self):
        """ This function chooses the best way to find features"""

        transformers = []

        custom = self.CustomFeature()
        #transformers.append(('custom', custom))
        n_features = int(self.n_features/2)

        #kbest = SelectKBest(score_func=chi2, k=n_features)
        #transformers.append(('kbest', kbest))

        pca = PCA(n_components=n_features)
        transformers.append(('pca', pca))

        extraTC = ExtraTreesClassifier()
        transformers.append(('extraTC', extraTC))

        #scaler = StandardScaler()
        #transformers.append(('scaler', scaler))
        #binarizer = Binarizer()
        return FeatureUnion(transformers)

    class CustomFeature(TransformerMixin):
        """ A custome class for featuring """

        def transform(self, X, **transform_params):
            #X = pd.DataFrame(X)
            return X

        def fit(self, X, y=None, **fit_params):
            return self
