#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module will prepare the dataset, i.e, modify the data. Thoughts:
    - Applying scaling, normalizing, cleaning, etc.
    - Processing will be performed depending on the inferred algorithm.
and so forth.
"""

from __future__ import print_function

# __all__ = [
#     'pipeline']

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, Normalizer,\
StandardScaler, RobustScaler, LabelEncoder, FunctionTransformer




class Prepare():
    """ A class for data preparation """

    # data = None

    def __init__(self, definer):
        self.problem_type = definer.problem_type
        self.infer_algorithm = definer.infer_algorithm
        self.categoricalData = False
        # self.className = definer.className
        # self.data_name = definer.data_name

    def pipeline(self):
        """ This function chooses the best way to scale a data"""

        transformers = []

        #clean = self.Clean()
        #transformers.append(('clean', FunctionTransformer(clean, validate=False)))
        #transformers.append(('clean', clean))

        #catToNumeric = self.CategoricalToNumeric()
        #transformers.append(('catToNumeric', catToNumeric))

        if self.infer_algorithm in ["NeuralN", "K-N"]:
            minmax = MinMaxScaler(feature_range=(0,1))
            normalizer = Normalizer()
            transformers.append(('minmax', minmax))
            transformers.append(('normalizer', normalizer))
        elif self.infer_algorithm in ["LinearR", "LogisticR"]:
            #scaler = RobustScaler()
            scaler = StandardScaler()
            transformers.append(('scaler', scaler))
        else:
            #scaler = StandardScaler()
            scaler = RobustScaler()
            transformers.append(('scaler', scaler))

        #scaler = StandardScaler()
        #transformers.append(('scaler', scaler))
        #binarizer = Binarizer()
        #print(transformers)
        return FeatureUnion(transformers)

    class Clean(BaseEstimator, TransformerMixin):
        """ A class for removing NAN values """

        def transform(self, X, y=None, **transform_params):
            #X = pd.DataFrame(X)
            X.dropna(inplace=True)
            #return X.dropna()
            return X

        def fit_transform(self, X, y=None, **fit_params):
            self.fit(X, y, **fit_params)
            return self.transform(X)

        def fit(self, X, y=None, **fit_params):
            return self

    class CategoricalToNumeric(BaseEstimator, TransformerMixin):
        """ A class for parsing categorical columns """

        def categoricalColumns(self, df):
            cols = df.columns
            cols_numeric = df._get_numeric_data().columns
            return list(set(cols) - set(cols_numeric))

        def categoricalToNumeric(self, df):
            cat_columns = self.categoricalColumns(df)
            if cat_columns:
                self.categoricalData = True
                for category in cat_columns:
                    encoder = LabelEncoder()
                    #df.loc[:, category+'_n'] = encoder.fit_transform(df[category])
                    df.loc[:, category] = encoder.fit_transform(df[category])

            #df.drop(cat_columns, axis=1, inplace=True)
            return df

        def transform(self, X, y=None, **transform_params):
            #X = pd.DataFrame(X)
            return self.categoricalToNumeric(X)
            #return X.dropna()

        def fit_transform(self, X, y=None, **fit_params):
            self.fit(X, y, **fit_params)
            return self.transform(X)

        def fit(self, X, y=None, **fit_params):
            return self

    class StringToNumber(BaseEstimator, TransformerMixin):
        def __init__(self, column='class'):
            self.column = column

        def transform(self, X, y=None, **transform_params):
            result = []
            encoder = LabelEncoder()
            y.loc[:, self.column] = encoder.fit_transform(y[self.column])
            # for index, rowdata in X.iterrows():
            #     rowdict = {}
            #     for kvp in self.kpairs:
            #         rowdict.update( { rowdata[ kvp[0] ]: rowdata[ kvp[1] ] } )
            #     result.append(rowdict)
            # return result
            return y

        def fit(self, *_):
            return self

