#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module will define the dataset. Thoughts:
    - Type of model: Classification, Regression, Clustering.
    - Data: save the dataset.
    - header: the dataset's header.
and so forth.
"""

# __all__ = [
#     'pipeline']

import pandas as pd

from collections import OrderedDict
from core.tools import sizeof_df
from sklearn.preprocessing import MinMaxScaler, Normalizer, \
    StandardScaler, RobustScaler, LabelEncoder, FunctionTransformer

class Define():
    """Define module.

    Parameters
    ------------
    data_name : string
    The dataset's name which is expected to be a csv file.

    header : list
    The dataset's header, i.e, the features and the class name.

    response : string
    The name of the variable will be used for prediction.

    Attributes
    -----------
    n_features : int
    number of features or predictors.

    samples : int
    Number of rows in the dataset.

    """


    def __init__(self,
            data_path=None,
            df=None,
            header=None,
            response='class',
            num_features=None,
            cat_features=None,
            problem_type='classification'):

        self.data_path = data_path
        self.df = df
        self.header = header
        self.response = response
        self.metadata = dict()

        self.problem_type = problem_type
        self.infer_algorithm = 'LogisticR'
        self.n_features = None
        self.num_features = num_features
        self.cat_features = cat_features
        self.samples = None
        self.size = None
        self.data = None
        self.X = None
        self.y = None

    def pipeline(self):

        definers = []
        definers.append(self.read)
        definers.append(self.guess_metadata)

        [m() for m in definers]

        return self

    def read(self):
        """Read the dataset.

        Returns
        -------
        out : ndarray

        """
        try:
            if self.df is not None and self.response is not None:
                self.data = self.df
                self.header = self.df.columns
                self.data.dropna(inplace=True)

                self.X = self.data.loc[:, self.data.columns != self.response]
                self.y = self.data[self.response]

            elif self.data_path is not None and self.response is not None:
                if self.header is not None:
                    self.data = pd.read_csv(self.data_path, names=self.header)
                    self.header = self.header
                else:
                    self.data = pd.read_csv(self.data_path)

                self.data.dropna(inplace=True)

                self.X = self.data.loc[:, self.data.columns != self.response]
                self.y = self.data[self.response]

            # return self
        except Exception as e:
            print("Error reading")
            print(e)

    def guess_metadata(self):
        self.n_features = len(self.data.columns)-1
        self.samples = len(self.data)
        # self.size = sizeof_file(self.data_path)
        self.size = sizeof_df(self.data)
        num_features = list(self.data._get_numeric_data().columns)
        cat_features = list(set(self.data.columns) - set(num_features))
        # if cat_features is not None:
        #     cat_features = cat_features.remove('class')
        if ('class' in self.data.columns):
            response = 'class'
        else:
            response = self.data.columns[-1]

        # if str(self.data[response].dtype) == 'object':
        #     encoder = LabelEncoder()
        #     self.data.loc[:, response] = encoder.fit_transform(self.data[response])
        # dict_description = OrderedDict()
        if 'float' in str(self.data[response].dtype):
            problem_type = 'regression'
        else:
            problem_type = 'classification'

        dict_description = dict()
        dict_description["name"] = self.data_path
        dict_description["n_features"] = self.n_features
        dict_description["samples"] = self.samples
        dict_description["size"] = self.size
        dict_description["num_features"] = num_features
        dict_description["cat_features"] = cat_features
        dict_description["response"] = response
        dict_description["problem_type"] = problem_type

        self.metadata = dict_description

        # return  self
        # return dict_description


    def infer(self):
        """ Infer algorithm considering dataset shape, n_features, etc.

        """
        pass


