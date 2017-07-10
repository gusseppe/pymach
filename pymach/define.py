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

__all__ = [
    'pipeline']

import pandas as pd

from collections import OrderedDict
from tools import sizeof_file

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
            data_path,
            header=None,
            response='class',
            problem_type='classification'):

        self.data_path = data_path
        self.header = header
        self.response = response

        self.problem_type = 'classification'
        self.infer_algorithm = 'LogisticR'
        self.n_features = None
        self.samples = None
        self.size = None
        self.data = None
        self.X = None
        self.y = None

    def pipeline(self):

        definers = []
        definers.append(self.read)
        definers.append(self.description)

        [m() for m in definers]

        return self

    def read(self):
        """Read the dataset.

        Returns
        -------
        out : ndarray

        """
        try:
            if self.data_path is not None and self.response is not None:
                if self.header is not None:
                    self.data = pd.read_csv(self.data_path, names=self.header)
                    self.header = self.header
                else:
                    self.data = pd.read_csv(self.data_path)

                self.data.dropna(inplace=True)

                self.X = self.data.loc[:, self.data.columns != self.response]
                self.y = self.data[self.response]
        except:
            print("Error reading")

    def description(self):
        self.n_features = len(self.data.columns)-1
        self.samples = len(self.data)
        self.size = sizeof_file(self.data_path)

        dict_description = OrderedDict()
        dict_description["name"] = self.data_path
        dict_description["n_features"] = self.n_features
        dict_description["samples"] = self.samples
        dict_description["size"] = self.size

        return dict_description


    def infer(self):
        """ Infer algorithm considering dataset shape, n_features, etc.

        """
        pass


