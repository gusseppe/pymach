#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
This module will define the dataset. Thoughts:
    - Type of model: Clasification, Regression, Clustering.
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

    class_name : string
    The name of the variable will be used for prediction.

    Attributes
    -----------
    n_features : int
    number of features or predictors.

    samples : int
    Number of rows in the dataset.

    """

    problem_type = 'classification'
    infer_algorithm = 'LogisticR'
    n_features = None
    samples = None
    size = None
    data = None
    header = None
    X = None
    y = None

    def __init__(self, 
            data_name, 
            header=None, 
            class_name='class',
            problem_type='classification'):

        self.data_name = data_name
        self.header = header
        self.class_name = class_name

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
            if self.data_name is not None and self.class_name is not None:
                if self.header is not None:
                    Define.data = pd.read_csv(self.data_name, names=self.header)
                    Define.header = self.header
                else:    
                    Define.data = pd.read_csv(self.data_name)

                Define.data.dropna(inplace=True)

                Define.X = Define.data.ix[:, Define.data.columns != self.class_name]
                Define.y = Define.data[self.class_name]
        except:
            print("Error reading")
            
    def description(self):
        Define.n_features = len(Define.data.columns)-1
        Define.samples = len(Define.data)
        Define.size = sizeof_file(self.data_name)

        dict_description = OrderedDict()
        dict_description["name"] = self.data_name
        dict_description["n_features"] = Define.n_features
        dict_description["samples"] = Define.samples
        dict_description["size"] = Define.size

        return dict_description


    def infer(self):
        """ Infer algorithm considering dataset shape, n_features, etc. 
        
        """
        pass


