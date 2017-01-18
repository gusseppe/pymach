#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause
"""
En esta clase se define que problema se va a solucionar.
Sea de clasificacion, regression, clustering. Ademas se debe
dar una idea de los posibles algoritmos que pueden ser usados.
"""

import pandas as pd

class Define():

    typeModel = 'clasification'
    typeAlgorithm = 'LogisticR'
    className = 'species'
    nameData = 'iris.csv'
    n_features = None
    samples = None
    X = None
    y = None

    def __init__(self):
        pass

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
        Define.data = pd.read_csv(Define.nameData)
        Define.X = Define.data.ix[:, Define.data.columns != Define.className]
        Define.y = Define.data[Define.className]

    def description(self):
        Define.n_features = len(Define.data.columns)-1
        #Define.className = Define.data.columns[-1]
        Define.samples = len(Define.data)

    #def likelyAlgorithms(self):
        #if Define.samples < 50:
            #print("Not enough data")
        #else:
            #pass




        

