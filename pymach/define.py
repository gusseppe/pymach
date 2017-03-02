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
    #className = 'species'
    #nameData = 'iris.csv'
    n_features = None
    samples = None
    data = None
    header = None
    X = None
    y = None

    def __init__(self, nameData, header=None, className=None):
        self.nameData = nameData
        self.header = header
        self.className = className

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
            if self.nameData is not None and self.className is not None:
                if self.header is not None:
                    Define.data = pd.read_csv(self.nameData, names=self.header)
                    Define.header = self.header
                else:    
                    Define.data = pd.read_csv(self.nameData)

                Define.data.dropna(inplace=True)

                Define.X = Define.data.ix[:, Define.data.columns != self.className]
                Define.y = Define.data[self.className]
        except:
            print("Error reading")
            
    def description(self):
        Define.n_features = len(Define.data.columns)-1
        Define.samples = len(Define.data)


