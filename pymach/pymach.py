#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

"""
This module provides the logic of the whole project.

"""
#import analyze
import define
import analyze
import prepare
#print obj.data
#print obj.description()
#print obj.classBalance()
#print obj.hist()
#print obj.density()
#print obj.corr()
if __name__ == "__main__":
    analyzer = analyze.Analyze(define)
    d = analyzer.read()
    preparer = prepare.Prepare(define)
    preparer = preparer.pipeline()
    print preparer
    y = preparer.fit_transform(d)
    print y[0:2,:]
    #print d
    #analyzer.pipeline()
    #print analyzer.data
    #obj.read("iris.csv")
    #print(obj.scatter())

