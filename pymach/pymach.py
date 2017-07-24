#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

"""
This module provides the logic of the whole project.

"""
#import define
#import analyze
#import prepare
#import feature_selection
#import evaluate

#import time

#name = "datasets/iris.csv"
##name = "datasets/miningData.csv"
##name = "datasets/LocalizationOld.csv"
##name = "datasets/seguridad.csv"
##name = "datasets/breast-cancer-wisconsin.csv"
##name = "breast-cancer-wisconsin.csv"
##name = "inputBus.csv"
## className = "Ruta"
##className = "CATEGORY"
##className = "accident_type"
##className = "position"
#className = "class"

#def main():
    ##STEP 0: Define workflow parameters
    #definer = define.Define(nameData=name, className=className).pipeline()

    ##STEP 1: Analyze data by ploting it
    ##analyze.Analyze(definer).pipeline()

    ##STEP 2: Prepare data by scaling, normalizing, etc.
    #preparer = prepare.Prepare(definer).pipeline()

    ##STEP 3: Feature selection
    #featurer = feature_selection.FeatureSelection(definer).pipeline()

    ##STEP4: Evalute the algorithms by using the pipelines
    ##evaluator = evaluate.Evaluate(definer, preparer, featurer).pipeline()

#if __name__ == '__main__':
    #start = time.time()
    #main()
    #end = time.time()

    #print()
    #print("Execution time for all the steps: ", end-start)

# TESTING
import define
import analyze
import prepare
import fselect
import evaluate
import improve

data_name = "iris.csv"
class_name = "class"
definer = define.Define(
        data_path=data_name,
        header=None,
        response=class_name).pipeline()

preparer = prepare.Prepare(definer).pipeline()
selector = fselect.Select(definer).pipeline()
evaluator = evaluate.Evaluate(definer, preparer, selector)
improver = improve.Improve(evaluator).pipeline()
# improver.improve_pipelines()
print(improver.score_report)
print(improver.full_report)
