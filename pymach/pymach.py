#!/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Gusseppe Bravo <gbravor@uni.pe>
# License: BSD 3 clause

"""
This module provides the logic of the whole project.

"""
import define
import analyze
import prepare
import feature_selection
import evaluate

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn import cross_validation
import pandas as pd


#name = "inputLocalizacion.csv"
name = "iris.csv"
#name = "inputBus.csv"
# className = "Ruta"
#className = "position"
className = "class"

#STEP 0: Define workflow parameters
definer = define.Define(nameData=name, className=className).pipeline()

#STEP 1: Analyze data by ploting it
#analyze.Analyze(definer).pipeline()

#STEP 2: Prepare data by scaling, normalizing, etc. 
preparer = prepare.Prepare(definer).pipeline()

#STEP 3: Feature selection
featurer = feature_selection.FeatureSelection(definer).pipeline()

#STEP4: 
evaluator = evaluate.Evaluate(definer, preparer, featurer).pipeline()

#print definer.data.head(20)
#print ""

#preparer.fit(definer.X, definer.y)
#x,y = preparer.ransform(definer.X, definer.y)
#x,y = preparer.fit_transform(definer.X, definer.y)
#t = pd.DataFrame(t, columns=definer.data.columns)
#print t.head(20)








#print(evaluator.pipelines)
#print(evaluator.bestAlgorithms)
#pipeline = Pipeline([
    #('preparer', preparer),
    #('featurer', featurer),
    #('svc', SVC(kernel='linear')),
    #])

##STEP Evaluation
#test_size = 0.33
#seed = 7
#X_train, X_test, Y_train, Y_test =  cross_validation.train_test_split(definer.X, definer.y,
#test_size=test_size, random_state=seed)

#f = open('test', 'w')
#f.write('Model:\n\n')
#f.writelines(str(pipeline.fit(X_train, Y_train)))
#f.write('\n\n Score:')
#f.write(str(pipeline.score(X_test, Y_test)))
#f.close()
#print pipeline.fit(X_train, Y_train)
#print pipeline.score(X_test, Y_test)
#print pipeline.predict(X_test)

