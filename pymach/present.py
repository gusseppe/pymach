#Standard Libraries
import os
import datetime
import json
import random

#Third Libraries
import pandas as pd
import numpy as np

#Local Libraries
import define
import analyze
import prepare
import feature_selection
import evaluate
import tools

from flask import Flask, flash, render_template, \
        redirect, json, request, url_for, jsonify
from threading import Lock
from mpld3 import plugins
from werkzeug import secure_filename
from datetime import datetime
from collections import OrderedDict


app = Flask(__name__)

APP_PATH = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(APP_PATH, 'uploads')
app.config['MODELS_FOLDER'] = os.path.join(APP_PATH, 'models')
ALLOWED_EXTENSIONS = ['txt', 'csv', 'ml']


#name = "iris.csv"
header = None

def draw_fig(figures, className, name):

    definer = define.Define(
            data_name=name,
            header=header,
            class_name=className).pipeline()

    analyzer = analyze.Analyze(definer)

    dict_figures = OrderedDict()
    for fig in figures:
        dict_figures[fig] = analyzer.plot(fig)
        #if fig == "description":
            #desc = analyzer.description()
            #return desc.to_html(show_dimensions=True)
        #elif fig == "model":
            #preparer = prepare.Prepare(definer).pipeline()
            #featurer = feature_selection.FeatureSelection(definer).pipeline()
            #evaluator = evaluate.Evaluate(definer, preparer, featurer).pipeline()
            #results = evaluator.report
            #return results.to_html(show_dimensions=True)
    
    return dict_figures

def report_model(class_name, data_name):
    definer = define.Define(
            data_name=data_name,
            header=header,
            class_name=class_name).pipeline()

    preparer = prepare.Prepare(definer).pipeline()
    featurer = feature_selection.FeatureSelection(definer).pipeline()
    evaluator = evaluate.Evaluate(definer, preparer, featurer).pipeline()
    #results = evaluator.report

    #print(evaluator.plot_models()) 
    plot = evaluator.plot_models()
    table = evaluator.report
    dict_report = {'plot':plot, 'table':table}
    #return  evaluator.plot_models()
        
    #return results.to_html(show_dimensions=True)

    return dict_report
    #return mpld3.fig_to_html(fig)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#@app.route('/')
#def home():
    #return render_template("uploadData.html")

########################### Start Upload Button ##################################
@app.route('/')
@app.route('/defineData', methods = ['GET', 'POST'])
def defineData():
    """  Show the files which have been uploaded """
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('uploadData.html', files = dirs)	

@app.route('/storeData', methods = [ 'GET', 'POST'])
def storedata():
    """  Upload a new file """
    if request.method == 'POST':
        file = request.files['file']
        #if file and allowed_file(file.filename):
        if file:
            now = datetime.now()
            filename = os.path.join(app.config['UPLOAD_FOLDER'], "%s" % (file.filename))
            file.save(filename)
            return jsonify({"success":True})

        return redirect(url_for('defineData'))
    else:
        return redirect(url_for('defineData'))

@app.route('/chooseData', methods = ['GET', 'POST'])
def chooseData():
    """  choose a file and show its content """
    from itertools import islice
    tools.localization()

    plot_type = 'data'
    classname = "class"
    data_name = ''
    path_name = ''
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':
        data_name = request.form['submit']
        path_data = os.path.join(app.config['UPLOAD_FOLDER'], data_name)

    dataset = []
    with open(path_data) as myfile:
        dataset = list(islice(myfile, 20))
        dataset = [line[:-1] for line in dataset]

    return render_template(
            'uploadData.html',
            files = dirs,
            dataset=dataset,
            data_name=data_name)	


########################### End Upload Button ##################################

########################### Start Analyze Button ##################################
@app.route('/analyzeData', methods = ['GET', 'POST'])
def analyzeData():
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('analyzeData.html', files = dirs)	

@app.route('/plotData', methods = ['GET', 'POST'])
def plotData():
    figures = ['histogram', 'box', 'corr']
    classname = "class"
    data_name = ''
    path_name = ''
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':
        data_name = request.form['submit']
        path_data = os.path.join(app.config['UPLOAD_FOLDER'], data_name)

    return render_template(
            'analyzeData.html',
            files=dirs,
            figures=draw_fig(figures, classname, path_data),
            data_name=data_name)	

########################### End Analyze Button ##################################

########################### Start Model Button ##################################
@app.route('/models', methods = ['GET', 'POST'])
def models():
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('models.html', files = dirs)	

@app.route('/modelingData', methods = ['GET', 'POST'])
def modelingData():
    #plot_type = 'model'
    classname = "class"
    data_name = ''
    path_name = ''
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':
        data_name = request.form['submit']
        path_data = os.path.join(app.config['UPLOAD_FOLDER'], data_name)
        #name = name + request.form['submit']
    return render_template(
            'models.html', 
            files=dirs,
            report=report_model(classname, path_data),
            data_name=data_name)	

########################### End Model Button ##################################

@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    attributes = []
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    data_class = 'class'
    filename = 'iris.csv'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    model = 'Naive Bayes'
    f = open(filepath, 'r')
    g = open(filepath, 'r')
    for item in g.readline().split(','):
        if item.strip() != data_class:
            attributes.append(item)
    print(attributes, ' this is something')
    return render_template('showPrediction.html', file = f, attributes = attributes, data_class = data_class, model = model)


if __name__ == '__main__':
   app.run(host='0.0.0.0', debug = True, port=8002)
