import os
import define
import analyze
import prepare
import feature_selection
import evaluate
import json
import random
import pandas as pd
#import matplotlib
#import matplotlib.pyplot as plt
import numpy as np
import datetime
#import mpld3

from flask import Flask, flash, render_template, redirect, json, request, url_for, jsonify
from threading import Lock
from mpld3 import plugins
from werkzeug import secure_filename
from datetime import datetime
from collections import OrderedDict

#matplotlib.use('Agg')
#plt.ioff()

#lock = Lock()


app = Flask(__name__)

APP_PATH = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(APP_PATH, 'uploads')
app.config['MODELS_FOLDER'] = os.path.join(APP_PATH, 'models')
ALLOWED_EXTENSIONS = ['txt', 'csv', 'ml']

# Setting up matplotlib sytles using BMH
#MATPLOT_STYLES_PATH = os.path.join(APP_PATH, 'static/bmh_matplotlibrc.json')
#s = json.load(open("./static/bmh_matplotlibrc.json"))
#s = json.load(open(MATPLOT_STYLES_PATH))
#matplotlib.rcParams.update(s)

#name = "iris.csv"
header = None

def draw_fig(figures, className, name):

    #with lock:
    #fig, ax = plt.subplots()
    #fig, ax = plt.subplots(figsize=(10, 7))
    definer = define.Define(name, header, className).pipeline()
    analyzer = analyze.Analyze(definer)
    #fig = plt.figure(figsize=(10,10))
    #if fig_type == "data":
        #d = definer.data
        #return d.head(20).to_html(show_dimensions=True)
    dict_figures = OrderedDict()
    for fig in figures:
        dict_figures[fig] = analyzer.plot(fig)
        #if fig == "description":
            #desc = analyzer.description()
            #return desc.to_html(show_dimensions=True)
        #elif fig == "hist":
            ##analyzer.hist(ax)
            #return analyzer.histogram()
            ##with open('div') as f:
                ##t = f.read()
                ##print t
                ##return t
        #elif fig == "box":
            #return analyzer.boxplot()
            ##analyzer.box(ax)
        #elif fig == "density":
            #pass
            ##analyzer.density(ax)
        #elif fig == "corr":
            #return analyzer.correlation()
            ##analyzer.corr(ax)
        #elif fig == "scatter":
            #return analyzer.scatter()
            ##analyzer.scatter(ax)
        #elif fig == "model":
            #preparer = prepare.Prepare(definer).pipeline()
            #featurer = feature_selection.FeatureSelection(definer).pipeline()
            #evaluator = evaluate.Evaluate(definer, preparer, featurer).pipeline()
            #results = evaluator.report
            #return results.to_html(show_dimensions=True)
    
    return dict_figures
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
            if file and allowed_file(file.filename):
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
    plot_type = 'data'
    classname = "class"
    name = ''
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':
        name = os.path.join(app.config['UPLOAD_FOLDER'], request.form['submit'])
    dataset = []
    with open(name) as myfile:
        dataset = list(islice(myfile, 20))
        dataset = [line[:-1] for line in dataset]
    #print dataset
    #return render_template('uploadData.html', files = dirs, f=f, dataframe=draw_fig(plot_type, classname, name))	
    return render_template('uploadData.html', files = dirs, dataset=dataset)	


########################### End Upload Button ##################################

########################### Start Analyze Button ##################################
@app.route('/analyzeData', methods = ['GET', 'POST'])
def analyzeData():
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('analyzeData.html', files = dirs)	

@app.route('/plotData', methods = ['GET', 'POST'])
def plotData():
    figures = ['histogram', 'box', 'corr', 'scatter']
    classname = "class"
    name = ''
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':
        name = os.path.join(app.config['UPLOAD_FOLDER'], request.form['submit'])
        #name = name + request.form['submit']

    return render_template('analyzeData.html', files=dirs, figures=draw_fig(figures, classname, name))	
	
########################### End Analyze Button ##################################

########################### Start Model Button ##################################
@app.route('/models', methods = ['GET', 'POST'])
def models():
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('models.html', files = dirs)	

@app.route('/modelingData', methods = ['GET', 'POST'])
def modelingData():
    plot_type = 'model'
    classname = "class"
    name = ''
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':
        name = os.path.join(app.config['UPLOAD_FOLDER'], request.form['submit'])
        #name = name + request.form['submit']
    return render_template('models.html', files=dirs, model=draw_fig(plot_type, classname, name))	

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
