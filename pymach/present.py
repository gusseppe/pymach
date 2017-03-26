import os
import define
import analyze
import prepare
import feature_selection
import evaluate
import json
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import datetime
import mpld3

from flask import Flask, flash, render_template, redirect, json, request, url_for, jsonify
from threading import Lock
from mpld3 import plugins
from werkzeug import secure_filename
from datetime import datetime

matplotlib.use('Agg')
plt.ioff()

lock = Lock()


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODELS_FOLDER'] = 'models'
ALLOWED_EXTENSIONS = ['txt', 'csv', 'ml']

# Setting up matplotlib sytles using BMH
s = json.load(open("./static/bmh_matplotlibrc.json"))
matplotlib.rcParams.update(s)

#name = "iris.csv"
header = None

def draw_fig(fig_type, className, name):

    with lock:
        #fig, ax = plt.subplots()
        fig, ax = plt.subplots(figsize=(10, 7))
        definer = define.Define(name, header, className).pipeline()
        analyzer = analyze.Analyze(definer)
        #fig = plt.figure(figsize=(10,10))
        if fig_type == "data":
            d = definer.data
            return d.head(20).to_html(show_dimensions=True)
        elif fig_type == "description":
            desc = analyzer.description()
            return desc.to_html(show_dimensions=True)
        elif fig_type == "hist":
            analyzer.hist(ax)
        elif fig_type == "box":
            analyzer.box(ax)
        elif fig_type == "density":
            analyzer.density(ax)
        elif fig_type == "corr":
            analyzer.corr(ax)
        elif fig_type == "scatter":
            analyzer.scatter(ax)
        elif fig_type == "model":
            preparer = prepare.Prepare(definer).pipeline()
            featurer = feature_selection.FeatureSelection(definer).pipeline()
            evaluator = evaluate.Evaluate(definer, preparer, featurer).pipeline()
            results = evaluator.report
            return results.to_html(show_dimensions=True)
    
    return mpld3.fig_to_html(fig)

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/defineData', methods = ['GET', 'POST'])
def defineData():
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    #print(dirs)
    return render_template('uploadData.html', files = dirs)	

@app.route('/storeData', methods = [ 'GET', 'POST'])
def guardarData():
    if request.method == 'POST':
            file = request.files['file']
            if file and allowed_file(file.filename):
                    now = datetime.now()
                    filename = os.path.join(app.config['UPLOAD_FOLDER'], "%s" % (file.filename))
                    file.save(filename)
                    return jsonify({"success":True})
            return redirect(url_for('defineData'))
    else:
            return redirect(url_for('home'))

@app.route('/chooseData', methods = ['GET', 'POST'])
def chooseData():
    #data = json.loads(request.data)
    plot_type = 'data'
    classname = "class"
    name = 'uploads/'
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':
        name = name + request.form['submit']
    return render_template('uploadData.html', files = dirs, dataframe=draw_fig(plot_type, classname, name))	

@app.route('/analyzeData', methods = ['GET', 'POST'])
def analyzeData():
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('analyzeData.html', files = dirs)	

@app.route('/plotData', methods = ['GET', 'POST'])
def plotData():
    plot_type1 = 'hist'
    plot_type2 = 'corr'
    plot_type3 = 'scatter'
    classname = "class"
    name = 'uploads/'
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':
        name = name + request.form['submit']
    ##print(dirs)
    return render_template('analyzeData.html', files = dirs, plot1=draw_fig(plot_type1, classname, name), plot2=draw_fig(plot_type2, classname, name), plot3=draw_fig(plot_type3, classname, name))	
	
@app.route('/models', methods = ['GET', 'POST'])
def models():
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('models.html', files = dirs)	

@app.route('/modelingData', methods = ['GET', 'POST'])
def modelingData():
    plot_type = 'model'
    classname = "class"
    name = 'uploads/'
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':
        name = name + request.form['submit']
    return render_template('models.html', files=dirs, model=draw_fig(plot_type, classname, name))	

#@app.route('/prediction', methods = ['GET', 'POST'])
#def prediction():
	#dirs = os.listdir(app.config['MODELS_FOLDER'])
        #print dirs
	#return render_template('prediction.html', files = dirs)	

#@app.route('/predictData', methods = ['GET', 'POST'])
#def predictionData():
	#dirs = os.listdir(app.config['MODELS_FOLDER'])
	#return render_template('prediction.html', files = dirs)	

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
   app.run(host='0.0.0.0', debug = True, port=8001)
