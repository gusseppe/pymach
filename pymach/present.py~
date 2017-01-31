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
from flask import Flask, flash, render_template, redirect, json, request, url_for
from threading import Lock
from mpld3 import plugins
from werkzeug import secure_filename


matplotlib.use('Agg')
plt.ioff()

name = "input.csv"
header = None
#header = ["position", "b1", "b2", "b3", "b4", "b5"]
className = ""


lock = Lock()

# Setting up matplotlib sytles using BMH
s = json.load(open("./static/bmh_matplotlibrc.json"))
matplotlib.rcParams.update(s)

def draw_fig(fig_type, className):

    with lock:
        fig, ax = plt.subplots()
        definer = define.Define(name, header, className).pipeline()
        analyzer = analyze.Analyze(definer)
        #fig = plt.figure(figsize=(2,5))
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
            evaluator = evaluate.Evaluate(definer, preparer, featurer).pipeline(ax)
    
    return mpld3.fig_to_html(fig)

app = Flask(__name__)
app.secret_key = 'random string'

@app.route('/')
def home():
    return render_template('index.html')

#@app.route('/model')
#def modelPage():
    #return render_template('model.html')

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['data_file']
    text = request.form['text']
    if not f and not text:
        #return redirect(url_for('query'))
        return render_template('index.html')
        #return "No file"
    else:
        f.save(secure_filename('input.csv'))
        cl = open("classname", "w")
        cl.write(text)
        #return redirect(url_for('query'))
        #return redirect(url_for('query', classname = text))
        return render_template('analyze.html')

@app.route('/analyze/query', methods=['GET', 'POST'])
def query():

    #loadData("input.csv", "position", None)
    #print classname
    data = json.loads(request.data)
    cl = open('classname', "r")
    classname = cl.readline()
    #print classname
    cl.close()
    return draw_fig(data["plot_type"], classname)

#@app.route('/model/query', methods=['GET', 'POST'])
#def model():

    ##loadData("input.csv", "position", None)
    ##print classname
    #data = json.loads(request.data)
    #cl = open('classname', "r")
    #classname = cl.readline()
    ##print classname
    #cl.close()
    #return draw_fig(data["plot_type"], classname)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
