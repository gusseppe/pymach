# Standard Libraries
import os
# import datetime

# Third Libraries
# import pandas as pd
# import numpy as np

# Local Libraries
import define
import analyze
import prepare
import fselect
import evaluate
import improve
import tools

from flask import Flask, render_template, \
        redirect, request, url_for, jsonify
from collections import OrderedDict


app = Flask(__name__)

APP_PATH = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_DIR'] = os.path.join(APP_PATH, 'uploads')
app.config['MODELS_DIR'] = os.path.join(APP_PATH, 'models')
app.config['MARKET_DIR'] = os.path.join(APP_PATH, 'market')
ALLOWED_EXTENSIONS = ['txt', 'csv', 'ml']


def report_analyze(figures, className, data_path, data_name):

    definer = define.Define(
            data_path=data_path,
            header=None,
            response=className).pipeline()

    analyzer = analyze.Analyze(definer)

    dict_figures = OrderedDict()
    for fig in figures:
        data_name = data_name.replace(".csv", "")
        plot_path = os.path.join(app.config['MARKET_DIR'], data_name, 'analyze')
        tools.path_exists(plot_path)
        plot_path_plot = os.path.join(plot_path, fig+'.html')
        dict_figures[fig] = analyzer.plot(fig)
        analyzer.save_plot(plot_path_plot)

    return dict_figures


def report_model(class_name, data_path, data_name):
    definer = define.Define(
            data_path=data_path,
            header=None,
            response=class_name).pipeline()

    preparer = prepare.Prepare(definer).pipeline()
    selector = fselect.Select(definer).pipeline()
    evaluator = evaluate.Evaluate(definer, preparer, selector).pipeline()

    plot = evaluator.plot_models()
    table = evaluator.report

    data_name = data_name.replace(".csv", "")
    plot_path = os.path.join(app.config['MARKET_DIR'], data_name, 'model')
    tools.path_exists(plot_path)
    plot_path_plot = os.path.join(plot_path, 'boxplot.html')
    evaluator.save_plot(plot_path_plot)
    plot_path_report = os.path.join(plot_path, 'report.csv')
    evaluator.save_report(plot_path_report)

    dict_report = {'plot': plot, 'table': table}

    return dict_report


def report_improve(class_name, data_name):
    definer = define.Define(
            data_path=data_name,
            header=None,
            response=class_name).pipeline()

    preparer = prepare.Prepare(definer).pipeline()
    selector = fselect.Select(definer).pipeline()
    evaluator = evaluate.Evaluate(definer, preparer, selector)
    improver = improve.Improve(evaluator).pipeline()

    plot = improver.plot_models()
    table = improver.report
    dict_report = {'plot': plot, 'table': table}

    return dict_report

def report_market(data_name):

    analyze_report = OrderedDict()
    model_report = OrderedDict()

    data_name = data_name.replace(".csv", "")
    app_dirs = os.listdir(app.config['MARKET_DIR'], data_name)

    for market_app in market_apps:
        plot_path = os.path.join(app.config['MARKET_DIR'], data_name, 'model')
        tools.path_exists(plot_path)
        plot_path = os.path.join(plot_path, fig+'.html')
        dict_figures[fig] = analyzer.plot(fig)
        analyzer.save_plot(plot_path)

    plot = improver.plot_models()
    table = improver.report
    dict_report = {'plot': plot, 'table': table}

    return dict_report

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#@app.route('/')
#def home():
    #return render_template("uploadData.html")

########################### Start Upload Button ##################################
@app.route('/')
@app.route('/defineData', methods=['GET', 'POST'])
def defineData():
    """  Show the files that have been uploaded """
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    return render_template('uploadData.html', files=dirs)


@app.route('/storeData', methods=['GET', 'POST'])
def storedata():
    """  Upload a new file """
    if request.method == 'POST':
        file = request.files['file']
        #if file and allowed_file(file.filename):
        if file:
            # now = datetime.now()
            filename = os.path.join(app.config['UPLOAD_DIR'], "%s" % (file.filename))
            file.save(filename)
            return jsonify({"success":True})

        return redirect(url_for('defineData'))
    else:
        return redirect(url_for('defineData'))


@app.route('/chooseData', methods=['GET', 'POST'])
def chooseData():
    """  choose a file and show its content """
    from itertools import islice
    # tools.localization()

    data_name = ''
    data_path = ''
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    if request.method == 'POST':
        data_name = request.form['submit']
        data_path = os.path.join(app.config['UPLOAD_DIR'], data_name)

    dataset = []
    with open(data_path) as myfile:
        dataset = list(islice(myfile, 20))
        dataset = [line[:-1] for line in dataset]

    return render_template(
            'uploadData.html',
            files=dirs,
            dataset=dataset,
            data_name=data_name)


########################### End Upload Button ##################################

########################### Start Analyze Button ##################################
@app.route('/analyze_base', methods=['GET', 'POST'])
def analyze_base():
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    return render_template('analyzeData.html', files=dirs)


@app.route('/analyze_app', methods=['GET', 'POST'])
def analyze_app():
    figures = ['histogram', 'box', 'corr']
    classname = "class"
    data_name = ''
    data_path = ''
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    if request.method == 'POST':
        data_name = request.form['submit']
        data_path = os.path.join(app.config['UPLOAD_DIR'], data_name)

    return render_template(
            'analyzeData.html',
            files=dirs,
            figures=report_analyze(figures, classname, data_path, data_name),
            data_name=data_name)

########################### End Analyze Button ##################################

########################### Start Model Button ##################################
@app.route('/model_base', methods=['GET', 'POST'])
def model_base():
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    return render_template('models.html', files=dirs)


@app.route('/model_app', methods=['GET', 'POST'])
def model_app():
    classname = "class"
    data_name = ''
    data_path = ''
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    if request.method == 'POST':
        data_name = request.form['submit']
        data_path = os.path.join(app.config['UPLOAD_DIR'], data_name)

    return render_template(
            'models.html',
            files=dirs,
            report=report_model(classname, data_path, data_name),
            data_name=data_name)

########################### End Model Button ##################################

########################### Start Improve Button ##################################
@app.route('/improve_base', methods=['GET', 'POST'])
def improve_base():
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    return render_template('improve.html', files=dirs)

@app.route('/improve_app', methods=['GET', 'POST'])
def improve_app():
    classname = "class"
    data_name = ''
    data_path = ''
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    if request.method == 'POST':
        data_name = request.form['submit']
        data_path = os.path.join(app.config['UPLOAD_DIR'], data_name)

    return render_template(
            'improve.html',
            files=dirs,
            report=report_model(classname, data_path, data_name),
            data_name=data_name)

########################### End Improve Button ##################################

########################### Start Model Button ##################################
@app.route('/market_base', methods=['GET', 'POST'])
def market_base():
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    return render_template('market.html', files=dirs)


@app.route('/market_app', methods=['GET', 'POST'])
def market_app():
    classname = "class"
    data_name = ''
    data_path = ''
    dirs = os.listdir(app.config['MARKET_DIR'])
    if request.method == 'POST':
        data_name = request.form['submit']
        data_path = os.path.join(app.config['MARKET_DIR'], data_name)
    return render_template(
            'market.html',
            files=dirs,
            report=report_model(classname, data_path, data_name),
            data_name=data_name)

########################### End Market Button ##################################

# @app.route('/prediction', methods=['GET', 'POST'])
# def prediction():
    # attributes = []
    # dirs = os.listdir(app.config['UPLOAD_DIR'])
    # data_class = 'class'
    # filename = 'iris.csv'
    # filepath = os.path.join(app.config['UPLOAD_DIR'], filename)
    # model = 'Naive Bayes'
    # f = open(filepath, 'r')
    # g = open(filepath, 'r')
    # for item in g.readline().split(','):
        # if item.strip() != data_class:
            # attributes.append(item)
    # print(attributes, ' this is something')
    # return render_template('showPrediction.html', file = f, attributes = attributes, data_class = data_class, model = model)


if __name__ == '__main__':
   app.run(host='0.0.0.0', debug=True, port=8002)
