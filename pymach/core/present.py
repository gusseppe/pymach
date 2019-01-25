# Standard Libraries
import os
import subprocess
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

import pandas as pd

from flask import Flask, render_template, \
        redirect, request, url_for, jsonify, flash
from werkzeug.utils import secure_filename
from collections import OrderedDict



app = Flask(__name__)

APP_PATH = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_DIR'] = os.path.join(APP_PATH, 'uploads')
app.config['MODELS_DIR'] = os.path.join(APP_PATH, 'models')
app.config['MARKET_DIR'] = os.path.join(APP_PATH, 'market')
ALLOWED_EXTENSIONS = ['txt', 'csv', 'ml', 'html']


def report_analyze(figures, response, data_path, data_name):

    definer = define.Define(
            data_path=data_path,
            header=None,
            response=response).pipeline()

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


def report_model(response, data_path, data_name):
    definer = define.Define(
            data_path=data_path,
            header=None,
            response=response,
            problem_type='regression').pipeline()

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


def report_improve(response, data_name):
    definer = define.Define(
            data_path=data_name,
            header=None,
            response=response).pipeline()

    preparer = prepare.Prepare(definer).pipeline()
    selector = fselect.Select(definer).pipeline()
    evaluator = evaluate.Evaluate(definer, preparer, selector)
    improver = improve.Improve(evaluator).pipeline()

    plot = improver.plot_models()
    table = improver.report
    dict_report = {'plot': plot, 'table': table}

    return dict_report

def report_market(data_name):

    # analyze_report = OrderedDict()
    # model_report = OrderedDict()

    data_name = data_name.replace(".csv", "")
    app_path = os.path.join(app.config['MARKET_DIR'], data_name)
    # app_dirs = os.listdir(app_path)

    # Show Model info
    try:
        model_path = os.path.join(app_path, 'model')
        plot_model = ''
        with open(os.path.join(model_path, 'boxplot.html')) as f:
            plot_model = f.read()

        table_model = pd.read_csv(os.path.join(model_path, 'report.csv'))
        dict_report_model = {'plot':plot_model, 'table':table_model}  # return 1
    except:
        dict_report_model = {'plot':None, 'table':None}  # return 1


    # Show Analyze info
    try:
        analyze_path = os.path.join(app_path, 'analyze')
        plot_analyze = OrderedDict()
        for plot in os.listdir(analyze_path):
            with open(os.path.join(analyze_path, plot)) as f:
               fig = plot.replace('.html', '')
               plot_analyze[fig] = f.read()

        # Join full report: model and analyze
        dicts_market = {'model':dict_report_model, 'analyze':plot_analyze}
    except:
        dicts_market = {'model':dict_report_model, 'analyze':None}  # return 2


    return dicts_market

def allowed_file(file_name):
    return '.' in file_name and file_name.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

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
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        #if file and allowed_file(file.file_name):
        file_name = ''
        data_name = ''

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
        # if file:
            file_name = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_DIR'], file_name)
            file.save(file_path)
            # file_name = str(file.filename)
            # data_name = file_name.replace(".csv", "")
            # print(data_name)
            # command = 'csvtotable -c "Iris dataset" iris.csv iris.html'
            return jsonify({"success":True})

            # result = subprocess.run(['csvtotable', '-c',
            #                          data_name, file_name, data_name+'.html'],
            #                         stdout=subprocess.PIPE)

        # return redirect(url_for('showData', filename=file_name))
        return redirect(url_for('defineData'))
    else:
        return redirect(url_for('defineData'))


@app.route('/chooseData', methods=['GET', 'POST'])
def chooseData():
    """  choose a file and show its content """
    from itertools import islice
    # tools.localization()

    file_name = ''
    data_name = ''
    data_path = ''
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    if request.method == 'POST':
        file_name = request.form['submit']
        data_name = file_name.replace(".csv", "")
        data_path = os.path.join(app.config['UPLOAD_DIR'], data_name+'.html')

    # result = subprocess.run(['csvtotable', '-c', '--display-length','50',
    #                          data_name, data_name+'.csv', data_name+'.html'],
    #                         stdout=subprocess.PIPE)

    try:
        dataset = None
        with open(data_path) as f:
            dataset = f.read()

    except:
        data_path = os.path.join(app.config['UPLOAD_DIR'], file_name)
        with open(data_path) as myfile:
            dataset = list(islice(myfile, 40))
            dataset = [line[:-1] for line in dataset]

    return render_template(
            'uploadData.html',
            files=dirs,
            dataset=dataset,
            data_name=data_name)


########################### End Upload Button ##################################

# Convert the uploaded csv file into a responsive table.
# ########################## Start Convert table ##################################
# @app.route('/chooseData/<filename>')
# def showData(filename):
#     """  choose a file and show its content """
#     from itertools import islice
#
#     data_name = filename.replace(".csv", "")
#     dirs = os.listdir(app.config['UPLOAD_DIR'])
#     # result = subprocess.run(['csvtotable', '-c',
#     #                          data_name, filename, data_name+'.html'],
#     #                         stdout=subprocess.PIPE)
#
#     dataset = 'asdasd'
#     print(filename + 'start')
#     data_path = os.path.join(app.config['UPLOAD_DIR'], filename)
#     comm = 'csvtotable -c' + " Iris " + filename + ' ' + data_name+'.html'
#     os.system(comm)
#     # with open(data_path) as f:
#     #     dataset = f.read()
#     #     print(dataset[0])
#     print(filename + 'end')
#     # data_path = os.path.join(app.config['UPLOAD_DIR'], data_name+'.html')
#     #
#     # dataset = None
#     # with open(data_path) as f:
#     #     dataset = f.read()
#
#     # with open(data_path) as myfile:
#     #     dataset = list(islice(myfile, 40))
#     #     dataset = [line[:-1] for line in dataset]
#
#     return render_template(
#         'uploadData.html',
#         files=dirs,
#         dataset=dataset,
#         data_name=data_name)

# ########################## End Convert table ##################################

# ########################## Start Analyze Button ##################################
@app.route('/analyze_base', methods=['GET', 'POST'])
def analyze_base():
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    return render_template('analyzeData.html', files=dirs)


@app.route('/analyze_app', methods=['GET', 'POST'])
def analyze_app():
    figures = ['histogram', 'box', 'corr', 'scatter']
    response = "class"
    data_name = ''
    data_path = ''
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    if request.method == 'POST':
        data_name = request.form['submit']
        data_path = os.path.join(app.config['UPLOAD_DIR'], data_name)

    return render_template(
            'analyzeData.html',
            files=dirs,
            figures=report_analyze(figures, response, data_path, data_name),
            data_name=data_name)

########################### End Analyze Button ##################################

########################### Start Model Button ##################################
@app.route('/model_base', methods=['GET', 'POST'])
def model_base():
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    return render_template('models.html', files=dirs)


@app.route('/model_app', methods=['GET', 'POST'])
def model_app():
    response = "class"
    data_name = ''
    data_path = ''
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    if request.method == 'POST':
        data_name = request.form['submit']
        data_path = os.path.join(app.config['UPLOAD_DIR'], data_name)

    return render_template(
            'models.html',
            files=dirs,
            report=report_model(response, data_path, data_name),
            data_name=data_name)

########################### End Model Button ##################################

########################### Start Improve Button ##################################
@app.route('/improve_base', methods=['GET', 'POST'])
def improve_base():
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    return render_template('improve.html', files=dirs)

@app.route('/improve_app', methods=['GET', 'POST'])
def improve_app():
    response = "class"
    data_name = ''
    data_path = ''
    dirs = os.listdir(app.config['UPLOAD_DIR'])
    if request.method == 'POST':
        data_name = request.form['submit']
        data_path = os.path.join(app.config['UPLOAD_DIR'], data_name)

    return render_template(
            'improve.html',
            files=dirs,
            report=report_improve(response, data_path, data_name),
            data_name=data_name)

########################### End Improve Button ##################################

########################### Start Model Button ##################################
@app.route('/market_base', methods=['GET', 'POST'])
def market_base():
    dirs = os.listdir(app.config['MARKET_DIR'])
    return render_template('market.html', files=dirs)


@app.route('/market_app', methods=['GET', 'POST'])
def market_app():
    response = "class"
    data_name = ''
    data_path = ''
    dirs = os.listdir(app.config['MARKET_DIR'])
    if request.method == 'POST':
        data_name = request.form['submit']
        # data_path = os.path.join(app.config['MARKET_DIR'], data_name)
    return render_template(
            'market.html',
            files=dirs,
            report=report_market(data_name),
            data_name=data_name)

########################### End Market Button ##################################

# @app.route('/prediction', methods=['GET', 'POST'])
# def prediction():
    # attributes = []
    # dirs = os.listdir(app.config['UPLOAD_DIR'])
    # data_class = 'class'
    # file_name = 'iris.csv'
    # filepath = os.path.join(app.config['UPLOAD_DIR'], file_name)
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
