import os
from datetime import datetime
from flask import Flask, render_template, jsonify, redirect, url_for, request


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = ['txt', 'csv']

@app.route('/')
def home():
   return render_template("home.html")

@app.route('/defineData', methods = ['GET', 'POST'])
def defineData():
    files = []
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file)
        f = open(filepath, 'r')
        files.append(f)
    return render_template('uploadData.html', filenames = dirs, files = files)

@app.route('/storeData', methods = [ 'GET', 'POST'])
def guardarData():
	if request.method == 'POST':
		file = request.files['file']
		if file and allowed_file(file.filename):
			now = datetime.now()
			filename = os.path.join(app.config['UPLOAD_FOLDER'], "%s" % (file.filename))
			file.save(filename)
			return jsonify({ "success": True })
		return redirect(url_for('defineData'))
	else:
		return redirect(url_for('home'))

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/analyzeData', methods = ['GET', 'POST'])
def analyzeData():
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    data_class = request.form['class']
    filename = request.form['filename']
    print('this is just the beginning ', request.form)
    return render_template('analyzeData.html', data_class = data_class, filename = filename)

@app.route('/models', methods = ['GET', 'POST'])
def models():
	dirs = os.listdir(app.config['UPLOAD_FOLDER'])
	return render_template('uploadData.html', files = dirs)

@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
	dirs = os.listdir(app.config['UPLOAD_FOLDER'])
	return render_template('uploadData.html', files = dirs)

@app.route('/table', methods = ['GET', 'POST'])
def table():
    dirs = os.listdir(app.config['UPLOAD_FOLDER'])
    filename = request.form['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f = open(filepath, 'r')
    return render_template('showTable.html', file = f, filename = filename)

if __name__ == '__main__':
   app.run( debug = True)
