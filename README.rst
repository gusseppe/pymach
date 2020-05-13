===============================
pymach
===============================

Pymach is a tool to accelerate the development of models based on Machine Learning, check if there are patterns to exploit on a dataset. It aims to develop a first roadmap that gives an idea of the problem we are dealing with. The following pictures best explain the concept behind pymach.


Installation
------------
**Run using google colab:**

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/drive/1eI59Mud0oczOl6UMmTocA7vN1x0tOC3V?usp=sharing
   


**Using conda**

* conda create -n pymach python=3.6

* source activate pymach

* git clone https://github.com/gusseppe/pymach

* cd pymach

* pip install -r requirements.txt

* cd pymach

* python index.py


**Using Docker**

* git clone https://github.com/gusseppe/pymach
* cd pymach/pymach
* docker run -it -p 9080:9088 -v ${PWD}:/app gusseppe/pymach python index.py
* Open a browser and enter http://localhost:9080

You can modify the files in your current folder and see the modification in your container.

Define
--------
.. image:: https://github.com/gusseppe/pymach/blob/master/examples/define.png

Analyze
--------

.. image:: https://github.com/gusseppe/pymach/blob/master/examples/analyze1.png

.. image:: https://github.com/gusseppe/pymach/blob/master/examples/analyze2.png

Model
--------

.. image:: https://github.com/gusseppe/pymach/blob/master/examples/model1.png

.. image:: https://github.com/gusseppe/pymach/blob/master/examples/model2.png

Predict
--------

.. image:: https://cloud.githubusercontent.com/assets/6261900/23687975/63a71f9c-037f-11e7-828f-45725a0fafe1.png

* Free software: MIT license
* Documentation: https://pymach.readthedocs.io.


Features
--------

* TODO

Credits
---------

This package was created using Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

