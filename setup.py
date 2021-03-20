#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    #'Click>=6.0',
    # TODO: put package requirements here
     'appdirs==1.4.3',
     'click==6.7',
     'colorlover==0.2.1',
     'cufflinks==0.8.2',
     'cycler==0.10.0',
     'dash==0.36.0rc1',
     'dash-core-components==0.41.0',
     'dash-html-components==0.13.4',
     'dash-renderer==0.17.0rc1',
     'dash-table==3.1.11',
     'dash-table-experiments==0.6.0',
     'decorator==4.3.0',
     'Flask==0.12.1',
     'Flask-Caching==1.4.0',
     'Flask-Compress==1.4.0',
     'h5py==2.9.0',
     'hickle==3.3.2',
     'ipython==6.0.0',
     'ipython-genutils==0.2.0',
     'itsdangerous==0.24',
     'jedi==0.10.2',
     'Jinja2==2.11.3',
     'jsonschema==2.6.0',
     'jupyter-core==4.3.0',
     'lightgbm==2.2.2',
     'MarkupSafe==1.0',
     'matplotlib==2.0.2',
     'MyApplication==0.1.0',
     'nbformat==4.3.0',
     'networkx==2.2',
     'numpy==1.12.1',
     'packaging==16.8',
     'pandas==0.20.1',
     'pexpect==4.2.1',
     'pickleshare==0.7.4',
     'pkg-resources==0.0.0',
     'plotly==2.7.0',
     'pluggy==0.5.1',
     'prompt-toolkit==1.0.14',
     'ptyprocess==0.5.1',
     'py==1.4.34',
     'Pygments==2.2.0',
     'pyparsing==2.2.0',
     'python-dateutil==2.6.0',
     'pytz==2017.2',
     'requests==2.14.2',
     'retrying==1.3.3',
     'scikit-learn==0.18.1',
     'scipy==0.19.0',
     'seaborn==0.9.0',
     'simplegeneric==0.8.1',
     'six==1.10.0',
     'traitlets==4.3.2',
     'typing==3.6.1',
     'virtualenv==15.1.0',
     'wcwidth==0.1.7',
     'Werkzeug==0.12.1',
     'xgboost==0.81',

]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='pymach',
    version='0.1.0',
    description="Pymach is a tool to accelerate the development of models based on Machine Learning, looking for the best model for your data.",
    long_description=readme + '\n\n' + history,
    author="Gusseppe Bravo Rocca",
    author_email='gbravor@uni.pe',
    url='https://github.com/gusseppe/pymach',
    packages=[
        'pymach',
    ],
    package_dir={'pymach':
                 'pymach'},
    entry_points={
        'console_scripts': [
            'pymach=pymach.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='pymach',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
