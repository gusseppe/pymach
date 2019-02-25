# -*- coding: utf-8 -*-
import json
import base64
import datetime
import io
import os
import glob

import pandas as pd
import numpy as np
import dash_table
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from app import app, indicator
from core import define
from core import prepare
from core import fselect
from core import evaluate


current_path = os.getcwd()
MARKET_PATH = os.path.join(current_path, 'market')


# @functools.lru_cache(maxsize=32)
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df, filename, datetime.datetime.fromtimestamp(date)

def list_files_market():
    files_path = glob.glob(os.path.join(MARKET_PATH, '*', '*.csv'))
    # folders = [folder for folder in os.listdir(MARKET_PATH)]
    files = [os.path.basename(file) for file in files_path]
    files_dict = [
        {"label": file, "value": file} for file in files
    ]
    return files_dict


def list_models(problem_type='classification'):
    # models = ['GradientBoostingClassifier', 'ExtraTreesClassifier',
    #           'RandomForestClassifier', 'DecisionTreeClassifier',
    #           'LinearDiscriminantAnalysis', 'SVC', 'KNeighborsClassifier',
    #           'LogisticRegression', 'AdaBoostClassifier', 'VotingClassifier',
    #           'GaussianNB', 'MLPClassifier']
    models = []
    if problem_type == 'classification':
        models = ['AdaBoostClassifier', 'GradientBoostingClassifier',
                  'BaggingClassifier', 'RandomForestClassifier',
                  'KNeighborsClassifier', 'DecisionTreeClassifier',
                  'MLPClassifier', 'ExtraTreesClassifier', 'SVC',
                  'LinearDiscriminantAnalysis', 'GaussianNB',
                  'LogisticRegression', 'VotingClassifier',
                  'XGBoostClassifier', 'LGBMClassifier']
    elif problem_type == 'regression':
        models = ['AdaBoostRegressor', 'GradientBoostingRegressor',
                  'BaggingRegressor', 'RandomForestRegressor',
                  'KNeighborsRegressor', 'DecisionTreeRegressor',
                  'MLPRegressor', 'ExtraTreesRegressor', 'SVR',
                  'LinearRegression', 'BayesianRidge',
                  'XGBoostRegressor', 'LGBMRegressor']

    # files_dict = [
    #     {"label": m, "value": m} for m in models
    # ]
    return models

def list_prepare():
    preparers = ['MinMaxScaler', 'Normalizer',
              'StandardScaler', 'RobustScaler']
    files_dict = [
        {"label": m, "value": m} for m in preparers
    ]
    return files_dict

def list_select():
    selectors = ['SelectKBest', 'PrincipalComponentAnalysis',
                 'ExtraTrees', ]
    files_dict = [
        {"label": m, "value": m} for m in selectors
    ]
    return files_dict

layout = [


    html.Div([
        ########################### Indicators I ##################################
        html.Div(
            [
                indicator(
                    "#119DFF", "Type of Problem", "problem_type_tune_indicator"
                ),
                indicator(
                    "#119DFF", "Filename", "filename_tune_indicator"
                ),
                html.Div(
                    [
                        html.P(
                            'Uploaded files',
                            className="twelve columns indicator_text"
                        ),
                        dcc.Dropdown(
                            id="files_uploaded_tune_dropdown",
                            options=list_files_market(),
                            value="",
                            clearable=False,
                            searchable=False,
                            className='indicator_value'
                        ),
                    ],
                    className="four columns indicator",
                ),
                # indicatorii(
                #     "#EF553B",
                #     "Size",
                #     "right_leads_indicator",
                # ),
            ],
            className="row",
        ),
        # dash_table.DataTable(id='datatable-upload-container'),
        # dcc.Graph(id='datatable-upload-graph')
    ],
        className="row",
        style={"marginBottom": "10"},
    ),
    ########################### Indicators II ##################################
    html.Div(
        [
            indicator(
                "#00cc96", "Number of samples", "n_samples_tune_indicator"
            ),
            indicator(
                "#119DFF", "Number of features", "n_features_tune_indicator"
            ),
            indicator(
                "#EF553B",
                "Size in memory",
                "size_tune_indicator",
            ),
        ],
        className="row",
        style={"marginBottom": "10"},
    ),
    ########################### Indicators III ##################################
    html.Div([
        html.Div(
            [
                html.P(
                    'Methods',
                    className="twelve columns indicator_text"
                ),
                html.Hr(),
                dcc.Dropdown(
                    id="prepare_tune_dropdown",
                    options=list_prepare(),
                    # options = [
                    #     {"label": 'model'+str(file), "value": file} for file in range(20)
                    # ],
                    value=list_prepare(),
                    # value=[],
                    clearable=False,
                    searchable=False,
                    disabled=True,
                    className="twelve columns indicator_text",
                    multi=True,
                    # className="four columns chart_div"
                    # className='indicator_value'
                ),
                # dcc.Checklist(
                #     id='models_checklist',
                #     options=[
                #         {'label': 'New York City', 'value': 'NYC'},
                #         {'label': 'Montréal', 'value': 'MTL'},
                #         {'label': 'San Francisco', 'value': 'SF'}
                #     ],
                #     values=['MTL', 'SF']
                # )

            ],
            className="four columns chart_div",
        ),
        html.Div(
            [
                html.P(
                    'Methods',
                    className="twelve columns indicator_text"
                ),
                html.Hr(),
                dcc.Dropdown(
                    id="selector_tune_dropdown",
                    options=list_select(),
                    value=list_select(),
                    # value=[],
                    clearable=False,
                    searchable=False,
                    disabled=True,
                    className="twelve columns indicator_text",
                    multi=True,
                    # className="four columns chart_div"
                    # className='indicator_value'
                ),

            ],
            className="four columns chart_div",
        ),
        html.Div(
            [
                html.P(
                    'Modelers',
                    className="twelve columns indicator_text"
                ),
                html.Hr(),
                dcc.Dropdown(
                    id="models_tune_dropdown",
                    # options=[],
                    options=[],
                    # options = [
                    #     {"label": 'model'+str(file), "value": file} for file in range(20)
                    # ],
                    value=[],
                    # value=[],
                    clearable=False,
                    searchable=False,
                    # disabled=True,
                    className="twelve columns indicator_text",
                    multi=True,
                    # className="four columns chart_div"
                    # className='indicator_value'
                ),
            ],
            className="four columns chart_div",
        ),

    ],
        className="row",
        style={"marginBottom": "10", "marginTop": "10"},
    ),

    html.Hr(),
    ########################### Table results ##################################
    html.Div(
        [

            html.Div(
                [
                    # html.P("Agrupacion por cantidad de CPE"),

                    # html.P(
                    #     'Ranking Models - CV=10 - Train dataset',
                    #     className="twelve columns indicator_text"
                    # ),
                    # html.Div(id='output-data-upload'),
                    dash_table.DataTable(
                        id='results_tune_table',
                        # data=dff.to_dict('rows'),
                        # columns=[
                        #     {'name': i, 'id': i, 'deletable': True} for i in sorted(dff.columns)
                        # ],
                        style_header={
                            # 'backgroundColor': 'white',
                            'backgroundColor': '#248f24',
                            'color': 'white',
                            'fontWeight': 'bold'
                        },
                        style_cell_conditional=[{
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }],
                        row_selectable="multi",
                        row_deletable=True,
                        selected_rows=[0],
                        # pagination_settings={
                        #     'current_page': 0,
                        #     'page_size': 10
                        # },
                        # pagination_mode='be',

                        # sorting='be',
                        # sorting_type='single',
                        # sorting_settings=[]
                    )

                ],
                className="four columns",
            ),
            html.Div(id='metrics_tune_graph',
                     className="four columns",
                     ),
            html.Div(id='fi_tune_graph',
                     className="four columns"),
            # style={"paddingRight": "15"},
            # html.Div(
            #     [
            #         html.P(
            #             'Metrics',
            #             className="twelve columns indicator_text"
            #         ),
            #         dcc.Graph(
            #             id="metrics_tune_graph",
            #             style={"height": "200%", "width": "100%"},
            #             config=dict(displayModeBar=True,
            #                         showLink=False),
            #         ),
            #
            #     ],
            #     className="four columns",
            # ),
            # html.Div(
            #     [
            #         html.P(
            #             'Feature Importance',
            #             className="twelve columns indicator_text"
            #         ),
            #         dcc.Graph(
            #             id="importance_tune_graph",
            #             style={"height": "200%", "width": "100%"},
            #             config=dict(displayModeBar=True,
            #                         showLink=False),
            #         ),
            #
            #     ],
            #     className="four columns",
            # ),

        ],
        className="row",
        style={"marginBottom": "10", "marginTop": "10"},
    ),
    ########################### Button ##################################
    html.Div(id='hidden_tune_div', style={'display': 'none'}),
    # html.Div(id='hidden_model_div'),
    html.Div(
        [
            html.Div(
                [
                    # submit button
                    html.Span(
                        "Tune hyperparameters",
                        id="run_tune_button",
                        n_clicks=0,
                        # className="btn btn-primary"
                        className="button button--primary add"
                    ),
                    #                     dcc.Input(
                    #                         id="output_chatbot",
                    #                         placeholder="Respuesta de Adaline: ",
                    #                         type="text",
                    #                         value="",
                    #                         disabled=True,
                    #                         style={"width": "100%"},
                    #                     ),
                ],
                #                 className="six columns",
                className="two columns",
                # style={"paddingRight": "15"},
            ),
            html.Div(
                id='save_file_tune_div',
                className="two columns",
                # style={"paddingRight": "15"},
            ),
        ],
        className="row",
        style={"marginBottom": "10", "marginTop": "10"},
    ),
]


@app.callback(
    [Output('filename_tune_indicator', 'children'),
     Output('n_samples_tune_indicator', 'children'),
     Output('n_features_tune_indicator', 'children'),
     Output('size_tune_indicator', 'children'),
     Output('cat_features_tune_dropdown', 'options'),
     Output('cat_features_tune_dropdown', 'value'),
     Output('num_features_tune_dropdown', 'options'),
     Output('num_features_tune_dropdown', 'value'),
     Output('response_tune_dropdown', 'options'),
     Output('response_tune_dropdown', 'value'),
     Output('problem_type_tune_indicator', 'children'),
     Output('models_tune_dropdown', 'options'),
     Output('models_tune_dropdown', 'value')],
    [Input('files_uploaded_tune_dropdown', 'value')])
def update_metadata_tune(uploaded_file):
    if uploaded_file != '':
        metadata_folder = os.path.join(MARKET_PATH, uploaded_file.replace('.csv', ''))
        metadata_filename = uploaded_file.replace('.csv', '') + '_meta.json'
        metadata_path = os.path.join(metadata_folder, metadata_filename)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        filename = uploaded_file
        n_samples = metadata['n_samples']
        n_features = metadata['n_features']
        num_features = metadata['num_features']
        cat_features = metadata['cat_features']
        response = metadata['response']
        problem_type = metadata['problem_type']
        size = metadata['size']

    else:
        filename = ''
        n_samples = ''
        n_features = ''
        size = ''
        cat_features = []
        num_features = []
        response = ''
        problem_type = ''

    num_options = [
        {"label": file, "value": file} for file in num_features
    ]
    cat_options = [
        {"label": file, "value": file} for file in cat_features
    ]
    response_options = [
        {"label": file, "value": file} for file in [response]
    ]
    models_options = [
        {"label": file, "value": file} for file in list_models(problem_type)
    ]
    models_value = np.random.choice(list_models(problem_type), 3, replace=False)

    out = tuple([filename, n_samples, n_features, size,
                 cat_options, cat_features, num_options, num_features,
                 response_options, response, problem_type,
                 models_options, models_value])
    return out


@app.callback([Output('results_tune_table', 'data'),
               Output('results_tune_table', 'columns')],
              [Input('files_uploaded_tune_dropdown', 'value')])
def show_out_models_tune(uploaded_file):
    if uploaded_file != '':

        print('tuning>>>>>', uploaded_file)
        metadata_folder = os.path.join(MARKET_PATH, uploaded_file.replace('.csv', ''))
        metadata_filename = uploaded_file.replace('.csv', '') + '_meta.json'
        metadata_path = os.path.join(metadata_folder, metadata_filename)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        folder_path = os.path.join(metadata_folder, 'model')
        path_report = os.path.join(folder_path, 'report.csv')
        # path_raw_report = os.path.join(folder_path, 'raw_report.json')

        report = pd.read_csv(path_report)
        # report.sort_values(['Mean'], ascending=False, inplace=True)
        report['Mean'] = np.round(report['Mean'], 3)
        report['STD'] = np.round(report['STD'], 3)

        columns_table=[
                    {'name': i, 'id': i, 'deletable': True} for i in report.columns
                ]

        # name = 'AdaBoostClassifier'
        # return tuple([plot, report.to_dict('rows'), columns_table, evaluator.plot_metrics(name)])
        return tuple([report.to_dict('rows'), columns_table])

    return tuple([None for _ in range(2)])


@app.callback(Output('models_tune_dropdown', 'value'),
              [Input('autocomplete_models_tune_button', 'n_clicks')],
              [State('problem_type_tune_indicator', 'children')])
def autocomplete_models_tune(n_clicks, problem_type):
    if n_clicks > 0:
        models_value = list_models(problem_type)
        print(models_value)

        return models_value

        # return tuple([None for _ in range(3)])


@app.callback(
    [Output('metrics_tune_graph', "children"),
     Output('fi_tune_graph', "children")],
    [Input('results_tune_table', "derived_virtual_data"),
     Input('results_tune_table', "derived_virtual_selected_rows")],
    [State('files_uploaded_tune_dropdown', 'value')])
def show_confmatrix_featimportance_tune(rows, selected_rows, uploaded_file):
    # print('selected>>>', selected_rows)
    if selected_rows is None:
        selected_rows = []

    if len(selected_rows) > 0:
        metadata_folder = os.path.join(MARKET_PATH, uploaded_file.replace('.csv', ''))
        folder_path = os.path.join(metadata_folder, 'model')
        path_metrics = os.path.join(folder_path, 'metrics.json')
        path_fi = os.path.join(folder_path, 'feature_importance.json')
        with open(path_metrics, 'r') as f:
            metrics = json.load(f)

        with open(path_fi, 'r') as f:
            fi = json.load(f)
        for k, v in fi.items():
            fi[k] = pd.DataFrame(v)

        list_figs_metrics = []
        list_figs_fi = []
        for index in selected_rows:
            print(rows[index])
            model_name = rows[index]['Model']
            evaluator = evaluate.Evaluate()
            evaluator.problem_type = 'classification'

            # metrics
            evaluator.metrics = metrics
            fig_metrics = evaluator.plot_metrics(model_name)
            list_figs_metrics.append(fig_metrics)

            # feature importance
            evaluator.feature_importance = fi
            fig_fi = evaluator.plot_feature_importance(model_name)
            list_figs_fi.append(fig_fi)

        return_figs_metrics = [dcc.Graph(figure=f) for f in list_figs_metrics]
        return_figs_fi = [dcc.Graph(figure=f) for f in list_figs_fi]

        return html.Div(return_figs_metrics), html.Div(return_figs_fi)
    else:
        return tuple([None, None])
