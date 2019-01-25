# -*- coding: utf-8 -*-
import json
import base64
import datetime
import io
import os
import glob
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html

from collections import OrderedDict
from dash.dependencies import Input, Output, State
from app import app, indicator
from core import define
from core import analyze
from core import explain

current_path = os.getcwd()
MARKET_PATH = os.path.join(current_path, 'market')


# @functools.lru_cache(maxsize=32)
def parse_contents(contents, filename, date):
    df = None
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
    print(files_dict)
    return files_dict

layout = [


    html.Div([
        ########################### Indicators I ##################################
        html.Div(
            [
                indicator(
                    "#119DFF", "Type of Problem", "problem_type_analyze_indicator"
                ),
                indicator(
                    "#119DFF", "Filename", "filename_analyze_indicator"
                ),
                html.Div(
                    [
                        html.P(
                            'Uploaded files',
                            className="twelve columns indicator_text"
                        ),
                        dcc.Dropdown(
                            id="files_uploaded_analyze_dropdown",
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
                "#00cc96", "Number of samples", "n_samples_analyze_indicator"
            ),
            indicator(
                "#119DFF", "Number of features", "n_features_analyze_indicator"
            ),
            indicator(
                "#EF553B",
                "Size in memory",
                "size_analyze_indicator",
            ),
        ],
        className="row",
        style={"marginBottom": "10"},
    ),
    ########################### Indicators II ##################################
    html.Div(
        [
            html.Div(
                [
                    html.P(
                        'Categorical features',
                        className="twelve columns indicator_text"
                    ),
                    dcc.Dropdown(
                        id="cat_features_analyze_dropdown",
                        options=[],
                        value=[],
                        multi=True,
                        clearable=False,
                        searchable=False,
                        disabled=True,
                        # className='indicator_value'
                    ),
                ],
                className="four columns indicator",
            ),
            html.Div(
                [
                    html.P(
                        'Numerical features',
                        className="twelve columns indicator_text"
                    ),
                    dcc.Dropdown(
                        id="num_features_analyze_dropdown",
                        options=[],
                        value=[],
                        multi=True,
                        clearable=False,
                        searchable=False,
                        disabled=True,
                        # className='indicator_value'
                    ),
                ],
                className="four columns indicator",
            ),
            html.Div(
                [
                    html.P(
                        'Response variable',
                        className="twelve columns indicator_text"
                    ),
                    dcc.Dropdown(
                        id="response_analyze_dropdown",
                        options=[],
                        value="",
                        clearable=False,
                        searchable=False,
                        disabled=True,
                        # className='indicator_value'
                    ),
                ],
                className="four columns indicator",
            ),
        ],
        className="row",
    ),
    ########################### Table ##################################
    # html.Div(
    #     [
    #
    #         # html.Div(
    #         #     [
    #         #         #                     html.P("Agrupacion por cantidad de CPE"),
    #         #
    #         #         # html.Div(id='output-data-upload'),
    #         #         dash_table.DataTable(
    #         #             id='table_analyze',
    #         #             # data=dff.to_dict('rows'),
    #         #             # columns=[
    #         #             #     {'name': i, 'id': i, 'deletable': True} for i in sorted(dff.columns)
    #         #             # ],
    #         #             style_header={
    #         #                 # 'backgroundColor': 'white',
    #         #                 'backgroundColor': '#2a3f5f',
    #         #                 'color': 'white',
    #         #                 'fontWeight': 'bold'
    #         #             },
    #         #             style_cell_conditional=[{
    #         #                 'if': {'row_index': 'odd'},
    #         #                 'backgroundColor': 'rgb(248, 248, 248)'
    #         #             }],
    #         #             pagination_settings={
    #         #                 'current_page': 0,
    #         #                 'page_size': 10
    #         #             },
    #         #             pagination_mode='be',
    #         #
    #         #             sorting='be',
    #         #             sorting_type='single',
    #         #             style_table={'overflowX': 'scroll'},
    #         #             sorting_settings=[]
    #         #         )
    #         #
    #         #     ],
    #         #     className="twelve columns",
    #         # ),
    #         html.Div(id='intermediate_value_analyze', style={'display': 'none'})
    #
    #     ],
    #     className="row",
    #     style={"marginBottom": "10", "marginTop": "10"},
    # ),
    html.Div(id='hidden_analyze_div', style={'display': 'none'}),


    ########################### Plots ##################################
    html.Div(
        [

            html.Div(
                [
                    #                     html.P("Agrupacion por cantidad de CPE"),

                    dcc.Graph(
                        id="hist_graph",
                        #                         figure=grafo_3d_cpe(1,2),
                        style={"height": "200%", "width": "100%"},
                        config=dict(displayModeBar=True,
                                    showLink=False),
                    ),

                ],
                className="eight columns",
            ),
            html.Div(
                [
                    #                     html.P("Agrupacion por cantidad de CPE"),

                    # dcc.Graph(
                    #     id="box_graph",
                    #     #                         figure=grafo_3d_cpe(1,2),
                    #     style={"height": "200%", "width": "100%"},
                    #     config=dict(displayModeBar=True,
                    #                 showLink=False),
                    # ),
                    dcc.Markdown(
                        # children='''
                        # ''',
                        id='hist_markdown',
                        className="twelve columns indicator_text")

                ],
                className="four columns chart_div",
            ),
        ],
        className="row",
        style={"marginBottom": "10", "marginTop": "10"},
    ),
    html.Div([
            html.Div(
                [
                    #                     html.P("Agrupacion por cantidad de CPE"),

                    dcc.Graph(
                        id="box_graph",
                        #                         figure=grafo_3d_cpe(1,2),
                        style={"height": "200%", "width": "100%"},
                        config=dict(displayModeBar=True,
                                    showLink=False),
                    ),

                ],
                className="eight columns",
            ),
            html.Div(
                [
                    #                     html.P("Agrupacion por cantidad de CPE"),

                    # dcc.Graph(
                    #     id="box_graph",
                    #     #                         figure=grafo_3d_cpe(1,2),
                    #     style={"height": "200%", "width": "100%"},
                    #     config=dict(displayModeBar=True,
                    #                 showLink=False),
                    # ),
                    dcc.Markdown(
                        # children='''
                        # ''',
                        id='box_markdown',
                        className="twelve columns indicator_text")

                ],
                className="four columns chart_div",
            ),
        ],
        className="row",
        style={"marginBottom": "10", "marginTop": "10"},
    ),
    html.Div([
            html.Div(
                [
                    #                     html.P("Agrupacion por cantidad de CPE"),

                    dcc.Graph(
                        id="corr_graph",
                        #                         figure=grafo_3d_cpe(1,2),
                        style={"height": "200%", "width": "100%"},
                        config=dict(displayModeBar=True,
                                    showLink=False),
                    ),

                ],
                className="eight columns",
            ),
            html.Div(
                [
                    #                     html.P("Agrupacion por cantidad de CPE"),

                    # dcc.Graph(
                    #     id="box_graph",
                    #     #                         figure=grafo_3d_cpe(1,2),
                    #     style={"height": "200%", "width": "100%"},
                    #     config=dict(displayModeBar=True,
                    #                 showLink=False),
                    # ),
                    dcc.Markdown(
                        # children='''
                        # ''',
                        id='corr_markdown',
                        className="twelve columns indicator_text")

                ],
                className="four columns chart_div",
            ),
        ],
        className="row",
        style={"marginBottom": "10", "marginTop": "10"},
    ),
    html.Div([
            html.Div(
                [
                    #                     html.P("Agrupacion por cantidad de CPE"),

                    dcc.Graph(
                        id="scatter_graph",
                        #                         figure=grafo_3d_cpe(1,2),
                        style={"height": "200%", "width": "100%"},
                        config=dict(displayModeBar=True,
                                    showLink=False),
                    ),

                ],
                className="eight columns",
            ),
            html.Div(
                [
                    #                     html.P("Agrupacion por cantidad de CPE"),

                    dcc.Markdown(
                        # children='''
                        # ''',
                        id='scatter_markdown',
                        className="twelve columns indicator_text")

                ],
                className="four columns chart_div",
            ),

        ],
        className="row",
        style={"marginBottom": "10", "marginTop": "10"},
    ),
    html.Div([
        html.Div(
            [
                #                     html.P("Agrupacion por cantidad de CPE"),

                dcc.Graph(
                    id="parallel_graph",
                    #                         figure=grafo_3d_cpe(1,2),
                    style={"height": "200%", "width": "100%"},
                    config=dict(displayModeBar=True,
                                showLink=False),
                ),

            ],
            className="eight columns",
        ),
        html.Div(
            [
                #                     html.P("Agrupacion por cantidad de CPE"),

                dcc.Markdown(
                    # children='''
                    # ''',
                    id='parallel_markdown',
                    className="twelve columns indicator_text")

            ],
            className="four columns chart_div",
        ),

    ],
        className="row",
        style={"marginBottom": "10", "marginTop": "10"},
    ),

    html.Hr(),
    ########################### Save file ##################################
    html.Div(
        [
            html.Div(
                [
                    # submit button
                    html.Span(
                        "Save",
                        id="save_analyze_button",
                        n_clicks=0,
                        # className="btn btn-primary"
                        className="button button--primary add"
                    ),
                ],
                #                 className="six columns",
                className="two columns",
                # style={"paddingRight": "15"},
            ),
            html.Div(
                id='save_file_div',
                className="two columns",
                # style={"paddingRight": "15"},
            ),
        ],
        className="row",
        style={"marginBottom": "10", "marginTop": "10"},
    ),
]


# @app.callback(Output('hidden_analyze_div', 'children'),
#               [Input('files_uploaded_analyze_dropdown', 'value')])
# def store_table(uploaded_file):
#     if uploaded_file != '':
#         print(uploaded_file)
#         chosen_file_path = os.path.join(MARKET_PATH, uploaded_file)
#         df = pd.read_csv(chosen_file_path)
#         filename = uploaded_file
#         date = datetime.datetime.now()
#
#         info = {
#             'df': df.to_json(orient='split', date_format='iso'),
#             'filename': filename,
#             'date': json.dumps(date, indent=4, sort_keys=True, default=str),
#         }
#         return json.dumps(info)

@app.callback(
    [Output('filename_analyze_indicator', 'children'),
     Output('n_samples_analyze_indicator', 'children'),
     Output('n_features_analyze_indicator', 'children'),
     Output('size_analyze_indicator', 'children'),
     Output('cat_features_analyze_dropdown', 'options'),
     Output('cat_features_analyze_dropdown', 'value'),
     Output('num_features_analyze_dropdown', 'options'),
     Output('num_features_analyze_dropdown', 'value'),
     Output('response_analyze_dropdown', 'options'),
     Output('response_analyze_dropdown', 'value'),
     Output('problem_type_analyze_indicator', 'children')],
    [Input('files_uploaded_analyze_dropdown', 'value')])
def update_metadata_analyze(uploaded_file):
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

    out = tuple([filename, n_samples, n_features, size,
                 cat_options, cat_features, num_options, num_features,
                 response_options, response, problem_type])
    return out


@app.callback([Output('hist_graph', 'figure'), Output('hist_markdown', 'children'),
               Output('box_graph', 'figure'), Output('box_markdown', 'children'),
               Output('corr_graph', 'figure'), Output('corr_markdown', 'children'),
               Output('scatter_graph', 'figure'), Output('scatter_markdown', 'children'),
               Output('parallel_graph', 'figure'), Output('parallel_markdown', 'children')],
              [Input('files_uploaded_analyze_dropdown', 'value')])
def show_plots_analyze(uploaded_file):
    if uploaded_file != '':
        metadata_folder = os.path.join(MARKET_PATH, uploaded_file.replace('.csv', ''))
        metadata_filename = uploaded_file.replace('.csv', '') + '_meta.json'
        metadata_path = os.path.join(metadata_folder, metadata_filename)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        filename_path = os.path.join(metadata_folder, uploaded_file)
        df = pd.read_csv(filename_path)
        definer = define.Define(df=df, num_features=metadata['num_features'],
                                cat_features=metadata['cat_features'],
                                response=metadata['response']).pipeline()

        analyzer = analyze.Analyze(definer)
        explainer = explain.Explain(definer)
        folder_path = os.path.join(metadata_folder, 'analyze')
        dict_figures = OrderedDict()
        figures = ['hist', 'box', 'corr', 'scatter', 'parallel']

        if os.path.exists(folder_path):
            print(f'plots already exist.')

            for fig in figures:
                # path_fig = os.path.join(folder_path, fig+'_fig.json')
                path_explain = os.path.join(folder_path, fig+'_explain')
                # dict_figures[fig+'_fig'] = tools.plotlyfromjson(path_fig)
                dict_figures[fig+'_fig'] = analyzer.plot(fig)
                with open(path_explain) as f:
                    dict_figures[fig+'_explain'] = f.read()

        else:
            # to save the figures
            os.makedirs(folder_path)

            for fig in figures:
                dict_figures[fig+'_fig'] = analyzer.plot(fig)
                dict_figures[fig+'_explain'] = explainer.explain_analyze(fig)
                plot_path = os.path.join(folder_path, fig+'_fig.json')
                explain_path = os.path.join(folder_path, fig+'_explain')
                analyzer.save(plot_path)
                explainer.save(explain_path)


        out = []
        for e in figures:
            out.append(dict_figures[e+'_fig'])
            out.append(dict_figures[e+'_explain'])

        out = tuple(out)
        return out
        # return dict_figures['hist_fig'], dict_figures['hist_explain'], \
        #        dict_figures['hist_fig'], dict_figures['hist_explain'], \
        #            box_fig, result_box, \
        #        corr_fig, result_corr, \
        #        scatter_fig, result_scatter, \
        #        parallel_fig, result_parallel,
    else:
        return tuple(None for _ in range(10))


