# -*- coding: utf-8 -*-
import json
import base64
import datetime
import io
import os
import glob

import pandas as pd
import dash_table
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State
from app import app, indicator
from core import define

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

layout = [


    html.Div([
        ########################### Indicators I ##################################
        html.Div(
            [
                # indicator(
                #     "#00cc96", "Number of rows", "left_leads_indicator"
                # ),
                # html.Iframe(id='iframe-upload',src=f'/upload'),
                html.Div(
                    [
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files - Max 1Mb')
                            ]),
                            style={
                                'width': '100%', 'height': '100%', 'lineHeight': '80px',
                                'borderWidth': '2px', 'borderStyle': 'dashed',
                                'borderRadius': '5px', 'textAlign': 'center', 'margin': '0px'
                            },
                            max_size=1000e3, # 100kb
                            multiple=True,
                        ),
                    ],
                    className='four columns indicator'
                ),
                indicator(
                    "#119DFF", "Filename", "filename_indicator"
                ),
                html.Div(
                    [
                        html.P(
                            'Uploaded files',
                            className="twelve columns indicator_text"
                        ),
                        dcc.Dropdown(
                            id="files_uploaded_dropdown",
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
                "#00cc96", "Number of samples", "n_samples_indicator"
            ),
            indicator(
                "#119DFF", "Number of features", "n_features_indicator"
            ),
            indicator(
                "#EF553B",
                "Size in memory",
                "size_indicator",
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
                    html.Hr(),
                    dcc.Dropdown(
                        id="cat_features_dropdown",
                        options=[],
                        value=[],
                        multi=True,
                        clearable=False,
                        searchable=False,
                        # className='indicator_value'
                    ),
                ],
                className="four columns chart_div",
            ),
            html.Div(
                [
                    html.P(
                        'Numerical features',
                        className="twelve columns indicator_text"
                    ),
                    html.Hr(),
                    dcc.Dropdown(
                        id="num_features_dropdown",
                        options=[],
                        value=[],
                        multi=True,
                        clearable=False,
                        searchable=False,
                        # className='indicator_value'
                    ),
                ],
                className="four columns chart_div",
            ),
            html.Div(
                [
                    html.P(
                        'Response variable',
                        className="twelve columns indicator_text"
                    ),
                    html.Hr(),
                    dcc.Dropdown(
                        id="response_dropdown",
                        options=[],
                        value="",
                        clearable=False,
                        searchable=False,
                        # className='indicator_value'
                    ),
                ],
                className="four columns indicator",
            ),
        ],
        className="row",
    ),
    ########################### Table ##################################
    html.Div(
        [

            html.Div(
                [
                    #                     html.P("Agrupacion por cantidad de CPE"),

                    # html.Div(id='output-data-upload'),
                    dash_table.DataTable(
                        id='table-paging-and-sorting',
                        # data=dff.to_dict('rows'),
                        # columns=[
                        #     {'name': i, 'id': i, 'deletable': True} for i in sorted(dff.columns)
                        # ],
                        style_header={
                            # 'backgroundColor': 'white',
                            'backgroundColor': '#2a3f5f',
                            'color': 'white',
                            'fontWeight': 'bold'
                        },
                        style_cell_conditional=[{
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }],
                        pagination_settings={
                            'current_page': 0,
                            'page_size': 10
                        },
                        pagination_mode='be',
                        style_table={'overflowX': 'scroll'},

                        sorting='be',
                        sorting_type='single',
                        sorting_settings=[]
                    )

                ],
                className="twelve columns",
            ),
            html.Div(id='intermediate-value', style={'display': 'none'})

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
                        id="save_button",
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
                id='save_file_div',
                className="two columns",
                # style={"paddingRight": "15"},
            ),
        ],
        className="row",
        style={"marginBottom": "10", "marginTop": "10"},
    ),
]

@app.callback(Output('intermediate-value', 'children'),
              [Input('upload-data', 'contents'),
               Input('files_uploaded_dropdown', 'value'),
               Input('upload-data', 'filename')],
              [State('upload-data', 'last_modified')])
def store_table_define(list_of_contents, uploaded_file, list_of_names, list_of_dates):
    # if a file is uploaded
    if list_of_contents is not None:
        print(list_of_names)
        # if uploaded_file == '':
        files = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]

        # print(files)
        # df, filename, date = parse_contents(list_of_contents, list_of_names, list_of_dates)
        df, filename, date = files[0]
        # children = [
        #     parse_contents(c, n, d) for c, n, d in
        #     zip(list_of_contents, list_of_names, list_of_dates)]
        # return children
        info = {
            'df': df.to_json(orient='split', date_format='iso'),
            'filename':filename,
            'date': json.dumps(date, indent=4, sort_keys=True, default=str),
        }
        return json.dumps(info)


    # if a file is chosen from the dropdown
    if uploaded_file != '':

        metadata_folder = os.path.join(MARKET_PATH, uploaded_file.replace('.csv', ''))
        chosen_file_path = os.path.join(metadata_folder, uploaded_file)
        df = pd.read_csv(chosen_file_path)
        filename = uploaded_file
        date = datetime.datetime.now()

        info = {
            'df': df.to_json(orient='split', date_format='iso'),
            'filename':filename,
            'date': json.dumps(date, indent=4, sort_keys=True, default=str),
        }
        return json.dumps(info)


@app.callback(
    [Output('table-paging-and-sorting', 'data'), Output('table-paging-and-sorting', 'columns'),
     Output('table-paging-and-sorting', 'style_data_conditional'),
     Output('filename_indicator', 'children'), Output('n_samples_indicator', 'children'),
     Output('n_features_indicator', 'children'), Output('size_indicator', 'children'),
     Output('cat_features_dropdown', 'options'), Output('cat_features_dropdown', 'value'),
     Output('num_features_dropdown', 'options'), Output('num_features_dropdown', 'value'),
     Output('response_dropdown', 'options'), Output('response_dropdown', 'value')],
    [Input('table-paging-and-sorting', 'pagination_settings'),
     Input('table-paging-and-sorting', 'sorting_settings'),
     Input('intermediate-value', 'children'),
     Input('files_uploaded_dropdown', 'value')])
def update_metadata_define(pagination_settings, sorting_settings, table_uploaded, uploaded_file):
    if table_uploaded is not None:
        info = json.loads(table_uploaded)
        df = pd.read_json(info['df'], orient='split')

        definer = define.Define(df=df).pipeline()
        filename = info['filename']
        n_samples = definer.samples
        n_features = definer.n_features
        size = definer.size
        date = info['date']
        num_features = definer.metadata['num_features']
        cat_features = definer.metadata['cat_features']
        response = definer.metadata['response']
        columns = definer.data.columns
        if uploaded_file != '':
            metadata_folder = os.path.join(MARKET_PATH, uploaded_file.replace('.csv', ''))
            metadata_filename = uploaded_file.replace('.csv', '') + '_meta.json'
            metadata_path = os.path.join(metadata_folder, metadata_filename)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # print(metadata)
            num_features = metadata['num_features']
            cat_features = metadata['cat_features']
            response = metadata['response']

    else:
        df = pd.DataFrame()
        filename = ''
        n_samples = ''
        n_features = ''
        size = ''
        date = ''
        cat_features = []
        num_features = []
        response = ''
        columns = []

    if len(sorting_settings):
        dff = df.sort_values(
            sorting_settings[0]['column_id'],
            ascending=sorting_settings[0]['direction'] == 'asc',
            inplace=False
        )
    else:
        # No sort is applied
        dff = df

    dff = dff.iloc[
        pagination_settings['current_page']*pagination_settings['page_size']:
        (pagination_settings['current_page'] + 1)*pagination_settings['page_size']
    ]

    num_options = [
        {"label": file, "value": file} for file in columns
    ]
    num_value = num_features
    cat_options = [
        {"label": file, "value": file} for file in columns
    ]
    cat_value = cat_features

    response_options = [
        {"label": file, "value": file} for file in columns
    ]

    columns_table=[
                {'name': i, 'id': i, 'deletable': True} for i in dff.columns
            ]

    style_data_conditional=[{
        'if': {'column_id': e},
        'backgroundColor': '#3D9970',
        'color': 'white',
    } for e in cat_features]

    return dff.to_dict('rows'), columns_table, style_data_conditional, \
           filename, n_samples, n_features, size, cat_options, \
           cat_value, num_options, num_value, response_options, \
           response

@app.callback(
    [Output("save_file_div", "children"),
     Output("files_uploaded_dropdown", "options"),
     Output("files_uploaded_analyze_dropdown", "options")],
    [Input("save_button", "n_clicks")],
    [
        State("filename_indicator", "children"),
        State('n_samples_indicator', 'children'),
        State('n_features_indicator', 'children'),
        State('size_indicator', 'children'),
        State('cat_features_dropdown', 'value'),
        State('num_features_dropdown', 'value'),
        State('response_dropdown', 'value'),
        State('intermediate-value', 'children'),
    ],
)
def save_define(n_clicks, filename, n_samples, n_features,
                size, cat_feat, num_feat, response, df):
    options_uploaded = list_files_market()
    if n_clicks > 0:
        if filename != '':
            folder_path = os.path.join(MARKET_PATH, filename.replace(".csv", "").lower())
            file_path = os.path.join(folder_path, filename)

            # plot_path = os.path.join(app.config['MARKET_DIR'], data_name, 'analyze')
            # tools.path_exists(plot_path)
            # plot_path_plot = os.path.join(plot_path, fig+'.html')
            # dict_figures[fig] = analyzer.plot(fig)
            # analyzer.save_plot(plot_path_plot)

            if os.path.isfile(file_path):
                out = f'{filename} and its metadata already exist.'
            else:
                os.makedirs(folder_path)
                info = json.loads(df)
                df = pd.read_json(info['df'], orient='split')
                definer = define.Define(df=df).pipeline()
                df.to_csv(file_path, index=False)
                options_uploaded = list_files_market()
                metadata = dict()
                metadata['n_samples'] = n_samples
                metadata['n_features'] = n_features
                metadata['size'] = size
                metadata['cat_features'] = cat_feat
                metadata['num_features'] = num_feat
                metadata['response'] = response
                metadata['problem_type'] = definer.metadata['problem_type']
                metadata['filename'] = filename

                metadata_filename = filename.replace('.csv', '') + '_meta.json'
                metadata_path = os.path.join(folder_path, metadata_filename)
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, sort_keys=True, indent=4)
                out = f'{filename} and its metadata are saved.'

            message = dcc.Input(
                # id="save_file_label",
                # placeholder="",
                type="text",
                value=out,
                disabled=True,
                style={"width": "100%"},
            )
            return message, options_uploaded, options_uploaded
    return None, options_uploaded, options_uploaded

# @app.server.route('/upload', methods=['GET', 'POST'])
# def upload_file():
#     # if request.method == 'POST':
#     #     file = request.files['file']
#     #     filename = secure_filename(file.filename)
#     #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#
#     return """
#         <form method=post enctype=multipart/form-data>
#           <input type=file name=file>
#           <input type=submit value=Upload>
#         </form>
#         """
    # return None

