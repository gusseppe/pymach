# -*- coding: utf-8 -*-
import json
import math
import os
import pandas as pd
import pickle
import flask
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.plotly as py
import functools
import dash_table_experiments as dt

from apps import graph_tools, cpe_tools, chatbot_tools
# from apps import chatbot_tools
from dash.dependencies import Input, Output, State
from plotly import graph_objs as go
from flask_caching import Cache
from app import app, indicator, millify, df_to_table, sf_manager

from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client.production

# cache = Cache(app.server, config={
#     'CACHE_TYPE': 'filesystem',
#     'CACHE_DIR': '/data/users/Gusseppe/reto_IGV/dashboard/piloto1/cache'
# })
# returns pie chart that shows lead source repartition
TIMEOUT = 60
total_monto = 0
#@cache.memoize(timeout=TIMEOUT)

#Init chatboot
id_chatbot, chatbot = chatbot_tools.init_chatbot()
chatbot_tools.train_chatbot(chatbot)

@functools.lru_cache(maxsize=32)
def grafo_cpe(dimen_graph, statement, title):
    response, action, ruc = chatbot_tools.run_chatbot(id_chatbot, chatbot, statement)
    df = cpe_tools.logic_vinculacion(ruc, action)
    G = graph_tools.init_graph(df)
    fig = graph_tools.draw_graph(G, dimen_graph, title)

    return fig

@functools.lru_cache(maxsize=32)
def graph_cpe_detail(ruc, dimen_graph, title):
    action = 'detalle'
    df = cpe_tools.logic_vinculacion(ruc, action)
    G = graph_tools.init_graph(df)
    fig = graph_tools.draw_graph(G, dimen_graph, title)

    return fig

@functools.lru_cache(maxsize=32)
def draw_time_series(ruc):
    current_path = os.getcwd()
    df_path = os.path.join(current_path, 'df_time_series.csv')
    df = pd.read_csv(df_path)
    df_perfil = cpe_tools.mostrar_perfil_ruc(ruc)
    title = f"Comportamiento CPE de {df_perfil['nom_comercial'].head(1).values[0]}"

    fig = graph_tools.draw_time_series(df, title)

    return fig

@functools.lru_cache(maxsize=32)

def chatbot_cpe(statement):
    
    response, action, ruc = chatbot_tools.run_chatbot(id_chatbot, chatbot, statement)
#     print(f'chatbot_cpe {ruc}')
#     if cpe_tools.existe_vinculacion(ruc):
#         return response
#     else:
#         return f" Te conozco: cpe_tools.mostrar_ficha(ruc). Pero aun no tengo vinculaciones."
# # #     print(action)
# # #     print(ruc)
    return response

layout = [

    # top controls
    html.Div(
        [
            html.Div(
                dcc.Dropdown(
                    id="periodo_dropdown2",
                    options=[
                        {"label": "Mayo", "value": "mayo"},
                        {"label": "Junio", "value": "junio"},
                        {"label": "Julio", "value": "julio"},
                    ],
                    value="mayo",
                    clearable=False,
                ),
                className="two columns",
            ),
            html.Div(
                dcc.Dropdown(
                    id="dimension_graph",
                    options=[
                        {"label": "3D", "value": "3d"},
                        {"label": "2D", "value": "2d"},
#                         {"label": "Otros", "value": "converted"},
#                         {"label": "Otros", "value": "lost"},
                    ],
                    value="2d",
                    clearable=False,
                ),
                className="two columns",
            ),

            # add button
#             html.Div(
#                 html.Span(
#                     "Buscar por RUC",
#                     id="new_lead",
#                     n_clicks=0,
#                     className="button button--primary",
#                     style={
#                         "height": "34",
#                         "background": "#119DFF",
#                         "border": "1px solid #119DFF",
#                         "color": "white",
#                     },
#                 ),
#                 className="two columns",
#                 style={"float": "right"},
#             ),
        ],
        className="row",
        style={"marginBottom": "10"},
    ),

    # indicators row div
    html.Div(
        [
            indicator(
                "#00cc96", "Cantidad de CPE", "left_relac_indicator"
            ),
            indicator(
                "#119DFF", "Monto total CPE", "middle_relac_indicator"
            ),
            indicator(
                "#EF553B",
                "Mes CPE",
                "right_relac_indicator",
            ),
        ],
        className="row",
    ),
    ## Chatbot
    html.Div(
            [
            html.Div(
                [
                    dcc.Dropdown(
                        id="input_chatbot_dropdown",
                        options=[
                            {"label": "A quién le vende este RUC: ", "value": "A quien le vende"},
                            {"label": "A quién le compra este RUC: ", "value": "A quien le compra"},
                        ],
                        placeholder='Funciones',
                        value="",
#                         value=['vendedor', 'comprador'],
#                         multi=True,
                        #clearable=False,
                    ),
                ],className="four columns",
            ),
            html.Div(
                [
                    dcc.Input(
                        id="input_chatbot_input",
#                         placeholder="Preguntale al grafo: p.e, Quien le vende a este RUC: 10212132244",
                        placeholder="Hola!, busca a una empresa por RUC ",
                        type="text",
#                         list = ['este ruc te refieres?', 'asdasd'],
                        value="",
                        style={"width": "100%"},
                    ),
                            ],
#                 className="six columns",
                className="four columns",
                style={"marginBottom": "10"},
                # style={"paddingRight": "15"},
            ),
            #dropdown empresas
            html.Div(
                [
                    dcc.Dropdown(
                        id="input_nombre_empresa_dropdown",
                        options=cpe_tools.mostrar_empresas(),
                        placeholder='Empresas',
                        value="",
                    ),
                ],className="two columns",
            ),

            html.Div(
                [
                # submit button
                    html.Span(
                        "GO",
                        id="submit_chatboot",
                        n_clicks=0,
                        #className="btn btn-primary"
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
                [
                    # html.Hr(),
                    dcc.Input(
                        id="output_chatbot",
                        placeholder="Adaline: ",
                        type="text",
                        value="",
                        disabled=True,
                        style={"width": "100%"},
                    ),
                ],
#                 className="six columns",
                className="twelve columns",
                # style={"paddingRight": "15"},
            ),
        ],
        className="row",
        style={"marginBottom": "10", "marginTop": "10"},
    ),
    
  
     # Grafo principal
    html.Div(
        [
#             html.Div(
#                 [
#                     html.P("Agrupacion por monto de CPE" ),
#                     dcc.Graph(
#                         id="monto_cpe",
#                         style={"height": "90%", "width": "98%"},
#                         config=dict(displayModeBar=False),
#                     ),
#                 ],
#                 className="four columns chart_div"
#             ),

            html.Div(
                [
#                     html.P("Agrupacion por cantidad de CPE"),
                    
                    dcc.Graph(
                        id="main_graph",
#                         figure=grafo_3d_cpe(1,2),
                        style={"height": "200%", "width": "100%"},
                        config=dict(displayModeBar=True,
                                    showLink=False),
                    ),
                        
                ],
                className="twelve columns",
            ),

        ],
        className="row",
        style={"marginBottom": "10", "marginTop": "10"},
    ),
    # Grafo detalle
    html.Div(
        [

            html.Div(
                [
                    #                     html.P("Agrupacion por cantidad de CPE"),

                    dcc.Graph(
                        id="node_details_graph",
                        style={"height": "200%", "width": "100%"},
                        config=dict(displayModeBar=True,
                                    showLink=False),
                    ),

                ],
                className="six columns",
            ),
            html.Div([
                html.Div([

                    dt.DataTable(
                        id='my-datatable2', rows=[{}],
                    ),
                ], className="six columns", id='my-datatable-div2',
                   style={"marginBottom": "10", "paddingLeft": "5"}),

                html.Div([

                    dt.DataTable(
                        id='table_detail_level_2', rows=[{}],
                    ),
                ], className="six columns", id='div_detail_level_2'),

            ]),

        ],
        className="row",
        style={"marginBottom": "10", "marginTop": "10"},
    ),

    # Time series
    html.Div(
        [

            html.Div(
                [
                    #                     html.P("Agrupacion por cantidad de CPE"),

                    dcc.Graph(
                        id="time_series_detail",
                        style={"height": "200%", "width": "100%"},
                        config=dict(displayModeBar=True,
                                    showLink=False),
                    ),

                ],
                className="twelve columns",
            ),

        ],
        className="row",
        style={"marginBottom": "10", "marginTop": "10"},
    ),

#     table div 1
#
#     html.Div([
#         html.Div([
#
# #                 id="relac_table2",
#
#                 dt.DataTable(
#                     id='my-datatable', rows=[{}],
#                 ),
#         ], className="twelve columns", id='my-datatable-div'),
#
#     ],
#         className="row", style={"marginBottom": "15", "marginTop": "5"},
#     ),
    
]


# updates left indicator based on df updates
@app.callback(
    Output("left_relac_indicator", "children"), [Input("dimension_graph", "value")]
)
def left_relac_indicator_callback(status):
    return millify(db.cpe_mayo_2018.count())
#     return converted_relac


# updates middle indicator based on df updates
@app.callback(
    Output("middle_relac_indicator", "children"), [Input("dimension_graph", "value")]
)
def middle_relac_indicator_callback(status): 
    total_monto = 37e9
    
    return millify(total_monto)


# updates right indicator based on df updates
@app.callback(
    Output("right_relac_indicator", "children"), [Input("periodo_dropdown2", "value")]
)
def right_relac_indicator_callback(periodo):
    return periodo


# update pie chart figure based on dropdown's value and df updates
@app.callback(
    Output("main_graph", "figure"),
    [Input("submit_chatboot", "n_clicks"),
     Input("dimension_graph", "value")],
    [
        State("input_chatbot_dropdown", "value"),
        State("input_chatbot_input", "value"),
    ],
)
def grafo_cpe_callback(n_clicks, dimen_graph, in_drop, in_input):
    if n_clicks > 0:
        statement = str(in_drop) + ' ' + str(in_input)
        title = 'Red de vinculaciones CPE'
        return grafo_cpe(dimen_graph, statement, title)

    return None

@app.callback(
    Output("output_chatbot", "value"),
    [Input("submit_chatboot", "n_clicks")],
    [
        State("input_chatbot_dropdown", "value"),
        State("input_chatbot_input", "value")
    ],
)
def chatbot_cpe_callback(n_clicks, in_drop, in_input):
    if n_clicks > 0:
        statement = str(in_drop) + ' ' + str(in_input)
        return chatbot_cpe(statement)

    return "Preguntame algo :)"

# @app.callback(
#     Output('my-datatable-div', 'children'),
#     [Input("submit_chatboot", "n_clicks")],
#     [
#         State("input_chatbot_dropdown", "value"),
#         State("input_chatbot_input", "value")
#     ],
#
# )
# def show_df_search_callback(n_clicks, in_drop, in_input):
#     if n_clicks > 0:
#         statement = str(in_drop) + ' ' + str(in_input)
#         response, action, ruc = chatbot_tools.run_chatbot(id_chatbot, chatbot, statement)
#         df = cpe_tools.logic_vinculacion(ruc, action)
#
#         return  html.Div([dt.DataTable(
#                         #rows=[{}],
#                         rows=df.to_dict('records'),
#                         # optional - sets the order of columns
#                         columns=sorted(df.columns),
#                         # row_selectable=True,
#                         filterable=True,
#                         sortable=True,
#                         # selected_row_indices=list(df.index),
#                         id='my-datatable'
#         )], className="row")
#
#     #return None

@app.callback(
    Output('my-datatable-div2', 'children'),
    [Input('main_graph', 'clickData'), Input("submit_chatboot", "n_clicks")],

)
def show_df_detail_level_1_callback(clickData, n_clicks):
    if clickData is not None:
        ruc = clickData['points'][0]['text'].split('|')[0].split(':')[2].strip()
        df = cpe_tools.mostrar_perfil_ruc(ruc)
        
        return  html.Div(
            [
                # html.H6(f"Perfil de: {df['nom_comercial'].head(1).values[0]}"),
                dt.DataTable(
                        #rows=[{}],
                        rows=df.to_dict('records'),
                        # optional - sets the order of columns
                        columns=sorted(df.columns),
                        row_selectable=True,
                        # filterable=True,
                        sortable=True,
                        selected_row_indices=list(df.index),
                        id='my-datatable2'
        )], className="row")
    
    #return None

@app.callback(
    Output('node_details_graph', 'figure'),
    [Input('main_graph', 'clickData'),
    ],
    [
        State("dimension_graph", "value"),
    ],

)
def show_graph_detail_level_1_callback(clickData, dimen_graph):
    if clickData is not None:
        ruc = clickData['points'][0]['text'].split('|')[0].split(':')[2].strip()

        df_perfil = cpe_tools.mostrar_perfil_ruc(ruc)
        title = f"Redes de vinculaciones de {df_perfil['nom_comercial'].head(1).values[0]}"

        return graph_cpe_detail(ruc, dimen_graph, title)

@app.callback(
    Output('div_detail_level_2', 'children'),
    [Input('main_graph', 'clickData')],

)
def show_df_detail_level_2_callback(clickData):

    if clickData is not None:
        ruc = clickData['points'][0]['text'].split('|')[0].split(':')[2].strip()
        print(ruc)
        action = 'detalle'
        df = cpe_tools.logic_vinculacion(ruc, action)
        print(df.head())


        return  html.Div(
            [
                # html.H6(f"Tabla vinculaciones de {df['nom_comercial'].head(1).values[0]}"),
                dt.DataTable(
                    #rows=[{}],
                    rows=df.to_dict('records'),
                    # optional - sets the order of columns
                    columns=sorted(df.columns),
                    row_selectable=True,
                    # filterable=True,
                    sortable=True,
                    selected_row_indices=list(df.index),
                    id='table_detail_level_2'
                )], className="row")

    #return None

@app.callback(
    Output('input_chatbot_input', 'value'),
    [Input('input_nombre_empresa_dropdown', 'value')],

)
def fill_input_chatbot_callback(nombre_empresa):
    return nombre_empresa

@app.callback(
    Output('time_series_detail', 'figure'),
    [Input('main_graph', 'clickData'),
     ],
    [
        State("dimension_graph", "value"),
    ],

)
def show_graph_detail_level_1_callback(clickData, dimen_graph):
    if clickData is not None:
        ruc = clickData['points'][0]['text'].split('|')[0].split(':')[2].strip()

        return draw_time_series(ruc)
