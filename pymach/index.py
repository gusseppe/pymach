import os
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import flask
import plotly.plotly as py
import dash_table_experiments as dt
import math

from plotly import graph_objs as go
from app import app
from app import app, server
from dashboard.apps import  define_front, \
    analyze_front, model_front

#from pymongo import MongoClient
#client = MongoClient('localhost', 27017)
#db = client.test_database

current_path = os.getcwd()
app.layout = html.Div(
    [
        # header
        html.Div([

            html.Span("Pymach (alpha): tool to accelerate the development of Machine Learning models.", className='app-title'),
             html.Div(
                #html.Img(src='https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png',height="100%") sunat_logo

                #html.Img(src=app.get_asset_url('./logo_pymach.png'), height="80%"),
                #style={"float":"right","height":"70%"},
                style={"float":"right","height":"70%"},
                #className='app-title',
             ),
               
        
            # html.Div(
            #     html.Img(src='https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png',height="100%"),
            #
            #     # html.Img(src=app.get_asset_url('./logo_final.png'), height="100%"),
            #     # style={"float":"right","height":"70%"},
            #     style={"float":"middle","height":"100%"},
            #     className='app-title',
            #     ),
            #
            ],
            className="row header"
            ),

        # tabs
        html.Div([

            dcc.Tabs(
                id="tabs",
                style={"height":"20","verticalAlign":"middle"},
                children=[
                    dcc.Tab(label="Define", value="define"),
                    dcc.Tab(label="Analyze", value="analyze"),
                    dcc.Tab(label="Model", value="model"),
                    dcc.Tab(label="Tune", value="tune"),
                    dcc.Tab(label="Predict", value="predict"),
                ],
                value="define",
            )

            ],
            className="row tabs_div"
            ),
       
                
        # divs that save dataframe for each tab
#         html.Div(
#                 sf_manager.get_opportunities().to_json(orient="split"),  # opportunities df
# #                 pd.read_csv('/data/users/Gusseppe/reto_IGV/dashboard/dash-salesforce-crm/assets/get_opportunities.csv').to_json(orient="split"),  # opportunities df
#                 id="opportunities_df",
#                 style={"display": "none"},
#             ),
        #html.Div(sf_manager.get_leads().to_json(orient="split"), id="leads_df", style={"display": "none"}), # leads df
        #html.Div(sf_manager.get_leads().to_json(orient="split"), id="leads_df2", style={"display": "none"}),
#         html.Div(sf_manager.get_cases().to_json(orient="split"), id="cases_df", style={"display": "none"}), # cases df
#         html.Div(pd.read_csv('/data/users/Gusseppe/reto_IGV/dashboard/dash-salesforce-crm/assets/get_leads.csv').to_json(orient="split"), id="leads_df", style={"display": "none"}), # leads df
#         html.Div(pd.read_csv('/data/users/Gusseppe/reto_IGV/dashboard/dash-salesforce-crm/assets/get_cases.csv').to_json(orient="split"), id="cases_df", style={"display": "none"}), # cases df


        # Tab content
        html.Div(id="tab_content", className="row", style={"margin": "2% 3%"}),
        
        # html.Div([
        #
        #     dt.DataTable(
        #         id='my-datatable', rows=[{}]
        #     ),
        # ], style={"display": "none"}),
        
#         html.Link(href="https://use.fontawesome.com/releases/v5.2.0/css/all.css",rel="stylesheet"),
#         html.Link(href="https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css",rel="stylesheet"),
#         html.Link(href="https://fonts.googleapis.com/css?family=Dosis", rel="stylesheet"),
#         html.Link(href="https://fonts.googleapis.com/css?family=Open+Sans", rel="stylesheet"),
#         html.Link(href="https://fonts.googleapis.com/css?family=Ubuntu", rel="stylesheet"),
#         html.Link(href="https://cdn.rawgit.com/amadoukane96/8a8cfdac5d2cecad866952c52a70a50e/raw/cd5a9bf0b30856f4fc7e3812162c74bfc0ebe011/dash_crm.css", rel="stylesheet")
    ],
    className="row",
    style={"margin": "0%"},
)


@app.callback(Output("tab_content", "children"), [Input("tabs", "value")])
def render_content(tab):
    if tab == "define":
        return define_front.layout
    elif tab == "analyze":
        return analyze_front.layout
        # return cases.layout
    elif tab == "model":
        return model_front.layout
    else:
        pass
        # return opportunities.layout


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug=True, port=9088)
