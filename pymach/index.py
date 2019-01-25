import os
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from textwrap import dedent

from app import app, server
from dashboard.apps import define_front, \
    analyze_front, model_front, tune_front, predict_front


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
       

        # Tab content
        html.Div(id="tab_content", className="row", style={"margin": "2% 3%"}),
        

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
    elif tab == "model":
        return model_front.layout
    elif tab == "tune":
        return tune_front.layout
    elif tab == "predict":
        return predict_front.layout
    else:
        message = html.Div([
            dcc.Markdown(
                dedent(f'''
                > #### Under development
                > ###### Head to [pymach](http://www.github.com/gusseppe/pymach)

            '''))
        ],className='two columns indicator')
        return message


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug=False, port=9088)
