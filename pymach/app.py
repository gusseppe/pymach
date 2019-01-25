import math
import os
import flask
import dash
# import dash_auth
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
# from sfManager import sf_Manager

VALID_USERNAME_PASSWORD_PAIRS = [
    ['hello', 'world']
]

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

# auth = dash_auth.BasicAuth(
#     app,
#     VALID_USERNAME_PASSWORD_PAIRS
# )

current_path = os.getcwd()
external_css = [
    os.path.join(current_path, 'assets/stylesheet-oil-and-gas.css'),
    os.path.join(current_path, 'assets/all.css'),
    # os.path.join(current_path, 'assets/loading.css'),
    os.path.join(current_path, 'assets/dash_crm.css')
    # "/data/users/Gusseppe/reto_IGV/dashboard/dash-salesforce-crm/assets/stylesheet-oil-and-gas.css",
    # "/data/users/Gusseppe/reto_IGV/dashboard/dash-salesforce-crm/assets/all.css",
    # "/data/users/Gusseppe/reto_IGV/dashboard/dash-salesforce-crm/assets/dash_crm.css"
    # 'https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/css/bootstrap.min.css'
]

for css in external_css:
    app.css.append_css({"external_url": css})
    
app.scripts.config.serve_locally = True
app.config.suppress_callback_exceptions = True
app.config.requests_pathname_prefix = ''

# sf_manager = sf_Manager()

millnames = ["", " K", " M", " B", " T"] # used to convert numbers


# return html Table with dataframe values  
def df_to_table(df):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in df.columns])] +
        
        # Body
        [
            html.Tr(
                [
                    html.Td(df.iloc[i][col])
                    for col in df.columns
                ]
            )
            for i in range(len(df))
        ]
    )

def df_to_table2(dic):
    
    return dt.DataTable(
#         #rows=[{}],
#         #rows=df[0:50].to_dict('records'),
         rows=dic,
#         # optional - sets the order of columns
         columns=sorted(dic[0].keys()),
         filterable=True,
         sortable=True,
         id='table_cpe'
     )
    


#returns most significant part of a number
def millify(n):
    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        ),
    )

    return "{:.0f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])


#returns top indicator div
def indicator(color, text, id_value):
    return html.Div(
        [
            
            html.P(
                text,
                className="twelve columns indicator_text"
            ),
            html.P(
                id = id_value,
                className="indicator_value"
            ),
        ],
        className="four columns indicator",
        
    )
