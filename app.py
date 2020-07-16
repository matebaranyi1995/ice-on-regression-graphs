import json
import datetime
import random
import base64
import io

import dash
import dash_daq as daq
import dash_html_components as html
import dash_core_components as dcc
import dash_table as dte
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
# from flask_caching import Cache

from RegressionGraph import RegressionGraph
from ICERegression import ICERegression
from RegressionGraph import build_reggraph_by_r_script

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

numofcells = 20  # increase this if desired number of variables is greater

app = dash.Dash(__name__, external_stylesheets=['assets/modified.css'])

# CACHE_CONFIG = {
#    # try 'filesystem' if you don't want to setup redis
#    #'CACHE_TYPE': 'filesystem',
#    'CACHE_TYPE': 'redis',
#    'CACHE_REDIS_URL': os.environ.get('REDIS_URL', 'localhost:6379')
# }
# cache = Cache()
# cache.init_app(app.server, config=CACHE_CONFIG)

# app.scripts.config.serve_locally = True
# app.config.supress_callback_exceptions = True

app.layout = \
    html.Div([
        html.H1("ICE on regression graphs (demo)"),
        html.H5("Current time: {}".format(datetime.datetime.now())),

        html.Br(),
        html.H5(id='uploadarea_head',
                children="1. Upload a data file:"),

        html.Div(id='uploadarea', children=[
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select the file')
                ]),
                style={
                    'width': '50%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center'
                },
                multiple=False),
            html.Button(id='upload-button',
                        children='Click for upload (after selection)')]),

        html.Br(),
        dcc.Store(id='memory'),
        dcc.Store(id='listofvars'),

        html.Div(id='tablehead', children=''),

        html.Br(),
        html.H5(id='graphdataarea_head',
                children="2. Choose parents, type and group of each variable:"),

        dcc.Store(id='types'),
        dcc.Store(id='parents'),
        dcc.Store(id='boxes'),

        html.Div(id='graphdataarea',
                 children=html.Table([
                     html.Tr([
                         html.Td(dcc.Dropdown(
                             id='parent' + str(k),
                             disabled=True,
                             style={'display': 'none'}
                         )),
                         html.Td(dcc.Dropdown(
                             id='type' + str(k),
                             disabled=True,
                             style={'display': 'none'}
                         )),
                         html.Td(dcc.Dropdown(
                             id='box' + str(k),
                             disabled=True,
                             style={'display': 'none'}
                         ))
                     ]) for k in range(numofcells)
                 ], style={'display': 'none'})),

        html.Br(),
        html.H5(id='submitarea_head',
                children="3. Drawing the graph:"),

        html.Div(id='submitarea',
                 children=[
                     # html.Button(id='submit_button', children='Submit the inputs'),
                     html.Button(id='learnbyR_button', children='Learn structure (optional)'),
                     dcc.Store(id='reg_graph'),
                     html.Button(id='drawgraph_button', children='Draw the graph'),
                     # html.Br(),
                     # html.Div(id='debug', children=html.H5('print of Submit_2 button')),
                     html.Br(),
                     html.Img(id='reggraph_image')]),

        html.Br(),
        html.H5(id='parameterarea_head',
                children="4. Choose the parameters of ICE and train the model:"),

        html.Table(id='parameterarea',
                   children=[
                       html.Tr([
                           html.Td(html.Label("Bandwidth selection method:")),
                           html.Td(dcc.Dropdown(
                               id='bandwidth',
                               options=[{'label': 'Scott\'s rule of thumb', 'value': 'scott'},
                                        {'label': 'Least-squares Cross-validation', 'value': 'cv_ls'},
                                        {'label': 'AIC Hurvich criteria', 'value': 'aic'}],
                               multi=False,
                               value='scott',
                               style={
                                   'width': '150pt'
                               }
                           )),
                           html.Td(daq.BooleanSwitch(
                               id='modemax_sw',
                               on=False,
                               label="ModeMaximization",
                               labelPosition="bottom"
                           )),
                           html.Td(daq.BooleanSwitch(
                               id='round_sw',
                               on=False,
                               label="Rounding",
                               labelPosition="bottom"
                           )),
                           html.Td(daq.BooleanSwitch(
                               id='boxpar_sw',
                               on=False,
                               label="ParentsOfTheSameBox",
                               labelPosition="bottom"
                           )),
                           html.Td(html.Button(
                               id='ICE_button',
                               children='Train the model'
                           ))])
                   ]),
        dcc.Store(id='ICE'),

        html.Br(),
        html.H5(id='predarea_head',
                children="5. Predict with the previously trained model:"),

        html.Div(id='predarea',
                 children=[
                     html.Div(id='inputs_for pred',
                              children=[dte.DataTable(id='table_forinput')]),
                     # style={"display":"None"}),
                     html.Button(id='predict_button', children='Predict!'),
                     html.Div(id='pred_print', children='')]),
        html.Div(id='debug')
    ], className="container")


# Put the data into the memory
@app.callback(Output('memory', 'data'),
              [Input('upload-button', 'n_clicks')],
              [State('upload-data', 'contents'),
               State('upload-data', 'filename')])
def parse_content(nclicks, contents, filename):
    if nclicks is None:
        raise PreventUpdate
    content_string = contents.split(',')[1]
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            return df.to_json()
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            return df.to_json()
    except Exception as e:
        print(e)
        return None


def hide(style):
    if style == {'display': 'none'}:
        return {'display': 'initial'}
    else:
        return {'display': 'none'}


# Hide things
for i in ['upload', 'graphdata', 'submit', 'parameter', 'pred']:
    @app.callback(Output(i + 'area', 'style'),
                  [Input(i + 'area_head', 'n_clicks')],
                  [State(i + 'area', 'style')])
    def hide_upload(nclicks, style):
        if nclicks is None:
            raise PreventUpdate
        return hide(style)


# Extend the layout with respect to the uploaded file 
@app.callback(Output('tablehead', 'children'),
              [Input('memory', 'modified_timestamp')],
              [State('memory', 'data')])
def tablehead_generator(timestamp, data):
    if timestamp is None:
        raise PreventUpdate
    try:
        df = pd.read_json(data)
        return html.Div([
            html.H6("Head of the uploaded datatable:"),
            dte.DataTable(id='table_forhead',
                          data=df.head().to_dict("rows"),
                          columns=[{"name": col, "id": col} for col in df.columns]),
        ])
    except ValueError:
        pass


@app.callback(Output('graphdataarea', 'children'),
              [Input('memory', 'modified_timestamp')],
              [State('memory', 'data')])
def input_generator(timestamp, data):
    if timestamp is None:
        raise PreventUpdate
    try:
        df = pd.read_json(data)
        cols = df.columns.tolist()
        return html.Div([
            html.Table([
                html.Tr([
                    html.Td(html.Label(j)),
                    html.Td(dcc.Dropdown(
                        id='parent' + str(k),
                        options=[{'label': col, 'value': col} for col in cols],
                        multi=True,
                        value=''#[random.choice(cols[k + 1:])] if k < len(cols) - 1 else ''
                    )),
                    html.Td(dcc.Dropdown(
                        id='type' + str(k),
                        options=[{'label': typ[0], 'value': typ[1]} for typ in
                                 [['continuous', 'c'], ['ordered (discrete)', 'o'], ['unordered (discrete)', 'u']]],
                        value='c',
                        style={
                            'width': '150pt'
                        }
                    )),
                    html.Td(dcc.Dropdown(
                        id='box' + str(k),
                        options=[{'label': bx, 'value': bx} for bx in
                                 ['context', 'box 1', 'box 2', 'box 3', 'box 4', 'box 5', 'box 6']],
                        value='context',
                        style={
                            'width': '100pt'
                        }
                    ))
                ]) for j, k in zip(cols, range(len(cols)))
            ]),
            *[html.Div([
                dcc.Dropdown(
                    id='parent' + str(k),
                    disabled=True,
                    style={'display': 'none'}
                ),
                dcc.Dropdown(
                    id='type' + str(k),
                    disabled=True,
                    style={'display': 'none'}
                ),
                dcc.Dropdown(
                    id='box' + str(k),
                    disabled=True,
                    style={'display': 'none'}
                )]) for k in range(len(cols), numofcells)]
        ])
    except ValueError:
        pass


@app.callback(Output('inputs_for pred', 'children'),
              [Input('memory', 'modified_timestamp')],
              [State('memory', 'data')])
def input_generator_forpred(timestamp, data):
    if timestamp is None:
        raise PreventUpdate
    try:
        df = pd.read_json(data)
        return [html.H5("Type some inputs for prediction:"),
                dte.DataTable(id='table_forinput',
                              data=[{col: np.NaN for col in df.columns}],
                              columns=[{"name": col, "id": col} for col in df.columns],
                              editable=True)]
    except ValueError:
        pass


# Put a list of the variables into memory
@app.callback(Output('listofvars', 'data'),
              [Input('memory', 'modified_timestamp')],
              [State('memory', 'data')])
def savevarnames(timestamp, data):
    if timestamp is None:
        raise PreventUpdate
    try:
        df = pd.read_json(data)
        cols = df.columns.tolist()
        return json.dumps(cols)
    except ValueError:
        pass


@app.callback(Output('boxes', 'data'),
              # [Input('submit_button', 'n_clicks'),
              [Input('listofvars', 'data')] +
              [Input('box' + str(i), 'value') for i in range(numofcells)]
              )
def submit_b(lstv, *args):
    if lstv is None:
        raise PreventUpdate
    # if nclicks is None:
    #     raise PreventUpdate
    d = dict(zip(json.loads(lstv), args))
    return json.dumps(d)


@app.callback(Output('parents', 'data'),
              # [Input('submit_button', 'n_clicks'),
              [Input('listofvars', 'data')] +
              [Input('parent' + str(i), 'value') for i in range(numofcells)]
              )
def submit_p(lstv, *args):
    if lstv is None:
        raise PreventUpdate
    # if nclicks is None:
    #     raise PreventUpdate
    d = dict(zip(json.loads(lstv), args))
    return json.dumps(d)


@app.callback(Output('types', 'data'),
              # [Input('submit_button', 'n_clicks'),
              [Input('listofvars', 'data')] +
              [Input('type' + str(i), 'value') for i in range(numofcells)]
              )
def submit_t(lstv, *args):
    if lstv is None:
        raise PreventUpdate
    # if nclicks is None:
    #     raise PreventUpdate
    d = dict(zip(json.loads(lstv), args))
    return json.dumps(d)


def spec_inv_dict(dic):
    inv_dict = {}
    for k, v in dic.items():
        inv_dict.setdefault(v, [])
        inv_dict[v].append(k)
    return inv_dict


@app.callback([Output('parent' + str(i), 'value') for i in range(numofcells)],
              [Input('learnbyR_button', 'n_clicks')],
              [State('boxes', 'data'),
               State('types', 'data'),
               State('memory', 'data')]
              )
def newgraph_by_r_script(nclicks, box, typ, dataf):
    if nclicks is None or box is None or typ is None or dataf is None:
        raise PreventUpdate
    boxes = json.loads(box)
    inv_boxes = spec_inv_dict(boxes)
    types = json.loads(typ)
    data = pd.read_json(dataf)
    digr = build_reggraph_by_r_script(data, inv_boxes, types)
    cols = data.columns.tolist()
    oulist = [list(digr.predecessors(col)) for col in cols]
    nonelist = [None] * (numofcells - len(cols))
    return tuple(oulist + nonelist)


# Save the RegGraph object into memory
@app.callback(Output('reg_graph', 'data'),
              [Input('drawgraph_button', 'n_clicks')],
              [State('boxes', 'data'),
               State('parents', 'data'),
               State('types', 'data')])
def build_graph(nclicks, box, par, typ):
    if nclicks is None or box is None or par is None or typ is None:
        raise PreventUpdate
    boxes = json.loads(box)
    inv_boxes = spec_inv_dict(boxes)
    types = json.loads(typ)
    parents = json.loads(par)
    graph = RegressionGraph(incoming_digraph=parents, types=types,
                            boxes=inv_boxes, reversed_input=True)
    return graph.serialize()


# Draw the previously saved RegGraph
@app.callback(Output('reggraph_image', 'src'),
              [Input('drawgraph_button', 'n_clicks')],
              [State('reg_graph', 'data')])
def draw_graph(nclicks, graph):
    if graph is None or nclicks is None:
        raise PreventUpdate
    regg = RegressionGraph.deserialize(graph)
    g_image = regg.reggraph_to_dot(splinestyle="spline", labels=None)
    g_image = g_image.draw(format='png', prog='dot')
    encoded_image = base64.b64encode(g_image)
    return 'data:image/png;base64,{}'.format(encoded_image.decode())


# Save the model object into memory
@app.callback(Output('ICE', 'data'),
              [Input('ICE_button', 'n_clicks')],
              [State('reg_graph', 'data'),
               State('memory', 'data'),
               State('modemax_sw', 'on'),
               State('round_sw', 'on'),
               State('boxpar_sw', 'on'),
               State('bandwidth', 'value')])
def teach_ice(nclicks, graph, data, modemax, rounding, boxparents, bw):
    if nclicks is None or graph is None or data is None:
        raise PreventUpdate
    regg = RegressionGraph.deserialize(graph)
    # estset = estset #TODO: bármit ezzel
    datareb = pd.read_json(data)
    model = ICERegression(reggraph=regg,
                          data=datareb,
                          bw=bw,
                          modemax=modemax,
                          rounding=rounding,
                          boxparents=boxparents)
    model.learn()
    return model.serialize()  # TODO: megoldani a model encode-olást, maybe done?


# Prediction
@app.callback(Output('pred_print', 'children'),
              [Input('predict_button', 'n_clicks')],
              [State('table_forinput', 'data'),
               State('table_forinput', 'columns'),
               State('memory', 'data'),
               State('ICE', 'data')])
def predict(nclicks, rows, columns, traindat, serialized_model):
    if nclicks is None or rows is None or serialized_model is None:
        raise PreventUpdate
    datareb = pd.read_json(traindat)
    mod = ICERegression.deserialize(serialized_model, data=datareb)
    inp = pd.DataFrame(rows, columns=[c['name'] for c in columns]).dropna(axis=1)
    preds = mod.predict(inp).round(2)
    return html.Div([
        html.H5("Predicted values:"),
        dte.DataTable(id='pred_table',
                      data=preds.to_dict("rows"),
                      columns=[{"name": col, "id": col} for col in preds.columns])
    ])


###########################################################################

if __name__ == '__main__':
    app.run_server(debug=True)
