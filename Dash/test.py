import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, Event

app = dash.Dash()

app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})


app.layout = html.Div(children=[
    html.Div(children=[
    html.H4(children='Example Dash'),
    
    
    html.Div(id='div_row1',children=[
    
        # Div Retention
        html.Div(id='div', children=[
            html.P("Retention"),
            html.Button('Default Values', id='default_values'),
            dcc.Dropdown(
            id="sel_01",
            options=[
                {'label': 'True', 'value': True},
                {'label': 'False', 'value': False}

            ],
            value=False,
            multi=False
            ),
            dcc.Dropdown(
            id="sel_02",
            options=[
                {'label': 'True', 'value': True},
                {'label': 'False', 'value': False}

            ],
            value=False,
            multi=False
            ),
            html.P(id="p1",children=["placeholder"])
        ], className='six columns')

    ]),
    dcc.Interval(
        id='interval-component',
        interval=5*60*1000 # in milliseconds
    )
    ]),
    ])


@app.callback(
    Output(component_id='sel_01', component_property='value'),
    events=[Event('default_values', 'click')])
def update():
    return False

@app.callback(
    Output(component_id='sel_02', component_property='value'),
    events=[Event('default_values', 'click')])
def update():
    return False


@app.callback(
    Output(component_id='p1', component_property='children'),
    [Input(component_id='sel_01', component_property='value'),
    Input(component_id='sel_02', component_property='value')])
def update(val_1, val_2):
    return "text: "+str(val_1)+str(val_2)

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=5091)