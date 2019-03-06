# =============================================================================
# Dash GUI
# TODO: 
# implement Trading markers
# Model visualization
# Sharpe
# Portfolio value
# =============================================================================

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import plotly.graph_objs as go
import pandas as pd
import os
from dash.dependencies import Input, Output

# =============================================================================
# Importing Datasets
# =============================================================================
filedir = os.path.abspath('D:\\GitHub\\Projects\\Dash\\Data\\')

IBM_df = pd.read_csv(filedir + '\\daily_IBM.csv')
MSFT_df = pd.read_csv(filedir + '\\daily_MSFT.csv')
QCOM_df = pd.read_csv(filedir + '\\daily_QCOM.csv')

portfolio_val_train = pd.read_csv(filedir + '\\portfolio_val_train.csv')
portfolio_val_train = portfolio_val_train.rename(index=str, columns={'0': 'IBM_stock','1': 'MSFT_stock', '2': 'QCOM_stock', 
                                              '3': 'Cash', '4': 'Portfolio_val', '5': 'Perc_diff' })
portfolio_val_test = pd.read_csv(filedir + '\\portfolio_val_test.csv')
portfolio_val_test = portfolio_val_test.rename(index=str, columns={'0': 'IBM_stock','1': 'MSFT_stock', '2': 'QCOM_stock', 
                                              '3': 'Cash', '4': 'Portfolio_val', '5': 'Perc_diff' })

train_data = pd.read_csv(filedir + '\\train_data.csv')
train_data = train_data.transpose()
train_data = train_data.rename(index= str, columns={0: 'IBM', 1: 'MSFT', 2: 'QCOM'})
test_data = pd.read_csv(filedir + '\\train_data.csv')
test_data = test_data.transpose()
test_data = test_data.rename(index= str, columns={0: 'IBM', 1: 'MSFT', 2: 'QCOM'})



dataframes= {'IBM': IBM_df,
             'MSFT': MSFT_df,
             'QCOM': QCOM_df}

indexes = {'open': 'open',
           'high': 'high',
           'low': 'low',
           'close': 'close',
           'volume': 'volume'}


# Selected Dataframe
def get_data_object(user_selection):
    return dataframes[user_selection]

def get_column(dataframe, index):
    return dataframe[index]

# =============================================================================
# Layout
# =============================================================================
app = dash.Dash()

# Boostrap CSS. for layout structure
app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})

app.layout = html.Div(
    # 12 Columns (offset by one)
    html.Div([
        #### 1.Row ####
        html.Div([
                # Title, top left
                html.H1(
                    children='Reinforcement Learning',
                    style={
                        'marginTop': 10,
                        'marginBottom': 0},
                    className='nine columns'),
                # Logo, top right
                html.Img(
                    src="https://www.prophysics.ch/wp-content/uploads/2017/07/zhaw-logo-inner.png",
                    className='three columns',
                    style={
                        'height': '15%',
                        'width': '15%',
                        'float': 'right',
                        'position': 'relative',
                        'marginBottom': 0},)
                ], className="row"),
        #### 2.Row #### 
        html.Div([
            html.Div(
                children = 'Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua.',
                    className='six columns')
                ], className='row'),
        html.Div([
            dcc.Tabs(id='tabs', children=[
# =============================================================================
# 1. Tab
# =============================================================================
                dcc.Tab(label='Data Visualization',children=[
                    #### 3.Row ####                
                    html.Div([
                            # Stock Dropdown, top left
                            dcc.Dropdown(
                                    id='stock_dropdown',
                                    options=[{'label': df, 'value': df} for df in dataframes],
                                    value='',
                                    className= 'three columns'),
                            # Column Dropdown, top left                                    
                            dcc.Dropdown(
                                    id='column_dropdown',
                                    options=[{'label': column, 'value': column} for column in indexes],
                                    value='',
                                    className= 'three columns'),
                            # Infobox, top right
                            html.Div(
                                children= 'Select the stock and index for visualization',
                                className='six columns',
                                style={
                                    'float': 'right'})
                            ], className="row",style={'marginTop': 20}),
                    #### 4. Row ####
                    html.Div([
                        html.Div([
                            # Graph
                            dcc.Graph(
                                id='stock_plots') 
                                ],className= 'twelve columns'),
                        html.Div([
                            # Table
                            dt.DataTable(
                                    id='table',
                                    rows = [{}],
                                    filterable = True,
                                    sortable= True),
                                ], className= 'twelve columns')
                            ], className="row")
                        ]),
# =============================================================================
# 2. Tab
# =============================================================================
             dcc.Tab(label='Performance Metrics', children=[
                    #### 3.Row ####                
                    html.Div([
                            # Infobox, top right
                            html.Div(
                                children= 'Select the stock and index for visualization',
                                className='six columns',
                                style={
                                    'float': 'right'})
                            ], className="row",style={'marginTop': 20}),
                    #### 4. Row ####
                    html.Div([
                        html.Div([
                            # Graph
                            dcc.Graph(
                                id='stock_ammount',
                                figure={    
                                    'data':[                                        
                                        {'x': train_data.index, 'y': train_data['IBM'],
                                         'type': 'line', 'name': 'IBM'},
                                        {'x': train_data.index, 'y': train_data['QCOM'],
                                         'type': 'line', 'name': 'QCOM','visible': 'legendonly'},
                                        {'x': train_data.index, 'y': train_data['MSFT'],
                                         'type': 'line', 'name': 'MSFT', 'visible': 'legendonly'},
                                        {'x': portfolio_val_train.index, 'y': portfolio_val_train['IBM_stock'],
                                         'type': 'scatter','opacity': 0.4, 'name': 'IBM_stock'},
                                        {'x': portfolio_val_train.index, 'y': portfolio_val_train['QCOM_stock'],
                                         'type': 'scatter','opacity': 0.4, 'name': 'QCOM_stock','visible': 'legendonly'},
                                        {'x': portfolio_val_train.index, 'y': portfolio_val_train['MSFT_stock'],
                                         'type': 'scatter','opacity': 0.4, 'name': 'MSFT_stock','visible': 'legendonly'}
                                         ]
                                        })
                                        
                                ],className= 'twelve columns'),
#                        html.Div([
#                            # Table
#                            dt.DataTable(
#                                    id='table',
#                                    rows = [{}],
#                                    filterable = True,
#                                    sortable= True),
#                                ], className= 'twelve columns')
                            ], className="row")
                    ]),
# =============================================================================
# 3. Tab
# =============================================================================
             dcc.Tab(label='Information', children=[
                html.Div([
                    html.H1(
                        children='Reinforcement Learning')
                        ])
                    ])
                ])
            ])
    ], className='ten columns offset-by-one')
)

# =============================================================================
# Callbacks
# =============================================================================
@app.callback(
        Output('table', 'rows'), 
        [Input('stock_dropdown', 'value')])
def update_table(user_selection):
    df= get_data_object(user_selection)
    return df.to_dict('records')

@app.callback(
        Output('stock_plots', 'figure'),
        [Input('stock_dropdown', 'value'),
        Input('column_dropdown', 'value')])
def update_figure(stock_selection, column_selection):
    df = get_data_object(stock_selection)
    column = get_column(df, column_selection)
    return {'data': 
                [{'x': df.timestamp, 'y': column, 
                  'type': 'line', 'name': stock_selection}]}
                    



if __name__ == '__main__':
    app.run_server(debug=True)