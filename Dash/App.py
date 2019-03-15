# =============================================================================
# Dash GUI
# TODO: 
# NNet loss visualization (Tab3)
# 
# =============================================================================

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import pandas as pd
import os, fnmatch
import pickle
import numpy as np
from dash.dependencies import Input, Output
from itertools import chain
# =============================================================================
# Importing Datasets
# =============================================================================
filedir = os.path.abspath('D:\\GitHub\\Projects\\Dash\\Data\\')

# Stock Data
IBM_df = pd.read_csv(filedir + '\\daily_IBM.csv')
MSFT_df = pd.read_csv(filedir + '\\daily_MSFT.csv')
QCOM_df = pd.read_csv(filedir + '\\daily_QCOM.csv')

# Model Data
train_data = pd.read_csv(filedir + '\\train_data.csv')
train_data = train_data.transpose()
train_data = train_data.rename(index= str, columns={0: 'IBM', 1: 'MSFT', 2: 'QCOM'})

test_data = pd.read_csv(filedir + '\\test_data.csv')
test_data = test_data.transpose()
test_data = test_data.rename(index= str, columns={0: 'IBM', 1: 'MSFT', 2: 'QCOM'})

# Indexes for dropdown reference
model_data = {'train': train_data,
             'test': test_data}

dataframes= {'IBM': IBM_df,
             'MSFT': MSFT_df,
             'QCOM': QCOM_df}

indexes = {'open': 'open',
           'high': 'high',
           'low': 'low',
           'close': 'close',
           'volume': 'volume'}

# =============================================================================
# Visualization functions
# =============================================================================
def action_visual(portfolio_val):
    m = portfolio_val
    n_time = portfolio_val.shape[0]
    n_stock = portfolio_val.shape[1]-3
    
    # Create empty lists inside dictionaries
    a = {'actions_{}'.format(i):[] for i in range(n_stock)}
    b = {'buy_{}'.format(i):[] for i in range(n_stock)}
    h = {'hold_{}'.format(i):[] for i in range(n_stock)}
    s = {'sell_{}'.format(i):[] for i in range(n_stock)} 

    for i in range(n_stock):
        stock = m[:,i]
        for j in range(1,n_time):            
            # Filling actions list with action integers (2=buy, 1=hold, 0=sell)
            a['actions_{}'.format(i)] = np.append(a['actions_{}'.format(i)], stock[j]-stock[j-1])
        a['actions_{}'.format(i)] = np.array(a['actions_{}'.format(i)])    
        a['actions_{}'.format(i)][a['actions_{}'.format(i)]>0] = 2
        a['actions_{}'.format(i)][a['actions_{}'.format(i)]==0] = 1
        a['actions_{}'.format(i)][a['actions_{}'.format(i)]<0] = 0
        # Getting x-axis position of actions with enumerate
        for g, j in enumerate(a['actions_{}'.format(i)]):
            if j == 2:
                b['buy_{}'.format(i)].append(g)
            elif j == 1:
                h['hold_{}'.format(i)].append(g)
            elif j == 0:
                s['sell_{}'.format(i)].append(g)      
    return b, h, s


def action_pos(action_visual, data):
    b, h, s = action_visual
    n_stock = len(b)
    
    # create empty lists inside dictionaries
    b_pos = {'buy_pos_{}'.format(i):[] for i in range(n_stock)}
    h_pos = {'hold_pos_{}'.format(i):[] for i in range(n_stock)}
    s_pos = {'sell_pos_{}'.format(i):[] for i in range(n_stock)}
    
    for i in range(n_stock):
        for j in range(len(b['buy_{}'.format(i)])):
            b_pos['buy_pos_{}'.format(i)].append(data[i:i+1].values.tolist()[0][b['buy_{}'.format(i)][j]])
        for j in range(len(h['hold_{}'.format(i)])):
            h_pos['hold_pos_{}'.format(i)].append(data[i:i+1].values.tolist()[0][h['hold_{}'.format(i)][j]])
        for j in range(len(s['sell_{}'.format(i)])):
            s_pos['sell_pos_{}'.format(i)].append(data[i:i+1].values.tolist()[0][s['sell_{}'.format(i)][j]])
    return b_pos, h_pos, s_pos
    
# =============================================================================
# Callback functions
# =============================================================================
def get_data_object(user_selection):
    return dataframes[user_selection]


def get_column(dataframe, index):
    return dataframe[index]


def get_model_data(user_selection):
    return model_data[user_selection]


def get_portfolio_val(user_selection):
    portfolio = pickle.load(open(filedir + '\\portfolio_metrics\\{}'.format(user_selection), 'rb'))
    portfolio = pd.DataFrame(portfolio)
    portfolio = portfolio.rename(index=str, columns={0: 'IBM_ammount',1: 'MSFT_ammount', 2: 'QCOM_ammount', 
                                              3: 'Cash', 4: 'Portfolio_val', 5: 'Perc_diff' })
    return portfolio


def get_available_portfolios(data_type):
    # Reading portfolio values in metrics folder
    listOfFiles = os.listdir('Data\\portfolio_metrics')
    pattern = "portfolio_val*{}*.p".format(data_type)
    available_portfolios = {}  
    for entry in listOfFiles:  
        if fnmatch.fnmatch(entry, pattern):
               available_portfolios.update({'{}'.format(entry): entry})
    return available_portfolios


def get_all_portfolios(user_selection):
    filename = user_selection.replace('portfolio_val_', '\\portfolio_metrics\\')
    portfolio = pickle.load(open(filedir +  filename, "rb" ))
    return portfolio

# =============================================================================
#    
# Layout
#
# =============================================================================
app = dash.Dash()
# Boostrap CSS. for layout structure
app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})
# app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

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
            # Infobox, top left
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
                    #### 1.Row ####                
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
                                style={'float': 'right'})
                            ], className="row",style={'marginTop': 30}),
                    #### 2. Row ####
                    html.Div([
                        html.Div([
                            # Stock Graph
                            dcc.Graph(
                                id='stock_plots') 
                                ],className= 'twelve columns'),
                        html.Div([
                            # Stock Table
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
                    #### 1.Row ####
                    html.Div([
                        # Train and Test button
                        html.Button('Training',
                                    id= 'train_button', 
                                    n_clicks_timestamp='0',
                                    className='two columns'),
                        html.Button('Test',
                                    id= 'test_button', 
                                    n_clicks_timestamp='0',
                                    className='two columns'),
                        # Checkbox for trading markers
                        dcc.Checklist(
                                id= 'trading_markers_checkbox',
                                options=[
                                    {'label': 'Trading markers', 
                                     'value': 'checked', 
                                     'disabled':True}],
                                values='',
                                className='two columns'),
                    # Portfolio dropdown                    
                        dcc.Dropdown(id='portfolio_val_dropdown', 
                                     multi=True, 
                                     value= 'default',
                                     style={'vertical-align': 'right'}, 
                                     className='six columns')                        
                            ], className="row",style={'marginTop': 30}),
                    #### 2.Row ####                    
                    html.Div([
                        # Train test graph
                        dcc.Graph(id='train_test_graph')
                            ], className="row"),
                    #### 3.Row ####                    
                    html.Div([
                        # Portfolio value and sharpe all episodes
                        dcc.Graph(id= 'portfolio_graph_all',
                                  className = 'six columns'),
                        dcc.Graph(id= 'sharpe_graph_all',
                                  className = 'six columns')  
                            ], className = 'row'),
                    #### 4.Row ####                                                 
                    html.Div([
                            # Portfolio value last episode
                        dcc.Graph(id= 'portfolio_graph_last')
                            ], className = 'row')
                    ]),
# =============================================================================
# 3. Tab
# =============================================================================
             dcc.Tab(label='Information', children=[
                html.Div([
                    html.H1(
                        children='Under construction'),
                        ])
                    ])
                ])
            ])
    ], className='ten columns offset-by-one')
)
# =============================================================================
#
# Callbacks
#
# =============================================================================
# =============================================================================
# 1. Tab Callbacks 
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
def update_stock_graph(stock_selection, column_selection):
    df = get_data_object(stock_selection)
    column = get_column(df, column_selection)
    return {'data': 
                [{'x': df.timestamp, 'y': column, 'type': 'line',
                  'name': stock_selection}]}
# =============================================================================
# 2. Tab Callbacks 
# =============================================================================
@app.callback(
        Output('trading_markers_checkbox', 'options'),
        [Input('portfolio_val_dropdown', 'value')])
def release_checkbox(dropdown_selection):
    if dropdown_selection == 'default':
        return [{'label': 'Trading markers', 
                 'value': 'unchecked', 
                 'disabled':True}]
    else: 
        return [{'label': 'Trading markers', 
                 'value': 'checked', 
                 'disabled':False}]


@app.callback(
        Output('portfolio_val_dropdown', 'options'),
        [Input('train_button', 'n_clicks_timestamp'),
         Input('test_button', 'n_clicks_timestamp')])
def update_portfolio_dropdown_train(train_button, test_button):
    if int(train_button)>int(test_button):
        available_portfolios = get_available_portfolios('train')
    elif int(test_button)>int(train_button):
        available_portfolios = get_available_portfolios('test')
    return [{'label': portfolio, 
             'value': portfolio} for portfolio in available_portfolios]


@app.callback(
        Output('portfolio_val_dropdown', 'value'),
        [Input('train_button', 'n_clicks_timestamp'),
         Input('test_button', 'n_clicks_timestamp')])
def default_portfolio_dropdown(train_button, test_button):
    return 'default'


@app.callback(
        Output('train_test_graph', 'figure'),
        [Input('train_button', 'n_clicks_timestamp'),
         Input('test_button', 'n_clicks_timestamp'),
         Input('trading_markers_checkbox', 'values'), 
         Input('portfolio_val_dropdown', 'value')])
def update_portfolio(train_button,test_button, checkbox_selection, dropdown_selection):
    # Get model data according to button value
    if int(train_button)> int(test_button):
        df = get_model_data('train')
        button_selection = 'train'
    elif int(test_button) > int(train_button):
        df = get_model_data('test')
        button_selection = 'test'
    
    end_portfolio = get_portfolio_val(dropdown_selection[0])
    end_portfolio_val = np.array(end_portfolio)
    
    # Data traces
    model_data =  [{'x': df.index, 
                    'y': df[i],
                    'type': 'line',
                    'line':{'width':1.2},
                    'name': i}for i in df.columns]
    ammount_of_stock = [{'x': end_portfolio.index, 
                         'y': end_portfolio[column],
                         'type': 'line', 
                         'line':{'width':1}, 
                         'name': column, 
                         'visible':'legendonly'}for column in end_portfolio.columns[0:3]]
    
    # Checkbox selection and corresponding plot
    if checkbox_selection==['checked']:
        try:
            action_x = action_visual(end_portfolio_val)
            action_y = action_pos(action_x, df.transpose())
            b, h, s = action_x
            b_pos, h_pos, s_pos = action_y    
            n_stock = df.shape[1]
            
            # Marker traces
            buy_markers = [{'x': b['buy_{}'.format(i)],
                            'y': b_pos['buy_pos_{}'.format(i)],
                            'type': 'scatter',
                            'mode': 'markers',
                            'marker':{'symbol':'triangle-up', 
                                      'size':10, 
                                      'color': 'green'},
                            'name': df.columns[i]}for i in range(n_stock)]
    
            sell_markers= [{'x': s['sell_{}'.format(i)],
                            'y': s_pos['sell_pos_{}'.format(i)],
                            'type': 'scatter',
                            'mode': 'markers',
                            'marker':{'symbol':'triangle-down', 
                                      'size':10, 
                                      'color': 'red'},
                            'name': df.columns[i]}for i in range(n_stock)]
    
            data = list(chain.from_iterable((model_data, 
                                             ammount_of_stock, 
                                             buy_markers, 
                                             sell_markers)))
            return {'data': data, 
                    'layout': {'title': '{} datasets'.format(button_selection.capitalize()), 
                               'legend': {'orientation':'h'},
                               'xaxis':{'range': (0,len(df))}}}
        except:
            print ('No test portfolio available')

    else:
        try:
           data = list(chain.from_iterable((model_data, 
                                            ammount_of_stock))) 
        except:
            data = model_data       
        return {'data':data, 
                'layout': {'title': '{} datasets'.format(button_selection.capitalize()), 
                           'legend': {'orientation':'h'}, 
                           'margin': {'t':70}}}


@app.callback(
        Output('portfolio_graph_last', 'figure'),
        [Input('portfolio_val_dropdown', 'value')])
def update_last_portfolio_graph(dropdown):
    # Data traces
    portfolio_value = [{'x': get_portfolio_val(portfolio).index, 
                        'y': get_portfolio_val(portfolio)['Portfolio_val'],
                        'type': 'line',
                        'line':{'width':1.2},
                        'name': portfolio}for portfolio in dropdown] 
    data = portfolio_value
    return {'data': data, 
            'layout': {'title': 'Portfolio value of last episode',
                       'legend': {'orientation':'h'},
                       'shapes':[{'type':'line', 
                                  'x0': 0, 
                                  'y0': 20000,
                                  'x1': len(get_portfolio_val(dropdown[0]).index), 
                                  'y1': 20000,
                                  'line':{'color': 'red',
                                          'dash': 'dashdot', 
                                          'width': 1.2}}],
                        'margin': {'t':30}}}


@app.callback(
        Output('portfolio_graph_all', 'figure'),
        [Input('portfolio_val_dropdown', 'value')])
def update_all_portfolios_graph(dropdown_selection):
    #Data traces
    all_portfolio_values = [{'x': get_all_portfolios(portfolio)[0], 
                             'orientation': 'v', 
                             'marker': {'line':{'color':'black', 
                                                'width': 1}},
                             'type': 'histogram', 
                             'name': portfolio }for portfolio in dropdown_selection] 
    data = all_portfolio_values
    return {'data':data, 
            'layout': {'title': 'Portfolio values over all episodes', 
                       'legend': {'orientation':'h'}, 
                       'margin': {'t':50}}}


@app.callback(
        Output('sharpe_graph_all', 'figure'),
        [Input('portfolio_val_dropdown', 'value')])
def update_all_sharpe_graph(dropdown_selection):
    #Data traces
    all_sharpe_ratios = [{'y': get_all_portfolios(portfolio)[1], 
                          'type': 'line', 
                          'name': portfolio }for portfolio in dropdown_selection] 
    data = all_sharpe_ratios
    return {'data':data, 
            'layout': {'title': 'Sharpe ratios over all episodes',
                       'legend': {'orientation':'h'}, 
                       'margin': {'t':50}}}

# =============================================================================
# Run App
# =============================================================================
if __name__ == '__main__':
    app.run_server(debug=True)