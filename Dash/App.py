# =============================================================================
# Dash GUI
# TODO: 
#
#
#
#
# =============================================================================

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import pandas as pd
import os
from dash.dependencies import Input, Output

# =============================================================================
# Importing Datasets
# =============================================================================
filedir = os.path.abspath('D:\\GitHub\\Projects\\Dash\\Data')
IBM_df_name = 'daily_IBM.csv'
MSFT_df_name = 'daily_MSFT.csv'
QCOM_df_name = 'daily_QCOM.csv'

IBM_filepath = os.path.join(filedir, IBM_df_name)
MSFT_filepath = os.path.join(filedir, MSFT_df_name)
QCOM_filepath = os.path.join(filedir, QCOM_df_name)

IBM_df = pd.read_csv(IBM_filepath)
MSFT_df = pd.read_csv(MSFT_filepath)
QCOM_df = pd.read_csv(QCOM_filepath)

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
# Initializing the app
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
                        'marginTop': 10},
                    className='nine columns'),
                # Logo, top right
                html.Img(
                    src="https://pbs.twimg.com/profile_images/710845774716911616/PyV-I0_8_400x400.jpg",
                    className='three columns',
                    style={
                        'height': '15%',
                        'width': '15%',
                        'float': 'right',
                        'position': 'relative'},)
                ], className="row"),   
        #### 2.Row ####                
        html.Div([
                # Dropdown, top left
                dcc.Dropdown(
                        id='stock_dropdown',
                        options=[{'label': df, 'value': df} for df in dataframes],
                        value='',
                        className= 'three columns'),
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
                ], className="row"),
        #### 3. Row ####
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