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
import pickle
from matplotlib import pyplot as plt
from dash.dependencies import Input, Output

# =============================================================================
# Importing Datasets
# =============================================================================
filedir = os.path.abspath('D:\\GitHub\\Projects\\Dash\\Data\\')

action_x = pickle.load(open(filedir +'\\action_x.p', 'rb'))
action_y = pickle.load(open(filedir +'\\action_y.p', 'rb'))


train_data = pd.read_csv(filedir + '\\train_data.csv')
train_data = train_data.transpose()
data= train_data.rename(index= str, columns={0: 'IBM', 1: 'MSFT', 2: 'QCOM'})

n_stock = data.shape[0]
b, h, s = action_x
b_pos, h_pos, s_pos = action_y

shape_size = 30
for i in range(n_stock):
#     print(i)
    x = range(1, data[i:i+1].shape[1]+1)
    y = data[i:i+1].values.tolist()[0]
    try:
        buy = b['buy_{}'.format(i)]
        hold = h['hold_{}'.format(i)]
        sell = s['sell_{}'.format(i)]
        buy_pos = b_pos['buy_pos_{}'.format(i)]
        hold_pos = h_pos['hold_pos_{}'.format(i)]
        sell_pos = s_pos['sell_pos_{}'.format(i)]
    except:
        break
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    ax.plot(x, y, color="grey", linewidth=0.5)
    ax.scatter(buy, buy_pos,
               s = shape_size,
               color = "green",
               marker = "^")
    # ax.scatter(hold, hold_pos,
    #            s = 5,
    #           color = "gold",
    #           marker = "_")
    ax.scatter(sell, sell_pos,
               s = shape_size,
               color = "red",
               marker = "v")

    plt.show()