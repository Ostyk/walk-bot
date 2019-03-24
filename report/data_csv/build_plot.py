import pandas as pd
import time
import numpy as np
import os 
from scipy.interpolate import spline
import matplotlib.pyplot as plt
plt.rcParams['text.usetex']=True
plt.rcParams['text.latex.unicode']=True
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

# Local variables

# Open the data file for reading lines
def plot(data = 'data.csv', norm_x = True):
    datafile = pd.read_csv(data)
    x = datafile['Step']
    y = datafile['Value']
    name =  i.split("-")[2].split('.')[0]
    x_label = 'frames'
    if norm_x: 
        x/=256
        x_label = 'Episodes'

    x_sm = np.array(x)
    y_sm = np.array(y)

    x_smooth = np.linspace(0, x_sm.max(), 100)
    y_smooth = spline(x, y, x_smooth)
    return x_smooth, y_smooth, name

fig, axs = plt.subplots(nrows=7, ncols=1, sharex=True, figsize=(20,28))

q = sorted([i for i in os.listdir() if i.endswith('.csv')])

for c, i in enumerate(q): 
    
    x_smooth, y_smooth, name = plot(data = i, norm_x = True)
    color = 'orange'
    axs[c].plot(x_smooth, y_smooth, color, linewidth=5)
    color = 'k'
    # Colorcode the tick tabs 
    axs[c].tick_params(axis='x', colors=color, labelsize=30)
    axs[c].tick_params(axis='y', colors=color, labelsize=30)
    color = 'k'
    # Colorcode the spine of the graph
    axs[c].spines['bottom'].set_color(color)
    axs[c].spines['top'].set_color(color)
    axs[c].spines['left'].set_color(color)
    axs[c].spines['right'].set_color(color)
    # Put the title and labels
    plt.xlabel("Episodes", color=color, size=50)
    name = " ".join(name.split('_'))
    axs[c].set_ylabel(str(name), color=color, size=50)
    axs[c].grid(alpha=0.7)
plt.tight_layout()
plt.subplots_adjust(wspace=1, hspace=0.1)
plt.savefig("../img/results.png")