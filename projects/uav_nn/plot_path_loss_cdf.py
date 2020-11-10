"""
plot_path_loss_cdf:  Plots the CDF of the path loss on the test data,
and compares that to the randomly generated path loss from the trained model.

To compare the test data and model in London-Tokyo:
    
    python plot_path_loss_cdf.py --model_city LonTok --plot_fn pl_cdf_lt.png

To compare the test data in London-Tokyo and all models:
    
    python plot_path_loss_cdf.py --model_city "LonTok Moscow Beijing Boston"
        --plot_fn pl_cdf_all.png    
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
import argparse

import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers
import tensorflow.keras.backend as K


path = os.path.abspath('../..')
if not path in sys.path:
    sys.path.append(path)
    
from mmwchanmod.common.constants import LinkState
from mmwchanmod.learn.models import ChanMod
from mmwchanmod.datasets.download import get_dataset, load_model
from mmwchanmod.learn.datastats import  data_to_mpchan 
    

"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Plots the omni directional path loss CDF')
parser.add_argument(\
    '--plot_dir',action='store',\
    default='plots', help='directory for the output plots')    
parser.add_argument(\
    '--plot_fn',action='store',\
    default='pl_cdf.png', help='plot file name')        
parser.add_argument(\
    '--ds_city',action='store',\
    default='LonTok', help='data set to load')    
parser.add_argument(\
    '--model_city',action='store',\
    default='LonTok Beijing', help='cities for the models to test')    
        
args = parser.parse_args()
plot_dir = args.plot_dir
ds_city = args.ds_city
model_city = args.model_city
plot_fn = args.plot_fn

# Overwrite command line if running from spyder
if 0:
    ds_city = 'LonTok'
    model_city = 'LonTok'
    plot_fn = 'pl_cdf_lt.png'
if 1:
    ds_city = 'LonTok'
    model_city = 'LonTok Moscow Beijing Boston'
    plot_fn = 'pl_cdf_all.png'

# Cities to test are the city for the followed by all the models
city_test = [ds_city] + model_city.split()


# Dictionaries looking up the model to use for each city  
model_city_dict  = {\
    'LonTok': ('uav_lon_tok', None), # OK\
    'Beijing': ('uav_beijing',None), # OK\
    'Boston': ('uav_boston',None),  # Good\    
    'Moscow': ('uav_moscow',None)}  # Good
    
    
# Dictionaries looking up the dataset for each city  
ds_city_dict = {\
    'LonTok': 'uav_lon_tok',\
    'Beijing': 'uav_beijing',\
    'Boston':  'uav_boston',\
    'Moscow':  'uav_moscow'    }   
    
use_true_ls = False
    
     
"""
Find the path loss CDFs
"""
pl_omni_plot = []
ls_plot = []
leg_str = []

ntest = len(city_test)
for i, city in enumerate(city_test):
    
    if (i == 0):
        """
        For first city, use the city data
        """
        # Load the data
        ds = ds_city_dict[city]
        data_dict = get_dataset(ds)
        cfg = data_dict['cfg']    
        data = data_dict['test_data']
        
        # Convert data to channel list
        chan_list, ls = data_to_mpchan(data, cfg)
        
        leg_str.append(city + ' data')
        
        
    else:
        """
        For subsequent cities, generate data from model
        """
        
        # Get the model directory to load
        mod_name, ckpt = model_city_dict[city]
        
        # Construct the channel model object
        K.clear_session()
        chan_mod = load_model(mod_name)
        
        # Load the configuration and link classifier model
        print('Simulating model %s' % mod_name)        
        
        # Generate samples from the path
        if use_true_ls:
            ls = data['link_state']
        else:
            ls = None
        chan_list, ls = chan_mod.sample_path(\
            data['dvec'], data['rx_type'], ls)
            
        leg_str.append(city + ' model')            
            
        
    # Compute the omni-directional path loss for each link    
    n = len(chan_list)
    pl_omni = np.zeros(n)
    for i, chan in enumerate(chan_list):
        if chan.link_state != LinkState.no_link:
            pl_omni[i] = chan.comp_omni_path_loss()
    
    # Save the results    
    ls_plot.append(ls)
    pl_omni_plot.append(pl_omni)
        
        
                

    
"""
Create the plot
"""
ntypes = len(cfg.rx_types)
nplot = len(pl_omni_plot)
plt.figure(figsize=(10,5))
fig, ax = plt.subplots(1, ntypes)

for i, rx_type in enumerate(cfg.rx_types):
    
    # Plot color    
    for iplot in range(nplot):
    
        
        # Find the links that match the type and are not in outage
        I = np.where((data['rx_type']==i)\
                     & (ls_plot[iplot] != LinkState.no_link))[0]
            
        # Select color and fmt
        if (iplot == 0):
            fmt = '-'
            color = [0,0,1]
        else:
            fmt = '--'
            t = (iplot-1)/(nplot-1)
            color = [0,t,1-t]

        # Plot the omni-directional path loss                 
        ni = len(I)
        p = np.arange(ni)/ni            
        ax[i].plot(np.sort(pl_omni_plot[iplot][I]), p, fmt, color=color)
              
    
    ax[i].set_title(rx_type)
    ax[i].set_xlabel('Path loss (dB)')
    if (i == 0):
        ax[i].set_ylabel('CDF')
    ax[i].grid()
    ax[i].set_xlim([90, 200])
    

fig.legend(leg_str, borderaxespad=0.1, loc='lower right',\
           bbox_to_anchor=(0, 0.15, 1, 0.85))

plt.subplots_adjust(right=0.85)
    
    
# Print plot
if 1:
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
        print('Created directory %s' % plot_dir)
    plot_path = os.path.join(plot_dir, plot_fn)
    plt.savefig(plot_path)
    print('Figure saved to %s' % plot_path)

    


