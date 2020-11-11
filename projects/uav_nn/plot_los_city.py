"""
Plots the LOS probability for each city
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.interpolate import interp1d

path = os.path.abspath('../..')
if not path in sys.path:
    sys.path.append(path)

from mmwchanmod.datasets.download import get_dataset
from mmwchanmod.common.constants import LinkState
from mmwchanmod.learn.datastats import hist_mean


# Cities to plot
cities = ['boston', 'london', 'beijing', 'moscow', 'tokyo']

# Distance to sort the line
d0 = 300

# Range of elevations to test
dzrange = [30,60]
ntypes = 2

fig, ax = plt.subplots(1,ntypes,figsize=(10,5))
for it in range(ntypes):
    dcen = []
    plos = []
    plos0 = []
    
    for city in cities:
        # Get dataset
        print('Processing %s' % city)
        ds_name = ('uav_%s' % city)
        ds = get_dataset(ds_name)
        cfg = ds['cfg']
        train_data = ds['train_data']
        test_data = ds['test_data']
        
        
        # Get the link state and LOS probability
        link_state = train_data['link_state']
        los = (link_state == LinkState.los_link)
        
        # Get the distance
        dvec = train_data['dvec']
        dx = np.sqrt(np.sum(dvec[:,0:1]**2,axis=1))
        dz = dvec[:,2]
        
        # Get link type
        It = np.where((train_data['rx_type'] == it) & \
            (dz >= dzrange[0]) &  (dz >= dzrange[1]) )[0]
        
        # Compute the empirical probability
        bin_edges, dcnt, plosi = hist_mean(dx[It], los[It], bins=10, range=[0,1000])
        dceni = (bin_edges[1:] + bin_edges[:-1])/2
        
        # Save them
        plos.append(plosi)
        dcen.append(dceni)
        
        # Find the los prob at d0
        f = interp1d(dceni, plosi)
        plos0.append(f(d0))
        
    # Sort by probability
    if (it == 0):
        plos0 = np.array(plos0)
        I = np.argsort(-plos0)
    
    # Plot the LOS probability for each city
    ncity = len(cities)
    leg_str = []
    for i in I:    
        leg_str.append(cities[i])
        ax[it].plot(dcen[i], plos[i], 'o-')
    ax[it].legend(leg_str)
    ax[it].grid()
    ax[it].set_xlabel('Horiz distance (m)')
    ax[it].set_ylabel('LOS Probability')
    title = '%s, z in [%d,%d]' % (cfg.rx_types[it], dzrange[0], dzrange[1])
    ax[it].set_title(title)
    
# Write figure
plot_dir = 'plots'
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)
plot_path = os.path.join(plot_dir, 'los_prob_city.png')
plt.savefig(plot_path, bbox_inches='tight')
