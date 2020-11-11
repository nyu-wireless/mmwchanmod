"""
plot_angle_dist.py:  Plots the angular distribution

For all the NLOS paths, the program:
* Computes the  AoA and AoD relative to the LOS path
* Plots the empirical distribution of the relative angles as 
  a function of the distance
* Generates random angles with the same conditions as the model,
  and plots the relative angle as a function of the distance
  for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import argparse
import os
import sys

path = os.path.abspath('../..')
if not path in sys.path:
    sys.path.append(path)
    
from mmwchanmod.common.constants import  AngleFormat
from mmwchanmod.learn.models import ChanMod
from mmwchanmod.datasets.download import get_dataset, load_model

"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Plots the omni directional path loss CDF')
parser.add_argument(\
    '--plot_dir',action='store',\
    default='plots', help='directory for the output plots')    
parser.add_argument(\
    '--plot_fn',action='store',\
    default='angle_dist.png', help='plot file name')        
parser.add_argument(\
    '--ds_name',action='store',\
    default='uav_lon_tok', help='data set to load')    
parser.add_argument(\
    '--mod_name',action='store',\
    default='uav_lon_tok', help='model to load') 
    
args = parser.parse_args()
plot_dir = args.plot_dir
plot_fn = args.plot_fn
ds_name = args.ds_name
mod_name = args.mod_name



def plot_ang_dist(ax,chan_mod,dvec,nlos_ang,nlos_pl,iang,pl_tol=30, dmax=1000):
    """
    Plots the conditional distribution of the relative angle.
    
    Parameters
    ----------
    ax : pyplot axis
        Axis to plot on
    chan_mod : ChanMod structure
        Channel model.
    dvec : (nlink,ndim) array
            vector from cell to UAV
    nlos_ang : (nlink,npaths_max,nangle) array
            Angles of each path in each link.  
            The angles are in degrees
    nlos_pl : (nlink,npaths_max) array 
            Path losses of each path in each link.
            A value of pl_max indicates no path
    iang: integer from 0 to DataFormat.nangle-1
        Index of the angle to be plotted
    np_plot:  integer
        Number of paths whose angles are to be plotted
    """
    # Get the distances
    dist = np.sqrt(np.sum(dvec**2,axis=1))
    dist_plot = np.tile(dist[:,None],(1,chan_mod.npaths_max))
    dist_plot = dist_plot.ravel()
    
    # Transform the angles.  The transformations compute the
    # relative angles and scales them by 180
    ang_tr = chan_mod.transform_ang(dvec, nlos_ang, nlos_pl)
    ang_rel = ang_tr[:,iang*chan_mod.npaths_max:(iang+1)*chan_mod.npaths_max]*180
    ang_rel = ang_rel.ravel()
    
    # Find valid paths
    pl_tgt = np.minimum(nlos_pl[:,0]+pl_tol, chan_mod.pl_max)
    Ivalid = (nlos_pl < pl_tgt[:,None])
    Ivalid = np.where(Ivalid.ravel())[0]
    
    # Get the valid distances and relative angles
    ang_rel = ang_rel[Ivalid]
    dist_plot = dist_plot[Ivalid]      
    
    # Set the angle and distance range for the historgram
    drange = [0,dmax]
    if iang==AngleFormat.aoa_phi_ind or iang==AngleFormat.aod_phi_ind:
        ang_range = [-180,180]
    elif iang==AngleFormat.aoa_theta_ind or iang==AngleFormat.aod_theta_ind:
        ang_range = [-90,90]
    else:
        raise ValueError('Invalid angle index')
    
    # Compute the emperical conditional probability
    H0, dedges, ang_edges = np.histogram2d(dist_plot,ang_rel,bins=[10,40],\
                                           range=[drange,ang_range])       
    Hsum = np.sum(H0,axis=1)
    H0 = H0 / Hsum[:,None]
    
    # Plot the log probability.
    # We plot the log proability since the probability in linear
    # scale is difficult to view
    log_prob = np.log10(np.maximum(0.01,H0.T))
    im = ax.imshow(log_prob, extent=[np.min(dedges),np.max(dedges),\
               np.min(ang_edges),np.max(ang_edges)], aspect='auto')   
    return im






"""
Load the true data
"""

data_dict = get_dataset(ds_name)
cfg = data_dict['cfg']    
real_data = data_dict['test_data']

"""
Run the model
"""
    
# Construct the channel model object
K.clear_session()
chan_mod = load_model(mod_name)
    
# Generate samples from the path
sim_data = chan_mod.sample_path(\
    real_data['dvec'], real_data['rx_type'], real_data['link_state'], return_dict=True)
    
chan_mod0 = ChanMod(cfg=cfg)
    
    
"""
Plot the angular distributions
"""    
plt.rcParams.update({'font.size': 12})

ang_str = ['AoD Az', 'AoD El', 'AoA Az', 'AoA El']
    

fig, ax = plt.subplots(AngleFormat.nangle, 2, figsize=(5,10))
for iang in range(AngleFormat.nangle):
    
    for j in range(2):
        if j == 0:
            data = real_data
        else:
            data = sim_data
                        
        axi = ax[iang,j]
        im = plot_ang_dist(axi,chan_mod0,data['dvec'],data['nlos_ang'],\
                      data['nlos_pl'],iang,dmax=1500)
            
        if iang < 3:
            axi.set_xticks([])
        else:
            axi.set_xlabel('Dist (m)')
        if j == 1:
            axi.set_yticks([])
            title_str = ang_str[iang] + ' Model'   
        else:
            title_str = ang_str[iang] + ' Data'   
        axi.set_title(title_str)
fig.tight_layout()

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

    
if 1:     
    # Save the figure
    plot_path = os.path.join(plot_dir, plot_fn)
    plt.savefig(plot_path)
    
    
    


    


