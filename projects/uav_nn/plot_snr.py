"""
plot_snr.py:  Plots the SNR distribution in a single cell environment.

Right now this considers only a single sector.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

import tensorflow.keras.backend as K

from tqdm import trange

path = os.path.abspath('../..')
if not path in sys.path:
    sys.path.append(path)
    
from mmwchanmod.datasets.download import load_model 
from mmwchanmod.sim.antenna import Elem3GPP, Elem3GPPMultiSector
from mmwchanmod.sim.array import URA, RotatedArray
from mmwchanmod.sim.chanmod import MPChan, dir_path_loss
    
"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Plots the SNR distribution')    
parser.add_argument(\
    '--plot_dir',action='store',\
    default='plots', help='directory for the output plots')    
parser.add_argument(\
    '--plot_fn',action='store',\
    default='angle_dist.png', help='plot file name')        
parser.add_argument(\
    '--mod_name',action='store',\
    default='uav_lon_tok', help='model to load') 
    
args = parser.parse_args()
plot_dir = args.plot_dir
plot_fn = args.plot_fn
mod_name = args.mod_name


# Paramters
bw = 400e6   # Bandwidth in Hz
nf = 6  # Noise figure in dB
kT = -174   # Thermal noise in dB/Hz
tx_pow = 23  # TX power in dBm
npts = 100    # number of points for each (x,z) bin
aer_height=30  # Height of the aerial cell
downtilt = 10  # downtilt on terrestrial cells
fc = 28e9
nant_gnb = np.array([8,8])  # gNB array size
nant_ue = np.array([4,4])   # UE/UAV array size


plot_fn = 'snr_dist.png'

"""
Create the arrays
"""
# gNB cell array.  This will be rotated depending on whether
# it is aerial or terrestrial 
elem_gnb = Elem3GPP(thetabw=90, phibw=80)
arr_gnb0 = URA(elem=elem_gnb, nant=nant_gnb, fc=fc)


# UE array.  Array is pointing down.
elem_ue = Elem3GPP(thetabw=90, phibw=80)
arr_ue0 = URA(elem=elem_ue, nant=nant_ue, fc=fc)
arr_ue = RotatedArray(arr_ue0,theta0=-90)

# Number of x and z bins
nx = 40
nz = 20

# Range of x and z distances to test
xlim = np.array([0,500])
zlim = np.array([0,130])

    
# Construct and load the channel model object
K.clear_session()
chan_mod = load_model(mod_name)
    
    
rx_types = chan_mod.rx_types
nplot = len(chan_mod.rx_types)


"""
Main simulation loop
"""      
snr_med = np.zeros((nz,nx,nplot))

    
for iplot, rx_type0 in enumerate(rx_types):

    
    # Print cell type
    print('Simulating %s cell' % rx_type0)
    
    # Set the limits and x and z values to test
    dx = np.linspace(xlim[0],xlim[1],nx)        
    dz = np.linspace(zlim[0],zlim[1],nz)
    if rx_type0 == 'Aerial':
        dz = dz - aer_height
    
    # Create the gNB array.
    # For aerial cells, point them upwards (theta0 = 90)
    # For terrestrial cells, point them with the downtilt
    if rx_type0 == 'Aerial':
        arr_gnb = RotatedArray(arr_gnb0,theta0=90)
    else:
        arr_gnb = RotatedArray(arr_gnb0,theta0=-downtilt)
        
    # Convert to meshgrid
    dxmat, dzmat = np.meshgrid(dx,dz)
    
    # Create the condition vectors
    dvec = np.zeros((nx*nz,3))
    dvec[:,0] = dxmat.ravel()
    dvec[:,2] = dzmat.ravel()
    rx_type_vec = np.tile(iplot, (nx*nz,))
        
        
    # Loop over multiple trials
    snr = np.zeros((nz,nx,npts))
            
    for i in trange(npts):
        # Generate random channels
        chan_list, link_state = chan_mod.sample_path(dvec, rx_type_vec) 

        # Compute the directional path loss of each link        
        n = len(chan_list)
        pl_gain = np.zeros(n)        
        for j, c in enumerate(chan_list):            
            pl_gain[j] = dir_path_loss(arr_gnb, arr_ue, c)[0]
        
       
        # Compute the effective SNR
        snri = tx_pow - pl_gain - kT - nf - 10*np.log10(bw)
    
        # Create the data for the plot    
        snri = snri.reshape((nz,nx))
        snri = np.flipud(snri)
        
        snr[:,:,i] = snri
     
    # Get the median SNR
    snr_med[:,:,iplot] = np.median(snr,axis=2) 
     
         
# Plot the results
for iplot, rx_type0 in enumerate(rx_types):
                    
    plt.subplot(1,nplot,iplot+1)
    plt.imshow(snr_med[:,:,iplot],\
               extent=[np.min(xlim),np.max(xlim),np.min(zlim),np.max(zlim)],\
               aspect='auto', vmin=-20, vmax=60)   
        
    # Add horizontal line indicating location of aerial cell
    if (rx_type0 == 'Aerial'):
        plt.plot(xlim, np.array([1,1])*aer_height, 'r--')
        
    if (iplot > 0):
        plt.yticks([])
    else:
        plt.ylabel('Elevation (m)')
    plt.xlabel('Horiz (m)')
    plt.title(rx_types[iplot])
        

# Add the colorbar
plt.tight_layout()
plt.subplots_adjust(bottom=0.1, right=0.87, top=0.9)
cax = plt.axes([0.92, 0.1, 0.05, 0.8])
plt.colorbar(cax=cax)        
    
if 1:
    # Print plot
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
        print('Created directory %s' % plot_dir)
    plot_path = os.path.join(plot_dir, plot_fn)
    plt.savefig(plot_path)
    print('Figure saved to %s' % plot_path)
    plt.savefig('snr_dist.png', bbox_inches='tight')
            
    


