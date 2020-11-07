"""
train_mod.py:  Training of the channel model

This program trains both the link state predictor
and path VAE models from the ray tracing data.  
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

import tensorflow as tf
tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers
import tensorflow.keras.backend as K
import argparse


path = os.path.abspath('../..')
if not path in sys.path:
    sys.path.append(path)
    
from mmwchanmod.learn.models import ChanMod
from mmwchanmod.datasets.download import get_dataset

"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Trains the channel model')
parser.add_argument('--nlatent',action='store',default=20,type=int,\
    help='number of latent variables')
parser.add_argument('--npaths_max',action='store',default=20,type=int,\
    help='max number of paths per link')
parser.add_argument('--nepochs_link',action='store',default=10,type=int,\
    help='number of epochs for training the link model')
parser.add_argument('--lr_link',action='store',default=1e-3,type=float,\
    help='learning rate for the link model')   
parser.add_argument('--nepochs_path',action='store',default=5000,type=int,\
    help='number of epochs for training the path model')
parser.add_argument('--lr_path',action='store',default=1e-4,type=float,\
    help='learning rate for the path model')     
parser.add_argument('--out_var_min',action='store',default=1e-4,type=float,\
    help='min variance in the decoder outputs.  Used for conditioning')     
parser.add_argument('--init_stddev',action='store',default=10.0,type=float,\
    help='weight and bias initialization')
parser.add_argument('--nunits_enc',action='store',nargs='+',\
    default=[200,80],type=int,\
    help='num hidden units for the encoder')    
parser.add_argument('--nunits_dec',action='store',nargs='+',\
    default=[80,200],type=int,\
    help='num hidden units for the decoder')    
parser.add_argument('--nunits_link',action='store',nargs='+',\
    default=[50,25],type=int,\
    help='num hidden units for the link state predictor')        
parser.add_argument('--model_dir',action='store',\
    default= '../../models/uav_boston', help='directory to store models')
parser.add_argument('--no_fit_link', dest='no_fit_link', action='store_true',\
    help="Does not fit the link model")
parser.add_argument('--no_fit_path', dest='no_fit_path', action='store_true',\
    help="Does not fit the path model")
parser.add_argument('--checkpoint_period',action='store',default=500,type=int,\
    help='Period in epochs for storing checkpoint.  A value of 0 indicates no checkpoints')    
parser.add_argument('--batch_ind',action='store',default=-1,type=int,\
    help='batch index.  -1 indicates no batch index')  
parser.add_argument(\
    '--data_dir',action='store',\
    default='../../data', help='directory of the data file')
parser.add_argument(\
    '--ds_name',action='store',\
    default='uav_lon_tok', help='dataset within the data directory')
    

args = parser.parse_args()
nlatent = args.nlatent
npaths_max = args.npaths_max
nepochs_path = args.nepochs_path
lr_path = args.lr_path
nepochs_link = args.nepochs_link
lr_link = args.lr_link
init_stddev = args.init_stddev
nunits_enc = args.nunits_enc
nunits_dec = args.nunits_dec
nunits_link = args.nunits_link
model_dir = args.model_dir
batch_ind = args.batch_ind
out_var_min = args.out_var_min
fit_link = not args.no_fit_link
fit_path = not args.no_fit_path
checkpoint_period = args.checkpoint_period
data_dir = args.data_dir
ds_name = args.ds_name


# Overwrite parameters based on batch index
# This is used in HPC training with multiple batches
#lr_batch = [1e-3,1e-3,1e-3,1e-4,1e-4,1e-4]
nlatent_batch  = [20,30]
nunits_enc_batch = [[200,80], [300,100]]
nunits_dec_batch = [[80,200], [100,300]]
ds_name_batch = ['uav_lon_tok', 'uav_beijing', 'uav_boston', 'uav_moscow']
dir_suffix = ['nl20_nh2_20201106', 'nl30_nh2_20201106']    
if batch_ind >= 0:
    nparam = len(nlatent_batch)
    ids = batch_ind // nparam
    iparam = batch_ind % nparam
    ds_name = ds_name_batch[ids]
    model_dir = ('../../models/%s_%s' % (ds_name, dir_suffix[iparam]) )
    #lr_path = lr_batch[batch_ind]
    nlatent = nlatent_batch[iparam]
    nunits_enc = nunits_enc_batch[iparam]
    nunits_dec = nunits_dec_batch[iparam]
    
# Print parameters
if batch_ind >= 0:
    print('batch_ind=%d' % batch_ind)
print('model_dir= %s' % model_dir)
print('nunits_enc=%s' % str(nunits_enc))
print('nunits_dec=%s' % str(nunits_dec))
print('lr_path=%12.4e' % lr_path)
print('nlatent=%d' % nlatent)
print('data_fn=%s' % ds_name)
    
"""
Build the model
"""

# Load the data
data = get_dataset(ds_name)
cfg = data['cfg']
train_data = data['train_data'] 
test_data = data['test_data'] 

# Construct the channel model object
K.clear_session()
chan_mod = ChanMod(nlatent=nlatent,cfg=cfg,\
                   nunits_enc=nunits_enc, nunits_dec=nunits_dec,\
                   nunits_link=nunits_link,\
                   out_var_min=out_var_min,\
                   init_bias_stddev=init_stddev,\
                   init_kernel_stddev=init_stddev, model_dir=model_dir)    
chan_mod.save_config()

"""
Train the link classifier
"""

if fit_link:
    # Build the link model
    chan_mod.build_link_mod()

    # Fit the link model 
    chan_mod.fit_link_mod(train_data, test_data, lr=lr_link,\
                          epochs=nepochs_link)
    
    # Save the link classifier model
    chan_mod.save_link_model()
    
else:
    # Load the link model
    chan_mod.load_link_model()  

"""
Train the path loss model
"""
if fit_path:
    chan_mod.build_path_mod()
    
    chan_mod.fit_path_mod(train_data, test_data, lr=lr_path,\
                          epochs=nepochs_path,\
                          checkpoint_period=checkpoint_period)
        




