"""
preproc_param.py:  Methods for parametrizing sklearn pre-processing classes
"""

import numpy as np
import sklearn.preprocessing

def preproc_to_param(preproc, preproc_type):
    """
    Extracts the parameters of certain sklearn objects to a dictionary.
    This is used if sklearn objects are serialized, they may not be
    compatible across different versions.  So, instead we convert 
    to a parameter dictionary and serialize that.
    

    Parameters
    ----------
    preproc : sklearn object
        Pre-processor to serialize
    preproc_type : 'StandardScaler', 'OneHotEncoder' or 'MinMaxScaler'
        Type of pre-processing object


    Returns
    -------
    param : dictionary
        Dictionary with the parameters of the object

    """
    param = dict()
    if preproc_type == 'StandardScaler':
        param['mean'] = preproc.mean_
        param['scale'] = preproc.scale_
    elif preproc_type == 'OneHotEncoder':        
        param['categories'] = preproc.categories_  
    elif preproc_type == 'MinMaxScaler':        
        param['data_min'] = preproc.data_min_
        param['data_max'] = preproc.data_max_
    else:
        raise ValueError('Unknown preprocessor %s' % preproc_type)
        
    return param

def param_to_preproc(param, preproc_type):
    """
    Creates an sklearn pre-processor from a set of saved parameters.
    This is used for de-serializing he object.
    

    Parameters
    ----------
    param : dictionary
        Dictionary with the parameters of the object
    preproc_type : 'StandardScaler', 'OneHotEncoder' or 'MinMaxScaler'
        Type of pre-processing object

    Returns
    -------
    preproc : sklearn object
        Pre-processor to serialize

    """
    if preproc_type == 'StandardScaler':
        mean = param['mean']
        scale = param['scale']
        
        # Create ficticious data with that mean and scale
        nsamp = 100
        u = np.linspace(-1,1,nsamp)
        u = u - np.mean(u)
        u = u / np.std(u)    
        X = mean[None,:] + u[:,None]*scale[None,:]
    
    
        # Create a new scaler and fit it with the fake data to get 
        # the same parameters as the old scaler
        p = sklearn.preprocessing.StandardScaler()
        p.fit(X)
    elif preproc_type == 'OneHotEncoder':
        
        # Create ficticious data with the categories
        cat = param['categories']
        u = cat[0][:,None]
        
        p = sklearn.preprocessing.OneHotEncoder(sparse=False)
        p.fit(u)
        
    elif preproc_type == 'MinMaxScaler':
        dmin = param['data_min']
        dmax = param['data_max']
        drange = dmax - dmin
        
        # Create ficticious data with that mean and scale
        nsamp = 100
        u = np.linspace(0,1,nsamp)
        X = u[:,None]*drange[None,:] + dmin[None,:]
    
    
        # Create a new scaler and fit it with the fake data to get 
        # the same parameters as the old scaler
        p = sklearn.preprocessing.MinMaxScaler()
        p.fit(X)
    
        
    else:
        raise ValueError('Unknown preprocessor %s' % preproc_type)
        
    return p
