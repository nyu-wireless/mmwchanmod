"""
download.py:  Downloads files

Programs are taken from  StackOverflow answer: https://stackoverflow.com/a/39225039
"""

import os
import zipfile
import shutil
import requests
import argparse
from tqdm import tqdm
import pickle
import numpy as np


def download_file_from_gdrive(gdrive_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    
    print('Downloading %s' % destination)

    session = requests.Session()

    response = session.get(URL, params = { 'id' : gdrive_id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : gdrive_id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    
    # Total size in MB.
    total_size = int(response.headers.get('content-length', 0)); 
  
    with open(destination, "wb") as f:
        with tqdm(total=total_size//CHUNK_SIZE, unit='kB', 
                           unit_scale=True, unit_divisor=1024) as pbar:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    pbar.update(1)
                    f.write(chunk)    
    
# Google drive names
ds_gdrive_ids = {\
    'uav_boston' : '1G748lB9jDKaAs5-QSMMCKC0N2eK_jW1M',\
    'uav_london' : '19nHl-MFRTOflwnDTN4oJt60ZMX6RZHgW',\
    'uav_tokyo'  : '1RbHh-9M70lRaZmzDOeYhPoSfzHIaBfJ2',\
    'uav_moscow' : '1dKZD9klTUzKJOGkV9ib01QE8y-qO3xL3',\
    'uav_beijing': '145HEyB_oHCMZIb3rYE-2iQpHlLdg5PPn'
}


def list_datasets(src='remote', prt=True):
    """
    Lists all datasets available
    
    Parameters
    ----------
    prt:  Boolean, default: True
        Prints the datasets
    src : {'remote', 'local'}:
        'remote' indicates to list files on server.
        'local' indicates to list files on local 

    Returns
    -------
    ds_names:  list of strings
        List of all available dataset names
    """
    
    if src == 'remote':
        ds_names = list(ds_gdrive_ids.keys())
              
    elif src == 'local':
        data_dir = os.path.join(os.path.dirname(__file__),'..','..','data')  
        data_dir = os.path.abspath(data_dir)
        ds_names = []
        if not os.path.isdir(data_dir):
            print('No local data directory %s' % data_dir)            
        else:
            # Add all files ending with .p, but remove the .p
            for f in  os.scandir(data_dir):
                name = f.name
                if name.endswith('.p'):
                    ds_names.append(name[:-2])
            
    # Print the list
    if prt:
       print('Available datasets:')
       for name in ds_names:
           print('  ', name)        
    return ds_names
    
                  
def get_dataset(ds_name, src='remote', overwrite=False, return_data=True):
    """
    Gets a dataset

    Parameters
    ----------
    ds_name : string
        Dataset to be downloaded. 
    src : {'remote', 'local'}:
        'remote' indicates to download from server.  
        'local' indicates to download from local directory-
    overwrite : boolean, default: False
        Overwrites dataset if already downloaded
    return_data : boolean, default: True
        Returns the data dictionary.  If False, the file
        is only downloaded.
        
        
    Returns
    -------
    data:  dictionary
        data['cfg']:  configuration of the dataset
        data['train_data']:  training data dictionary
        data['test_data']:  test data dictionary
    """    
        
    # Create the local data directory if needed    
    data_dir = os.path.join(os.path.dirname(__file__),'..','..','data')
    data_dir = os.path.abspath(data_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        print('Creating directory %s' % data_dir)
        
    # Get the dataset path
    ds_fn = ds_name + '.p'
    ds_path = os.path.join(data_dir, ds_fn)    
            
    # Download from the server, if needed
    if src == 'remote':
        
        
        if (not os.path.exists(ds_path)) or overwrite:
            # Check if dataset is valid
            if not (ds_name in ds_gdrive_ids):
                raise ValueError('Unknown dataset %s' % ds_name)
    
            # Get gdrive ID
            gdrive_id = ds_gdrive_ids[ds_name]
            
            # Download the file
            download_file_from_gdrive(gdrive_id, ds_path)
        elif not return_data:
            print('Data file %s already available' % ds_path)
    elif src != 'local':
        raise ValueError('src must be local or remote')
    
    # Exit if data does not need to be returned
    if not return_data:
        return
        
    # Load the pickle file from the local directory
    with open(ds_path, 'rb') as fp:
        cfg, train_data, test_data = pickle.load(fp)   
        
    data = {'cfg' : cfg, 'train_data' : train_data, 'test_data' : test_data}
    return data


def concat_datasets(ds_names, out_name='concat'):
    """
    Concantenates a set of datasets.
    
    Parameters
    ----------
    ds_name : array of strings
        Datasets to be merged.  Each dataset must be in the local
        dataset directory.  Use the `get_dataset()` command to download them.
    out_name:  string
        Name for the concatanted dataset.  A suffix `.p` will be added.
    """
    
    # Get data directory
    data_dir = os.path.join(os.path.dirname(__file__),'..','..','data')
    if not os.path.isdir(data_dir):
        raise ValueError('Could not find data directory %s' % data_dir)
        
    if len(ds_names) == 0:
        raise ValueError('Dataset list is empty')
        
    nlinks_tr = 0
    nlinks_ts = 0
    desc = 'Concatenation of '
    for ds_name in ds_names:
        # Get directory path
        ds_fn = ds_name + '.p'
        ds_path = os.path.join(data_dir, ds_fn)    
        if (not os.path.exists(ds_path)):
            err_str = 'Dataset file %s not found.' % ds_path
            err_str = err_str + os.sep +\
                'Use the get_dataset to load the dataset first'
            raise ValueError(err_str)
        
        # Load the pickle file
        with open(ds_path, 'rb') as fp:
            cfgi, train_datai, test_datai = pickle.load(fp)    
            
        # Get the number of links
        nlinks_tri = train_datai['dvec'].shape[0]
        nlinks_tsi = test_datai['dvec'].shape[0]
        nlinks_tr += nlinks_tri
        nlinks_ts += nlinks_tsi
        
        print('%-10s: train:  %d links, test: %d links' %\
              (ds_name, nlinks_tri, nlinks_tsi))
        
        # Concatenate items in dataset
        train_data = None
        test_data = None
        if train_data is None:
            train_data = train_datai
            test_data = test_datai
        else:
            for k in train_data:
                vtr = train_data[k]
                vts = test_data[k]
                if len(vtr.shape) == 1:
                    train_data[k] = np.hstack((train_data[k], vtr))
                    test_data[k] = np.hstack((test_data[k], vts))
                else:
                    train_data[k] = np.vstack((train_data[k], vtr))
                    test_data[k] = np.vstack((test_data[k], vts))
                    
                
    print('%-10s: train:  %d links, test: %d links' %\
      ('Total', nlinks_tr, nlinks_ts))
      
    # Set configuration
    desc = 'Concatenation of ' + ', '.join(ds_names)
    cfg = cfgi
    cfg.desc = desc       
    
    # Write the file
    data_fn = out_name + '.p'
    data_path = os.path.join(data_dir, data_fn)
    with open(data_path, 'wb') as fp:
        pickle.dump([cfg, train_data, test_data], fp)  
    print('Created concatanated dataset: %s' % data_fn)
                        
    
            
    
    
    
    


