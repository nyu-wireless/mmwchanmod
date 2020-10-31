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
    'uav_moscow' : '1dKZD9klTUzKJOGkV9ib01QE8y-qO3xL3'\
}


def list_datasets(prt=True):
    """
    Lists all datasets available
    
    Parameters
    ----------
    prt:  Boolean, default: True
        Prints the datasets

    Returns
    -------
    ds_names:  list of strings
        List of all available dataset names
    """
    
    ds_names = ds_gdrive_ids.keys()
    if prt:
        print('Available datasets:')
        for name in ds_names:
            print('  ', name)            
            
    return ds_names
    
                  
def get_dataset(ds_name, overwrite=False, return_data=True):
    """
    Gets a dataset

    Parameters
    ----------
    ds_name : string
        Dataset to be downloaded. 
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
        
    # Create the data directory if needed
    data_dir = '../data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        print('Creating directory %s' % data_dir)
                
    # Download the path
    ds_fn = ds_name + '.p'
    ds_path = os.path.join('../data', ds_fn)    
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
    
    # Exit if data does not need to be returned
    if not return_data:
        return
        
    # Load the pickle file
    with open(ds_path, 'rb') as fp:
        cfg, train_data, test_data = pickle.load(fp)   
        
    data = {'cfg' : cfg, 'train_data' : train_data, 'test_data' : test_data}
    return data
        
    
    
        
            
    
    
    
    


