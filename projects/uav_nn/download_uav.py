"""
download_uav.py:  Downloads the UAV datasets

Run this file before training.
"""

# Add the path
import os
import sys
path = os.path.abspath('../..')
if not path in sys.path:
    sys.path.append(path)
from mmwchanmod.datasets.download import list_datasets, get_dataset, concat_datasets

# Datasets to download
ds_names = ['uav_london', 'uav_tokyo']

# Concatanated dataset
out_name = 'uav_lon_tok'

# Downloads the datasets from the server, if needed
for ds in ds_names:
    get_dataset(ds, return_data=False)
        
# Create concatanted dataset
concat_datasets(ds_names, out_name)

# Print list
list_datasets(src='local')

