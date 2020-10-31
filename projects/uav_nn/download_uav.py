# -*- coding: utf-8 -*-
"""
download_uav.py:  Downloads the UAV datasets
"""
import os
import sys
path = os.path.abspath('../..')
if not path in sys.path:
    sys.path.append(path)
    
from mmwchanmod.datasets.download import get_dataset

datasets = ['uav_boston', 'uav_london', 'uav_moscow']

for ds in datasets:
    get_dataset(ds, return_data=False)
