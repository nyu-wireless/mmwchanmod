# -*- coding: utf-8 -*-
"""
download_uav.py:  Downloads the UAV datasets
"""
import os
import sys
path = os.path.abspath('../..')
if not path in sys.path:
    sys.path.append(path)
    
from mmwchanmod.datasets.donwload import get_dataset

get_dataset('uav_boston')
get_dataset('uav_london')
