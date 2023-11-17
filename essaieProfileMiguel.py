#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:08:39 2023

@author: nbadolo
"""

import numpy as np
import os
from lib_zpl_analysis import*
from glob import glob


prefix = '/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/SW_Col/star/both/V_N_R/Profile_fits/'
sufix = 'zpl_p23_make_polar_maps-ZPL_SCIENCE_P23_REDUCED_I-zpl_science_p23_REDUCED_I.fits'


infile = prefix + sufix
#file_list = glob(os.path.join(prefix, '*.fits'))
# print(file_list)
# print(len(file_list))
indir = os.path.dirname(prefix)
#indir = prefix
#print(indir)
#print(indir)
#stop 
plot_all_radial_profiles(indir, 'pdf', sup_poldeg=None, sqrtInt=False)