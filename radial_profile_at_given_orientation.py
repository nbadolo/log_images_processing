#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:07:31 2022

@author: nbadolo
"""


# =============================================================================
# Module for fitting  a radial profile at a given orientation 
# =============================================================================



import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt



nDim=1024
nSubDim = 200 # plage de pixels que l'on veut afficher


#-- Loading data...

z  = sub_v_arr[i]
#-- Extract the line...
# Make a line with "num" points...

x1, y1 = x_f, y_f
x2, y2 = 0, y_f - theta_f*x_f

num = 1000
# x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
x, y = np.linspace(x1, x2, num), np.linspace(y1, y2, num)

# Extract the values along the line, using cubic interpolation
zi = scipy.ndimage.map_coordinates(z, np.vstack((x,y))) #  les données pour le profile radial

#-- Plot...
fig, axes = plt.subplots(nrows=2)
axes[0].imshow(z)
axes[0].plot([x0, x1], [y0, y1], 'ro-')
axes[0].axis('image')

axes[1].plot(zi)

plt.show()
