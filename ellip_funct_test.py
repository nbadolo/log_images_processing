#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:01:47 2023

@author: nbadolo
"""

import numpy as np
from skimage import color, data, restoration, io, color, measure, draw, img_as_bool
from astropy.nddata import Cutout2D
from astropy.io import fits
from math import pi, cos, sin, atan
import scipy.optimize as opt
from AymardPack import DelHotPix


nDim=1024
nSubDim = 200 # plage de pixels que l'on veut afficher
size = (nSubDim, nSubDim)
nDimfigj=[9,10,11]
nDimfigk=[0,1,2]
lst_threshold = [0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]
n_threshold = len(lst_threshold)
pix2mas = 3.4  #en mas/pix
x_min = -pix2mas*nSubDim//2
x_max = pix2mas*(nSubDim//2-1)
y_min = -pix2mas*nSubDim//2
y_max = pix2mas*(nSubDim//2-1)
position = (nDim//2,nDim//2)
size = (nSubDim, nSubDim)

sub_v_star_arr = np.zeros((1, nSubDim, nSubDim))

star_name = 'SW_Col'
obsmod = 'both'
fltr = 'V_N_R'
fdir= '/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+ '/'
fdir_star = fdir + 'star/'+obsmod+ '/' 

fdir_star_fltr = fdir_star + fltr +'/'
#fdir_psf_fltr = fdir_psf + fltr + '/'
fname1='zpl_p23_make_polar_maps-ZPL_SCIENCE_P23_REDUCED'
fname2='-zpl_science_p23_REDUCED'

file_I_star = fdir_star_fltr + fname1+'_I'+fname2+'_I.fits'

image = file_I_star

# #simple
# hdu = fits.open(image)   
# data = hdu[0].data
# intensity = data[1,:,:]
# cutout = Cutout2D(intensity, position = position, size = size)
# zoom_hdu = hdu.copy()
# sub_v_star = cutout.data

#with prossessing
hdu = DelHotPix(image)
data = hdu[0].data
intensity = data[1,:,:]
cutout = Cutout2D(intensity, position = position, size = size)
zoom_hdu = hdu.copy()
sub_v_star = cutout.data

sub_v_star_arr[0] = sub_v_star

Ellips_star_im = np.zeros_like(sub_v_star_arr[0]) #creation d'un tableau de meme forme que sub_v
Ellips_star_im[sub_v_star > lst_threshold[0]*np.max(sub_v_star)] = sub_v_star[sub_v_star > lst_threshold[0]*np.max(sub_v_star)]# on retient les points d'intensité égale à 5% de Imax 
#Ellips_star_im_arr[i][j] = Ellips_star_im

Ellips_star = np.zeros_like(sub_v_star)          #creation d'un tableau de meme forme que sub_v
Ellips_star[sub_v_star > lst_threshold[0]*np.max(sub_v_star)] = 1   # on retient les points d'intensité 
#Ellips_star_arr[i][] = Ellips_star                # égale à 5% de Imax et à tous ces points 
     
im_star_white =Ellips_star                
regions_str = measure.regionprops(measure.label(im_star_white))
bubble_str = regions_str[0]

# initial guess (must be to change related on the % considered)
ys_i, xs_i = bubble_str.centroid
as_i = bubble_str.major_axis_length / 2.
bs_i = bubble_str.minor_axis_length / 2.
thetas_i  = pi/4
t = np.linspace(0, 2*pi, nSubDim)

def cost(params_s):
    x0s, y0s, a0s, b0s, thetas = params_s
    #coords = draw.disk((y0, x0), r, shape=image.shape)
    coords_s = draw.ellipse(y0s, x0s, a0s, b0s, shape=None, rotation= thetas)
    print('la forme de coord est ' + str(np.shape(coords_s)))
    template_star = np.zeros_like(im_star_white)
    template_star[coords_s] = 1
    print('la forme de template_star est ' + str( np.shape(template_star)))
    return -np.sum(template_star == im_star_white)
x_sf, y_sf, a_sf, b_sf, theta_sf = opt.fmin(cost, (xs_i, ys_i, as_i, bs_i, thetas_i))