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
from scipy.stats import sigmaclip
from AymardPack import DelHotPix
from AymardPack import EllRadialProf as erp
import matplotlib.pyplot as plt

nDim=1024
nSubDim = 200 # plage de pixels que l'on veut afficher
size = (nSubDim, nSubDim)
# nDimfigj=[9,10,11]
# nDimfigk=[0,1,2]
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
par_star_arr = np.zeros((5))
Ell_rot_star_arr = np.zeros((2, nSubDim))

star_name = 'SW_Col'
obsmod = 'both'
fltr = 'V_N_R'
fdir= '/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+ '/'
fdir_star = fdir + 'star/'+obsmod+ '/' 

fdir_star_fltr = fdir_star + fltr +'/'
#fdir_psf_fltr = fdir_psf + fltr + '/'
fname1='zpl_p23_make_polar_maps-ZPL_SCIENCE_P23_REDUCED'
fname2='-zpl_science_p23_REDUCED'

#file_I_star = fdir_star_fltr + fname1+'_I'+fname2+'_I.fits'   # Intensité
file_I_star = fdir_star_fltr + fname1+'_PI'+fname2+'_PI.fits' # Intensité polarisée

image = file_I_star

# #simple
# hdu = fits.open(image)   
# data = hdu[0].data
# intensity = data[1,:,:]
# cutout = Cutout2D(intensity, position = position, size = size)
# zoom_hdu = hdu.copy()
# sub_v_star = cutout.data

#with prossessing
#hdu = DelHotPix(image)
hdu = fits.open(image)
data = hdu[0].data

#data = np.array(data, dtype=np.float32)
#data = sigmaclip(data.astype(float))
#data = hdu[0].data
intensity = data[1,:,:]
cutout = Cutout2D(intensity, position = position, size = size)
zoom_hdu = hdu.copy()
sub_v_star = cutout.data

sub_v_star_arr[0] = sub_v_star
maxi = np.argmax(sub_v_star)
maxi = np.unravel_index(maxi, np.shape(sub_v_star))
print('le pixel le plus brillant est '  + str(maxi)) 
Ellips_star_im = np.zeros_like(sub_v_star_arr[0]) #creation d'un tableau de meme forme que sub_v
Ellips_star_im[sub_v_star > lst_threshold[0]*np.max(sub_v_star)] = sub_v_star[sub_v_star > lst_threshold[0]*np.max(sub_v_star)]# on retient les points d'intensité égale à 5% de Imax 
#Ellips_star_im_arr[i][j] = Ellips_star_im
#stop
Ellips_star = np.zeros_like(sub_v_star)          #creation d'un tableau de meme forme que sub_v
Ellips_star[sub_v_star > lst_threshold[0]*np.max(sub_v_star)] = 1   # on retient les points d'intensité 
#Ellips_star_arr[i][] = Ellips_star                # égale à 5% de Imax et à tous ces points 
     
im_star_white = Ellips_star                
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

theta_sf = np.pi/2 -theta_sf
par_star_arr = [x_sf, y_sf, a_sf, b_sf, theta_sf]

theta_sf_deg = theta_sf*180/pi
Ell_star = np.array([a_sf*np.cos(t) , b_sf*np.sin(t)])  
       #u,v removed to keep the same center location
M_rot_star = np.array([[cos(theta_sf) , -sin(theta_sf)],[sin(theta_sf) , cos(theta_sf)]]) 

       #2-D rotation matrix
  
Ell_rot_star_ = np.zeros((2, nSubDim))
Ell_rot_star = np.zeros((2, nSubDim))
for str_k in range(Ell_star.shape[1]):
    Ell_rot_star_[:,str_k] = np.dot(M_rot_star,Ell_star[:,str_k])  #fait le produit scal de la matrice de rotation par chaq couple parametriq 
    Ell_rot_star[:,str_k] = Ell_rot_star_[:,str_k]
    Ell_rot_star[0,str_k] = Ell_rot_star[0,str_k] + x_sf
    Ell_rot_star[1,str_k] = Ell_rot_star[1,str_k] + y_sf
    #return Ell_rot.ravel() # .ravel permet de passer de deux dimension à une seule
    Ell_rot_star_arr[:,str_k] = Ell_rot_star[:,str_k] 
    
    
im_s  = np.log10(sub_v_star_arr[0] + np.abs(np.min(sub_v_star_arr[0])) + 10) # intensity for the radial profile 
#imp_s = np.log10(sub_v_star_arr[1] + np.abs(np.min(sub_v_star_arr[1])) + 10) # polarized  intensity for the radial profile 
x0_s, y0_s, x1_s, y1_s, x2_s, y2_s,z_s, zi1_s, zi2_s, xx1_s, yy1_s, xx2_s, yy2_s, zzi1_s, zzi2_s = erp(par_star_arr[0], par_star_arr[1], par_star_arr[2], par_star_arr[3], par_star_arr[4], im_s, 100)
#x0p_s, y0p_s, x1p_s, y1p_s, x2p_s, y2p_s,zp_s, zi1p_s, zi2p_s, xx1p_s, yy1p_s, xx2p_s, yy2p_s, zzi1p_s, zzi2p_s = erp(par_star_arr[1][0][0], par_star_arr[1][0][1], par_star_arr[1][0][2], par_star_arr[1][0][3], par_star_arr[1][0][4], imp_s, 100)


plt.figure('profiles comparison')
plt.clf()
#plt.subplot(3,2,1) # De l'intensité
plt.imshow(z_s, cmap ='inferno', vmin = np.min(z_s), vmax = np.max(z_s), origin='lower') # full image of intensity
plt.plot(Ell_rot_star_arr[0,:] , Ell_rot_star_arr[1,:]) # ellipse de l'intensité à 10% de l'intensité max
##plt.plot([x0_s, x1_s], [y0_s, y1_s], 'ro-')  # tracé du demi-grand axe de l'ellipse
#plt.plot([x0, x1_], [y0, y1_], 'ro-')# tracé de l'autre demi-grand axe de l'ellipse
##plt.plot([x0_s, x2_s], [y0_s, y2_s], 'ro-') # tracé du demi petit axe de l'ellipse
#plt.plot([x0, x2_], [y0, y2_], 'ro-') # tracé de l'autre demi petit axe de l'ellipse
plt.xlabel('Relative R.A.(pix)', size=10)
plt.ylabel('Relative Dec.(pix)', size=10)
plt.title('radial p')