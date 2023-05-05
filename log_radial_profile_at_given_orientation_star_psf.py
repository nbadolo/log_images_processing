#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:06:04 2022

@author: nbadolo
"""



import numpy as np
import os
from astropy.io import fits
import math
from math import pi, cos, sin, atan
from scipy import optimize, signal
from scipy.signal import convolve2d as conv2
from AymardPack import Margaux_RL_deconv
from AymardPack import EllRadialProf as erp
from skimage import color, data, restoration
from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt
import scipy.optimize as opt
from skimage import io, color, measure, draw, img_as_bool
import matplotlib.colors as colors
from matplotlib.pyplot import Figure, subplot

#%% 
def log_image(star_name, obsmod):   
#%%        
    fdir= '/home/nbadolo/Bureau/Aymard/Donnees_sph/log/'+star_name+ '/'
    fdir_star = fdir + 'star/'+obsmod+ '/' 
    fdir_psf = fdir +'psf/'+obsmod+ '/'
    lst_fltr_star = os.listdir(fdir_star)
    n_lst_fltr_star = len(lst_fltr_star)
    lst_fltr2_star = []
    nDimfigj = [3, 4, 5]
    nDimfigk = [6, 7, 8]
    for p in range(n_lst_fltr_star):
        
        fdir_fltr_data_star = fdir_star + lst_fltr_star[p]
        lst_fltr_data_star = os.listdir(fdir_fltr_data_star)
        n_lst_fltr_data_star = len(lst_fltr_data_star)
        if n_lst_fltr_data_star != 0:
            lst_fltr2_star.append(lst_fltr_star[p])
    #print(lst_fltr2_star)
    
    
    lst_fltr_psf = os.listdir(fdir_psf)
    n_lst_fltr_psf = len(lst_fltr_psf)
    lst_fltr2_psf = []
    for n in range(n_lst_fltr_psf):
        
        fdir_fltr_data_psf = fdir_psf + lst_fltr_psf[n]
        lst_fltr_data_psf = os.listdir(fdir_fltr_data_psf) 
        n_lst_fltr_data_psf = len(lst_fltr_data_psf)
        if n_lst_fltr_data_psf != 0:
            lst_fltr2_psf.append(lst_fltr_psf[n])
    #print(lst_fltr2_psf)
    
    lst_fltr3 = list(set(lst_fltr2_star).intersection(lst_fltr2_psf))
    #print(lst_fltr3)
    n_lst_fltr3 = len(lst_fltr3)
    #print(n_lst_fltr3)
    for l in range(n_lst_fltr3):
        fdir_star_fltr = fdir_star + lst_fltr3[l] +'/'
        fdir_psf_fltr = fdir_psf + lst_fltr3[l] + '/'
        
        fname1='zpl_p23_make_polar_maps-ZPL_SCIENCE_P23_REDUCED'
        fname2='-zpl_science_p23_REDUCED'
        file_I_star= fdir_star_fltr + fname1+'_I'+fname2+'_I.fits'
        file_PI_star= fdir_star_fltr +fname1+'_PI'+fname2+'_PI.fits'
        file_DOLP_star= fdir_star_fltr +fname1+'_DOLP'+fname2+'_DOLP.fits'
        #file_AOLP_star= fdir_star_fltr + fname1+'_AOLP'+fname2+'_AOLP.fits'
    
        file_I_psf = fdir_psf_fltr + fname1+'_I'+fname2+'_I.fits'
        file_PI_psf = fdir_psf_fltr +fname1+'_PI'+fname2+'_PI.fits'
        file_DOLP_psf = fdir_psf_fltr + fname1+'_DOLP'+fname2+'_DOLP.fits'
        #file_AOLP_psf = fdir_psf_fltr +fname1+'_AOLP'+fname2+'_AOLP.fits'
      
        file_lst = [file_I_star,file_PI_star,
                  file_I_psf,file_PI_psf]
        
        file_star_lst= [file_I_star, file_PI_star]
        file_psf_lst = [file_I_psf, file_PI_psf]          
        
        nFrames = len(file_star_lst)
        nFrames_star = len(file_star_lst)
        nFrames_psf = len(file_psf_lst)
        nFrames_d = nFrames_star
        
       #parameters
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
        lst_Frame_name = ['Intensity', 'Pol_Intensity']
        
        
        # lists and arrays of the star
        t = np.linspace(0, 2*pi, nSubDim)
        x, y = np.meshgrid(np.arange(nSubDim),np.arange(nSubDim)) #cree un tableau 
        sub_v_star_arr = np.zeros((nFrames, nSubDim, nSubDim))
        Ellips_star_arr = np.zeros((nFrames, n_threshold,nSubDim,nSubDim))
        Ellips_star_im_arr = np.zeros((nFrames, n_threshold,nSubDim,nSubDim))
        Ellips_star_arr2 = np.zeros((n_threshold, nSubDim, nSubDim))
        Ellips_star_ = np.zeros((n_threshold, nSubDim))
        Ell_rot_star_arr = np.zeros((nFrames, n_threshold, 2, nSubDim))
        ind_star_arr  = np.zeros((nFrames,2)) # pour les indices des pixels les plus brillants(photcentres)
        ind_psf_arr  = np.zeros((nFrames,2)) # pour les indices des pixels les plus brillants(photcentres)
        par_star_arr = np.zeros((nFrames, n_threshold, 5)) # les paramètres de l'étoile
        Vmin_star = np.zeros((nFrames, n_threshold)) # pour l'image totale
        Vmax_star = np.zeros((nFrames, n_threshold))
        Vmin_star_r = np.zeros((nFrames, n_threshold))
        Vmax_star_r = np.zeros((nFrames, n_threshold))# pour l'image reelle tronquee à au pourcentage
        Vmin_star_w = np.zeros((nFrames, n_threshold))
        Vmax_star_w = np.zeros((nFrames, n_threshold)) # pour l'image binaire
        
        
        # lists and arrays of the psf   
        x, y = np.meshgrid(np.arange(nSubDim),np.arange(nSubDim)) #cree un tableau 
        sub_v_psf_arr = np.zeros((nFrames, nSubDim, nSubDim))
        Ellips_psf_arr = np.zeros((nFrames, n_threshold,nSubDim,nSubDim))
        Ellips_psf_im_arr = np.zeros((nFrames, n_threshold,nSubDim,nSubDim))
        Ellips_psf_arr2 = np.zeros((n_threshold, nSubDim, nSubDim))
        Ellips_psf_ = np.zeros((n_threshold, nSubDim))
        Ell_rot_psf_arr = np.zeros((nFrames, n_threshold, 2, nSubDim))
        par_psf_arr = np.zeros((nFrames, n_threshold, 5)) # les paramètres de l'étoile
        Vmin_psf = np.zeros((nFrames, n_threshold)) # pour l'image totale
        Vmax_psf = np.zeros((nFrames, n_threshold))
        Vmin_psf_r = np.zeros((nFrames, n_threshold))
        Vmax_psf_r = np.zeros((nFrames, n_threshold))# pour l'image reelle tronquee à au pourcentage
        Vmin_psf_w = np.zeros((nFrames, n_threshold))
        Vmax_psf_w = np.zeros((nFrames, n_threshold)) # pour l'image binaire
        
        strs = [str(x*100) + ' %' for x in lst_threshold]
        
        
        #opening star file
        
        for i in range(nFrames_star):
              hdu = fits.open(file_lst[i])   
              data = hdu[0].data
              intensity = data[1,:,:]
              
              cutout = Cutout2D(intensity, position = position, size = size)
              zoom_hdu = hdu.copy()
              sub_v_star = cutout.data
              
              sub_v_star_arr[i] = sub_v_star
              
              # =========================================================#
              #  Pour le calcul du decalage des centres de l'étoile      #
              # =========================================================#
                  
              
              #  Determination des coordonnées du pixel le plus brillant
                
              a_s = sub_v_star_arr[i]
              ind_star = np.unravel_index(np.argmax(a_s, axis = None), a_s.shape)
              ind_star_arr[i] = ind_star
              print(ind_star)
              
              for j in range(n_threshold):
                  Ellips_star_arr2[j] = sub_v_star
                  Ellips_star_im = np.zeros_like(sub_v_star_arr[i]) #creation d'un tableau de meme forme que sub_v
                  Ellips_star_im[sub_v_star > lst_threshold[j]*np.max(sub_v_star)] = sub_v_star[sub_v_star > lst_threshold[j]*np.max(sub_v_star)]# on retient les points d'intensité égale à 5% de Imax 
                  Ellips_star_im_arr[i][j] = Ellips_star_im
                  
                  Ellips_star = np.zeros_like(sub_v_star)          #creation d'un tableau de meme forme que sub_v
                  Ellips_star[sub_v_star > lst_threshold[j]*np.max(sub_v_star)] = 1   # on retient les points d'intensité 
                  Ellips_star_arr[i][j] = Ellips_star                # égale à 5% de Imax et à tous ces points 
                                                         # on donne 1 comme valeur d'intensité
            
                    
                  Vmin_star[i][j] = np.min(np.log10(sub_v_star+np.abs(np.min(sub_v_star))+10))
                  Vmax_star[i][j] = np.max(np.log10(sub_v_star+np.abs(np.min(sub_v_star))+10))
                  
                  Vmin_star_r[i][j] = np.min(np.log10(Ellips_star_im+np.abs(np.min(Ellips_star_im))+10))
                  Vmax_star_r[i][j] = np.max(np.log10(Ellips_star_im+np.abs(np.min(Ellips_star_im))+10)) 
                  
                  Vmin_star_w[i][j] = np.min(np.log10(Ellips_star+np.abs(np.min(Ellips_star))+10))
                  Vmax_star_w[i][j] = np.max(np.log10(Ellips_star+np.abs(np.min(Ellips_star))+10))  
             
              
                  im_star_white = Ellips_star_arr[i][j]
                  im_star_real = Ellips_star_im_arr[i][j]
                  
                  regions = measure.regionprops(measure.label(im_star_white))
                  bubble = regions[0]
                  
                  # initial guess (must be to change related on the % considered)
                  y_i, x_i = bubble.centroid
                  a_i = bubble.major_axis_length / 2.
                  b_i = bubble.minor_axis_length / 2.
                  theta_i  = pi/4
                  t = np.linspace(0, 2*pi, nSubDim)
                  
                  def cost(params):
                      x0, y0, a, b, theta = params
                      #coords = draw.disk((y0, x0), r, shape=image.shape)
                      coords = draw.ellipse(y0, x0, a, b, shape=None, rotation= theta)
                      template_star = np.zeros_like(im_star_white)
                      template_star[coords] = 1
                      return -np.sum(template_star == im_star_white)
                    
                  x_sf, y_sf, a_sf, b_sf, theta_sf = opt.fmin(cost, (x_i, y_i, a_i, b_i, theta_i))
                  
                  #def ellips(t, x_f, y_f, a_f, bb_f, theta_f):
                  theta_sf = np.pi/2 -theta_sf
                  par_star_arr[i][j] = [x_sf, y_sf, a_sf, b_sf, theta_sf]
                  
                  theta_sf_deg = theta_sf*180/pi
                  Ell_star = np.array([a_sf*np.cos(t) , b_sf*np.sin(t)])  
                         #u,v removed to keep the same center location
                  M_rot_star = np.array([[cos(theta_sf) , -sin(theta_sf)],[sin(theta_sf) , cos(theta_sf)]]) 
                  
                         #2-D rotation matrix
                    
                  Ell_rot_star_ = np.zeros((2, nSubDim))
                  Ell_rot_star = np.zeros((2, nSubDim))
                  for k in range(Ell_star.shape[1]):
                      Ell_rot_star_[:,k] = np.dot(M_rot_star,Ell_star[:,k]) # fait le produit scal de la matrice de rotation par chaq couple parametriq 
                      Ell_rot_star[:,k] = Ell_rot_star_[:,k]
                      Ell_rot_star[0,k] = Ell_rot_star[0,k] + x_sf
                      Ell_rot_star[1,k] = Ell_rot_star[1,k] + y_sf
                      #return Ell_rot.ravel() # .ravel permet de passer de deux dimension à une seule
                      Ell_rot_star_arr[i][j][:,k] = Ell_rot_star[:,k] 
                      
                      
                      
                      
                      
        #opening psf file
        for i in range(nFrames_psf):
            hdu = fits.open(file_lst[i])   
            data = hdu[0].data
            intensity = data[1,:,:]
                      
            cutout = Cutout2D(intensity, position = position, size = size)
            zoom_hdu = hdu.copy()
            sub_v_psf = cutout.data
                      
            sub_v_psf_arr[i] = sub_v_psf
                      
            for j in range(n_threshold):
                Ellips_psf_arr2[j] = sub_v_psf
                Ellips_psf_im = np.zeros_like(sub_v_psf_arr[i]) #creation d'un tableau de meme forme que sub_v
                Ellips_psf_im[sub_v_psf > lst_threshold[j]*np.max(sub_v_psf)] = sub_v_psf[sub_v_psf > lst_threshold[j]*np.max(sub_v_psf)]# on retient les points d'intensité égale à 5% de Imax 
                Ellips_psf_im_arr[i][j] = Ellips_psf_im
                
                Ellips_psf = np.zeros_like(sub_v_psf)          #creation d'un tableau de meme forme que sub_v
                Ellips_psf[sub_v_psf > lst_threshold[j]*np.max(sub_v_psf)] = 1   # on retient les points d'intensité 
                Ellips_psf_arr[i][j] = Ellips_psf                # égale à 5% de Imax et à tous ces points 
                                                                 # on donne 1 comme valeur d'intensité
                    
                            
                Vmin_psf[i][j] = np.min(np.log10(sub_v_psf+np.abs(np.min(sub_v_psf))+10))
                Vmax_psf[i][j] = np.max(np.log10(sub_v_psf+np.abs(np.min(sub_v_psf))+10))
                          
                Vmin_psf_r[i][j] = np.min(np.log10(Ellips_psf_im+np.abs(np.min(Ellips_psf_im))+10))
                Vmax_psf_r[i][j] = np.max(np.log10(Ellips_psf_im+np.abs(np.min(Ellips_psf_im))+10)) 
                          
                Vmin_psf_w[i][j] = np.min(np.log10(Ellips_psf+np.abs(np.min(Ellips_psf))+10))
                Vmax_psf_w[i][j] = np.max(np.log10(Ellips_psf+np.abs(np.min(Ellips_psf))+10))  
                     
                      
                im_psf_white = Ellips_psf_arr[i][j]
                im_psf_real = Ellips_psf_im_arr[i][j]
                          
                regions = measure.regionprops(measure.label(im_psf_white))
                bubble = regions[0]
                
                          # initial guess (must be to change related on the % considered)
                y_i, x_i = bubble.centroid
                a_i = bubble.major_axis_length / 2.
                b_i = bubble.minor_axis_length / 2.
                theta_i  = pi/4
                t = np.linspace(0, 2*pi, nSubDim)
                          
                def cost(params):
                    
                    x0, y0, a, b, theta = params
                    #coords = draw.disk((y0, x0), r, shape=image.shape)
                    coords = draw.ellipse(y0, x0, a, b, shape=None, rotation= theta)
                    template_psf = np.zeros_like(im_psf_white)
                    template_psf[coords] = 1
                    return -np.sum(template_psf == im_psf_white)
                            
                x_pf, y_pf, a_pf, b_pf, theta_pf = opt.fmin(cost, (x_i, y_i, a_i, b_i, theta_i))
                          
                #def ellips(t, x_f, y_f, a_f, bb_f, theta_f):
                theta_pf = np.pi/2 -theta_pf
                par_psf_arr[i][j] = [x_pf, y_pf, a_pf, b_pf, theta_pf]
                          
                theta_pf_deg = theta_pf*180/pi
                Ell_psf = np.array([a_pf*np.cos(t) , b_pf*np.sin(t)])  
                                 #u,v removed to keep the same center location
                M_rot_psf = np.array([[cos(theta_pf) , -sin(theta_pf)],[sin(theta_pf) , cos(theta_pf)]]) 
                          
                                 #2-D rotation matrix
                            
                Ell_rot_psf_ = np.zeros((2, nSubDim))
                Ell_rot_psf = np.zeros((2, nSubDim))
                for k in range(Ell_psf.shape[1]):
                    Ell_rot_psf_[:,k] = np.dot(M_rot_psf,Ell_psf[:,k]) # fait le produit scal de la matrice de rotation par chaq couple parametriq 
                    Ell_rot_psf[:,k] = Ell_rot_psf_[:,k]
                    Ell_rot_psf[0,k] = Ell_rot_psf[0,k] + x_pf
                    Ell_rot_psf[1,k] = Ell_rot_psf[1,k] + y_pf
                    #return Ell_rot.ravel() # .ravel permet de passer de deux dimension à une seule
                    Ell_rot_psf_arr[i][j][:,k] = Ell_rot_psf[:,k] 
                              
                              
              # plots   
                          
                         
            # plt.figure('full image and all the  contours' + f'{strs[j]}')
            # plt.clf()
            # plt.imshow(np.log10(sub_v_star_arr[i]+np.abs(np.min(sub_v_star_arr[i]))+10), cmap ='inferno', vmin=Vmin_star_r[i][j], vmax=Vmax_star_r[i][j], origin='lower')
            # #plt.plot( u + Ell_rot[0,:] , v + Ell_rot[1,:],'darkorange' ) # rotated ellipse
                         
            # plt.plot(Ell_rot_star_arr[i][0][0,:], Ell_rot_star_arr[i][0][1,:])
                         
                         
            # plt.show()
            # #plt.title('full image and all the  contours at ' + ' for ' + f'{lst_Frame_name[i]}'+' of '+ f'{star_name}', fontsize=10)
            # plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/'+ star_name +'/plots/fits/log_scale/fully_automatic/' +'full_image_and_all_the_contours' +'_for_' + f'{lst_Frame_name[i]}' + '.pdf', 
            #                          dpi=100, bbox_inches ='tight')
                       
                       
            # plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/'+ star_name +'/plots/fits/log_scale/fully_automatic/' +'full_image_and_all_the_contours'  + '_for_' + f'{lst_Frame_name[i]}' + '.png', 
            #                  dpi=100, bbox_inches ='tight')
            # plt.tight_layout()
            
            
        # ========================================================#
        # For the radial profile at a given orientation, theta_f  #     
        # ========================================================# 
        
        
        
    # star
    im_s = np.log10(sub_v_star_arr[0] + np.abs(np.min(sub_v_star_arr[0])) + 10) # intensity for the radial profile 
    imp_s = np.log10(sub_v_star_arr[1] + np.abs(np.min(sub_v_star_arr[1])) + 10) # polarized  intensity for the radial profile 
    x0_s, y0_s, x1_s, y1_s, x2_s, y2_s,z_s, zi1_s, zi2_s, xx1_s, yy1_s, xx2_s, yy2_s, zzi1_s, zzi2_s = erp(par_star_arr[0][0][0], par_star_arr[0][0][1], par_star_arr[0][0][2], par_star_arr[0][0][3], par_star_arr[0][0][4], im_s, 100)
    x0p_s, y0p_s, x1p_s, y1p_s, x2p_s, y2p_s,zp_s, zi1p_s, zi2p_s, xx1p_s, yy1p_s, xx2p_s, yy2p_s, zzi1p_s, zzi2p_s = erp(par_star_arr[1][0][0], par_star_arr[1][0][1], par_star_arr[1][0][2], par_star_arr[1][0][3], par_star_arr[1][0][4], imp_s, 100)
    
    #psf
    
    im_p = np.log10(sub_v_psf_arr[0] + np.abs(np.min(sub_v_psf_arr[0])) + 10) # intensity for the radial profile 
    imp_p = np.log10(sub_v_psf_arr[1] + np.abs(np.min(sub_v_psf_arr[1])) + 10) # polarized  intensity for the radial profile 
    x0_p, y0_p, x1_p, y1_p, x2_p, y2_p,z_p, zi1_p, zi2_p, xx1_p, yy1_p, xx2_p, yy2_p, zzi1_p, zzi2_p = erp(par_psf_arr[0][0][0], par_psf_arr[0][0][1], par_psf_arr[0][0][2], par_psf_arr[0][0][3], par_psf_arr[0][0][4], im_p, 100)
    x0p_p, y0p_p, x1p_p, y1p_p, x2p_p, y2p_p,zp_p, zi1p_p, zi2p_p, xx1p_p, yy1p_p, xx2p_p, yy2p_p, zzi1p_p, zzi2p_p = erp(par_psf_arr[1][0][0], par_psf_arr[1][0][1], par_psf_arr[1][0][2], par_psf_arr[1][0][3], par_psf_arr[1][0][4], imp_p, 100)
    
    
    #les plots
    plt.figure('profiles comparison')
    plt.clf()
    plt.subplot(3,2,1) # De l'intensité
    plt.imshow(z_s, cmap ='inferno', vmin = Vmin_star_r[0][0], vmax = Vmax_star_r[0][0], origin='lower') # full image of intensity
    plt.plot(Ell_rot_star_arr[0][0][0,:] , Ell_rot_star_arr[0][0][1,:]) # ellipse de l'intensité à 10% de l'intensité max
    plt.plot([x0_s, x1_s], [y0_s, y1_s], 'ro-')  # tracé du demi-grand axe de l'ellipse
    #plt.plot([x0, x1_], [y0, y1_], 'ro-')# tracé de l'autre demi-grand axe de l'ellipse
    plt.plot([x0_s, x2_s], [y0_s, y2_s], 'ro-') # tracé du demi petit axe de l'ellipse
    #plt.plot([x0, x2_], [y0, y2_], 'ro-') # tracé de l'autre demi petit axe de l'ellipse
    plt.xlabel('Relative R.A.(pix)', size=10)
    plt.ylabel('Relative Dec.(pix)', size=10)
    plt.title('radial profile of  ('+star_name+ ')'  f'{lst_Frame_name[0]}'  ' at theta', fontsize=10)
    
    plt.subplot(3,2,2) # De l'intensité polarisée  
    plt.imshow(zp_s, cmap ='inferno', vmin = Vmin_star_r[1][0], vmax =  Vmin_star_r[1][0], origin='lower')# # full image of polarised intensity
    plt.plot(Ell_rot_star_arr[1][0][0,:] , Ell_rot_star_arr[1][0][1,:])  # ellipse de l'intensitépolarisé à 10% de l'intensité max
    plt.plot([x0p_s, x1p_s], [y0p_s, y1p_s], 'ro-')  # tracé du demi-grand axe de l'ellipse
    #plt.plot([x0p, x1p_], [y0p, y1p_], 'ro-')-')# tracé de l'autre demi-grand axe de l'ellipse
    plt.plot([x0p_s, x2p_s], [y0p_s, y2p_s], 'ro-')   # tracé du demi petit axe de l'ellipse
    #plt.plot([x0p, x2p_], [y0p, y2p_], 'ro-') # tracé de l'autre demi petit axe de l'ellipse
    plt.xlabel('Relative R.A.(pix)', size=10)
    plt.ylabel('Relative Dec.(pix)', size=10)
    plt.title('radial profile  (' + star_name + ')'  f'{lst_Frame_name[1]}' ' at theta + pi/2' ,fontsize=10)
    
    plt.subplot(3,2,3)
    plt.imshow(z_p, cmap ='inferno', vmin = Vmin_psf_r[0][0], vmax = Vmin_psf_r[0][0], origin='lower') # full image of intensity
    plt.plot(Ell_rot_psf_arr[0][0][0,:] , Ell_rot_psf_arr[0][0][1,:]) # ellipse de l'intensité à 10% de l'intensité max
    plt.plot([x0_p, x1_p], [y0_p, y1_p], 'ro-')  # tracé du demi-grand axe de l'ellipse
    #plt.plot([x0, x1_], [y0, y1_], 'ro-')# tracé de l'autre demi-grand axe de l'ellipse
    plt.plot([x0_p, x2_p], [y0_p, y2_p], 'ro-') # tracé du demi petit axe de l'ellipse
    #plt.plot([x0, x2_], [y0, y2_], 'ro-') # tracé de l'autre demi petit axe de l'ellipse
    plt.xlabel('Relative R.A.(pix)', size=10)
    plt.ylabel('Relative Dec.(pix)', size=10)
    plt.title('radial profile of  ('+star_name+ ')'  f'{lst_Frame_name[0]}'  ' at theta', fontsize=10)
    
    plt.subplot(3,2,4)
    plt.imshow(zp_p, cmap ='inferno', vmin = Vmin_psf_r[1][0], vmax =  Vmin_psf_r[1][0], origin='lower')# # full image of polarised intensity
    plt.plot(Ell_rot_psf_arr[1][0][0,:] , Ell_rot_psf_arr[1][0][1,:])  # ellipse de l'intensitépolarisé à 10% de l'intensité max
    plt.plot([x0p_p, x1p_p], [y0p_p, y1p_p], 'ro-')  # tracé du demi-grand axe de l'ellipse
    #plt.plot([x0p, x1p_], [y0p, y1p_], 'ro-')-')# tracé de l'autre demi-grand axe de l'ellipse
    plt.plot([x0p_p, x2p_p], [y0p_p, y2p_p], 'ro-')   # tracé du demi petit axe de l'ellipse
    #plt.plot([x0p, x2p_], [y0p, y2p_], 'ro-') # tracé de l'autre demi petit axe de l'ellipse
    plt.xlabel('Relative R.A.(pix)', size=10)
    plt.ylabel('Relative Dec.(pix)', size=10)
    plt.title('radial profile  (' + star_name + ')'  f'{lst_Frame_name[1]}' ' at theta + pi/2', fontsize=10)
    
    plt.subplot(3,2,5)
    plt.plot(zi1_s) # profile radiale de l'intensité suivant le demi grand axe (star)
    #plt.plot(zi1_p) # profile radiale de l'intensité suivant le demi petit axe (psf)
    #plt.plot(zi1p) # profile radiale de l'intensité polarisée suivant le grand axe
    plt.legend(["star", "psf"])
    plt.title('radial profile of intensity demi grand axe')
    plt.xlabel('r (mas)', size=10)          
    plt.ylabel(r'Intensity in log$_{10}$ scale', size=10)
    plt.subplot(3,2,6)
    #plt.plot(zi1p_s) # profile radiale de l'intensité polarisée suivant le demi petit axe(star)
    #plt.plot(zi2) # profile radiale de l'intensité suivant le demi petit axe
    plt.plot(zi1p_p) # profile radiale de l'intensité suivant le demi petit axe
    plt.legend(["star", "psf"])
    plt.title('radial profile of polarised intensity demi petit axe')
    plt.xlabel('r (pix)', size=10) 
    plt.ylabel(r'Intensity in log$_{10}$ scale', size=10)
    plt.show()
    plt.show()
    plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/'+star_name+'/plots/radial/radial_profile_at_a_given_orientation.pdf', 
                    dpi=100, bbox_inches ='tight')
    plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/'+star_name+'/plots/radial/radial_profile_at_a_given_orientation.png', 
            dpi=100, bbox_inches ='tight')
    plt.tight_layout()
        

star_name  = 'SW_Col'
obsmod = 'both'
log_image(star_name, obsmod)
