#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Nov 18 15:06:04 2022

@author: nbadolo
"""
import numpy as np
from numpy import nan
import os
from astropy.io import fits
import math
from math import pi, cos, sin, atan
from scipy import optimize, signal
from scipy.signal import convolve2d as conv2
from AymardPack import Margaux_RL_deconv
from AymardPack import EllRadialProf as erp
from AymardPack import DelHotPix
from skimage import color, data, restoration
from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import scipy.optimize as opt
from skimage import io, color, measure, draw, img_as_bool
import matplotlib.colors as colors
from matplotlib.pyplot import Figure, subplot
#%% 
##parameters
nDim=1024
nSubDim = 100 # plage de pixels que l'on veut afficher
size = (nSubDim, nSubDim)
nDimfigj=[9,10,11]
nDimfigk=[0,1,2]
lst_threshold = [0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]
strs = [str(x*100) + ' %' for x in lst_threshold]
n_threshold = len(lst_threshold)
pix2mas = 3.4  #en mas/pix
x_min = -pix2mas*nSubDim//2
x_max = pix2mas*(nSubDim//2-1)
y_min = -pix2mas*nSubDim//2
y_max = pix2mas*(nSubDim//2-1)
position = (nDim//2,nDim//2)
size = (nSubDim, nSubDim)


txt_folder = 'sphere_txt_file'
file_path = '/home/nbadolo/Bureau/Aymard/Donnees_sph/' + txt_folder + '/'
file_name = 'no_common_data_lst.txt'
no_common_data_lst = open("{}/{}".format(file_path, file_name), "w")
no_common_data_lst.write("{}\n".format('Star name', 'Mode'))
#%%
def log_image(star_name, obsmod):   
        
    fdir= '/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+ '/'
    fdir_star = fdir + 'star/'+obsmod+ '/' 
    fdir_psf = fdir +'psf/'+obsmod+ '/'
    
    #star opening
    lst_fltr_star = os.listdir(fdir_star)
    n_lst_fltr_star = len(lst_fltr_star)
    lst_fltr2_star = []
    nDimfigj = [3, 4, 5]
    nDimfigk = [6,7, 8]
    for p in range(n_lst_fltr_star):
        
        fdir_fltr_data_star = fdir_star + lst_fltr_star[p]
        lst_fltr_data_star = os.listdir(fdir_fltr_data_star)
        n_lst_fltr_data_star = len(lst_fltr_data_star)
        if n_lst_fltr_data_star != 0:
            lst_fltr2_star.append(lst_fltr_star[p])
    #print(lst_fltr2_star)
    
    #psf opening
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
    
    #elements commons both to star and psf
    lst_fltr3 = list(set(lst_fltr2_star).intersection(lst_fltr2_psf))
    n_lst_fltr3 = len(lst_fltr3)
    
    # begining of elements 
    if n_lst_fltr3 == 0:
           print( f'No common data for {star_name} and his psf')
           no_common_data_lst.write("{}\n".format(f'{star_name}', obsmod))
           return(f'{star_name}', obsmod) # le recupere le nom de l'étoile si pas de psf
           pass
    else :
        
        for l in range(n_lst_fltr3):
            fdir_star_fltr = fdir_star + lst_fltr3[l] +'/'
            fdir_psf_fltr = fdir_psf + lst_fltr3[l] + '/'
            
            fname1 ='zpl_p23_make_polar_maps-ZPL_SCIENCE_P23_REDUCED'
            fname2 ='-zpl_science_p23_REDUCED'
            file_I_star = fdir_star_fltr + fname1 + '_I'+fname2+'_I.fits'
            file_PI_star = fdir_star_fltr +fname1 + '_PI'+fname2+'_PI.fits'
            file_DOLP_star = fdir_star_fltr +fname1 + '_DOLP'+fname2+'_DOLP.fits'
            #file_AOLP_star= fdir_star_fltr + fname1+'_AOLP'+fname2+'_AOLP.fits'
        
            file_I_psf = fdir_psf_fltr + fname1 + '_I' + fname2+'_I.fits'
            file_PI_psf = fdir_psf_fltr +fname1 + '_PI' + fname2+'_PI.fits'
            file_DOLP_psf = fdir_psf_fltr + fname1 + '_DOLP' + fname2 +'_DOLP.fits'
            #file_AOLP_psf = fdir_psf_fltr +fname1+'_AOLP'+fname2+'_AOLP.fits'
          
            file_lst = [file_I_star,file_PI_star,
                      file_I_psf,file_PI_psf]
            
            file_star_lst= [file_I_star, file_PI_star]
            file_psf_lst = [file_I_psf, file_PI_psf]
            label = ['Int.', 'Pol. Int']          
            
            #nFrames = len(file_star_lst)
            nFrames_star = len(file_star_lst)
            nFrames_psf = len(file_psf_lst)
            # nFrames_d = nFrames_star
            # nFrames = nFrames_star
           
            lst_Frame_name = ['Intensity', 'Pol_Intensity']
            
            
            # lists and arrays of the star
            t = np.linspace(0, 2*pi, nSubDim)
            x, y = np.meshgrid(np.arange(nSubDim),np.arange(nSubDim)) #cree un tableau 
            sub_v_star_arr = np.zeros((nFrames_star, nSubDim, nSubDim))
            Ellips_star_w_arr = np.zeros((nFrames_star, n_threshold,nSubDim,nSubDim))
            Ellips_star_im_arr = np.zeros((nFrames_star, n_threshold,nSubDim,nSubDim))
            Ellips_star_arr2 = np.zeros((n_threshold, nSubDim, nSubDim))
            Ellips_star_ = np.zeros((n_threshold, nSubDim))
            Ell_rot_star_arr = np.zeros((nFrames_star, n_threshold, 2, nSubDim))
            ind_star_arr  = np.zeros((nFrames_star,2)) # pour les indices des pixels les plus brillants(photcentres)
            par_star_arr = np.zeros((nFrames_star, n_threshold, 5)) # les paramètres de l'étoile
            Vmin_star = np.zeros((nFrames_star, n_threshold)) # pour l'image totale
            Vmax_star = np.zeros((nFrames_star, n_threshold))
            Vmin_star_r = np.zeros((nFrames_star, n_threshold))
            Vmax_star_r = np.zeros((nFrames_star, n_threshold))# pour l'image reelle tronquee à au pourcentage
            Vmin_star_w = np.zeros((nFrames_star, n_threshold))
            Vmax_star_w = np.zeros((nFrames_star, n_threshold)) # pour l'image binaire
            
            
            # lists and arrays of the psf   
            x, y = np.meshgrid(np.arange(nSubDim),np.arange(nSubDim)) #cree un tableau 
            sub_v_psf_arr = np.zeros((nFrames_psf, nSubDim, nSubDim))
            Ellips_psf_arr = np.zeros((nFrames_psf, n_threshold,nSubDim,nSubDim))
            Ellips_psf_im_arr = np.zeros((nFrames_psf, n_threshold,nSubDim,nSubDim))
            Ellips_psf_arr2 = np.zeros((n_threshold, nSubDim, nSubDim))
            Ellips_psf_ = np.zeros((n_threshold, nSubDim))
            Ell_rot_psf_arr = np.zeros((nFrames_psf, n_threshold, 2, nSubDim))
            par_psf_arr = np.zeros((nFrames_psf, n_threshold, 5)) # les paramètres de l'étoile
            Vmin_psf = np.zeros((nFrames_psf, n_threshold)) # pour l'image totale
            Vmax_psf = np.zeros((nFrames_psf, n_threshold))
            Vmin_psf_r = np.zeros((nFrames_psf, n_threshold))
            ind_psf_arr  = np.zeros((nFrames_psf,2)) # pour les indices des pixels les plus brillants(photcentres)
            Vmax_psf_r = np.zeros((nFrames_psf, n_threshold))# pour l'image reelle tronquee à au pourcentage
            Vmin_psf_w = np.zeros((nFrames_psf, n_threshold))
            Vmax_psf_w = np.zeros((nFrames_psf, n_threshold)) # pour l'image binaire
            #Ell_rot_star_arr = np.zeros()
            
            strs = [str(x*100) + ' %' for x in lst_threshold]
                    
            #opening star file
            
            for str_i in range(nFrames_star):
                
                  hdu_str = DelHotPix(file_star_lst[str_i]) # suppression des pixels chauds de l'image de l'étoile.
                  #hdu_str = fits.open(file_star_lst[str_i])   
                  data_str = hdu_str[0].data
                  intensity_str = data_str[1,:,:]
                  
                  cutout_str = Cutout2D(intensity_str, position = position, size = size)
                  #zoom_hdu = hdu.copy()
                  sub_v_star = cutout_str.data  # image rognée de l'étoile
                  
                  sub_v_star_arr[str_i] = sub_v_star  # affectation des differentes images rognée dans le tableau
                  
                  # =========================================================#
                  #  Pour le calcul du decalage des centres de l'étoile      #
                  # =========================================================#
                                    
                  #  Determination des coordonnées du pixel le plus brillant
                    
                  a_s = sub_v_star_arr[str_i]
                  ind_star = np.unravel_index(np.argmax(a_s, axis = None), a_s.shape)
                  ind_star_arr[str_i] = ind_star
                  print(ind_star)
                  
                  for str_j in range(n_threshold):
                      Ellips_star_arr2[str_j] = sub_v_star
                      Ellips_star_im = np.zeros_like(sub_v_star_arr[str_i]) #creation d'un tableau de meme forme que sub_v
                      Imean_s = np.mean(sub_v_star)
                      Imax_s = np.max(sub_v_star)
                      if Imax_s > Imean_s :
                          Ellips_star_im[sub_v_star > lst_threshold[str_j]*Imax_s] = sub_v_star[sub_v_star > lst_threshold[str_j]*Imax_s]# on retient les points d'intensité égale à threshold % de Imax 
                      Ellips_star_im_arr[str_i][str_j] = Ellips_star_im
                      
                      Ellips_star = np.zeros_like(sub_v_star)          #creation d'un tableau de meme forme que sub_v
                      Ellips_star[sub_v_star > lst_threshold[str_j]*np.max(sub_v_star)] = 1   # on retient les points d'intensité 
                      Ellips_star_w_arr[str_i][str_j] = Ellips_star                # égale à 5% de Imax et à tous ces points 
                                                             # on donne 1 comme valeur d'intensité
                                   
                      Vmin_star[str_i][str_j] = np.min(np.log10(sub_v_star+np.abs(np.min(sub_v_star))+10))
                      Vmax_star[str_i][str_j] = np.max(np.log10(sub_v_star+np.abs(np.min(sub_v_star))+10))
                      
                      Vmin_star_r[str_i][str_j] = np.min(np.log10(Ellips_star_im+np.abs(np.min(Ellips_star_im))+10))
                      Vmax_star_r[str_i][str_j] = np.max(np.log10(Ellips_star_im+np.abs(np.min(Ellips_star_im))+10)) 
                      
                      Vmin_star_w[str_i][str_j] = np.min(np.log10(Ellips_star+np.abs(np.min(Ellips_star))+10))
                      Vmax_star_w[str_i][str_j] = np.max(np.log10(Ellips_star+np.abs(np.min(Ellips_star))+10))  
                 
                  
                      im_star_white = Ellips_star_w_arr[str_i][str_j]
                      im_star_real = Ellips_star_im_arr[str_i][str_j]
                      
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
                          template_star = np.zeros_like(im_star_white)
                          template_star[coords_s] = 1
                          return -np.sum(template_star == im_star_white)
                      x_sf, y_sf, a_sf, b_sf, theta_sf = opt.fmin(cost, (xs_i, ys_i, as_i, bs_i, thetas_i))
                      
                      #def ellips(t, x_f, y_f, a_f, bb_f, theta_f):
                      theta_sf = np.pi/2 -theta_sf
                      par_star_arr[str_i][str_j] = [x_sf, y_sf, a_sf, b_sf, theta_sf]
                      
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
                          Ell_rot_star_arr[str_i][str_j][:,str_k] = Ell_rot_star[:,str_k] 
                          
                                               
            #opening psf file
            for psf_i in range(nFrames_psf):
                hdu_psf = DelHotPix(file_psf_lst[psf_i]) # suppression des pixels chauds de l'image de l'étoile
                #hdu_psf = fits.open(file_psf_lst[psf_i])   
                data_psf = hdu_psf[0].data
                intensity_psf = data_psf[0,:,:]
                          
                cutout_psf = Cutout2D(intensity_psf, position = position, size = size)
                #zoom_hdu = hdu.copy()
                sub_v_psf = cutout_psf.data
                         
                sub_v_psf_arr[psf_i] = sub_v_psf
                          
                for psf_j in range(n_threshold):
                    Ellips_psf_arr2[psf_j] = sub_v_psf
                    Ellips_psf_im = np.zeros_like(sub_v_psf_arr[psf_i]) #creation d'un tableau de meme forme que sub_v
                    Imean_p = np.mean(sub_v_psf)
                    Imax_p = np.max(sub_v_psf)
                    if Imax_p > Imean_p : # on contraint Imax à etre raisonable. pour eliminer d'éventuels mauvais pixels 
                        Ellips_psf_im[sub_v_psf > lst_threshold[psf_j]*Imax_p] = sub_v_psf[sub_v_psf > lst_threshold[psf_j]*Imax_p]# on retient les points d'intensité égale à threshold de Imax 
                        Ellips_psf_im_arr[psf_i][psf_j] = Ellips_psf_im
                    
                    Ellips_psf = np.zeros_like(sub_v_psf)          #creation d'un tableau de meme forme que sub_v
                    Ellips_psf[sub_v_psf > lst_threshold[psf_j]*np.max(sub_v_psf)] = 1   # on retient les points d'intensité 
                    Ellips_psf_arr[psf_i][psf_j] = Ellips_psf                # égale à threshold % de Imax et à tous ces points 
                                                                     # on donne 1 comme valeur d'intensité
                        
                                
                    Vmin_psf[psf_i][psf_j] = np.min(np.log10(sub_v_psf + np.abs(np.min(sub_v_psf))+10))
                    Vmax_psf[psf_i][psf_j] = np.max(np.log10(sub_v_psf + np.abs(np.min(sub_v_psf))+10))
                              
                    Vmin_psf_r[psf_i][psf_j] = np.min(np.log10(Ellips_psf_im + np.abs(np.min(Ellips_psf_im))+10))
                    Vmax_psf_r[psf_i][psf_j] = np.max(np.log10(Ellips_psf_im + np.abs(np.min(Ellips_psf_im))+10)) 
                              
                    Vmin_psf_w[psf_i][psf_j] = np.min(np.log10(Ellips_psf + np.abs(np.min(Ellips_psf))+10))
                    Vmax_psf_w[psf_i][psf_j] = np.max(np.log10(Ellips_psf + np.abs(np.min(Ellips_psf))+10))  
                         
                          
                    #im_psf_white = Ellips_psf_arr[psf_i][psf_j]
                    im_psf_white = Ellips_psf
                    #im_psf_real = Ellips_psf_im_arr[psf_i][psf_j]
                    im_psf_real = Ellips_psf_im
                              
                    regions_psf = measure.regionprops(measure.label(im_psf_white))
                    bubble_psf = regions_psf[0]
                    
                              # initial guess (must be to change related on the % considered)
                    yp_i, xp_i = bubble_psf.centroid
                    ap_i = bubble_psf.major_axis_length / 2.
                    bp_i = bubble_psf.minor_axis_length / 2.
                    thetap_i  = pi/4
                    t = np.linspace(0, 2*pi, nSubDim)
                              
                    def cost(params_p):

                        
                        x0p, y0p, ap, bp, thetap = params_p
                        #coords = draw.disk((y0, x0), r, shape=image.shape)
                        coords_p = draw.ellipse(y0p, x0p, ap, bp, shape = None, rotation= thetap)
                        template_psf = np.zeros_like(im_psf_white)
                        template_psf[coords_p] = 1
                        return -np.sum(template_psf == im_psf_white)
                                
                    x_pf, y_pf, a_pf, b_pf, theta_pf = opt.fmin(cost, (xp_i, yp_i, ap_i, bp_i, thetap_i))
                              
                    #def ellips(t, x_f, y_f, a_f, bb_f, theta_f):
                    theta_pf = np.pi/2 -theta_pf
                    par_psf_arr[psf_i][psf_j] = [x_pf, y_pf, a_pf, b_pf, theta_pf]
                              
                    theta_pf_deg = theta_pf*180/pi
                    Ell_psf = np.array([a_pf*np.cos(t) , b_pf*np.sin(t)])  
                                     #u,v removed to keep the same center location
                    M_rot_psf = np.array([[cos(theta_pf) , -sin(theta_pf)],[sin(theta_pf) , cos(theta_pf)]]) 
                              
                                     #2-D rotation matrix
                                
                    Ell_rot_psf_ = np.zeros((2, nSubDim))
                    Ell_rot_psf = np.zeros((2, nSubDim))
                    for psf_k in range(Ell_psf.shape[1]):
                        Ell_rot_psf_[:,psf_k] = np.dot(M_rot_psf,Ell_psf[:,psf_k]) # fait le produit scal de la matrice de rotation par chaq couple parametriq 
                        Ell_rot_psf[:,psf_k] = Ell_rot_psf_[:,psf_k]
                        Ell_rot_psf[0,psf_k] = Ell_rot_psf[0,psf_k] + x_pf
                        Ell_rot_psf[1,psf_k] = Ell_rot_psf[1,psf_k] + y_pf
                        #return Ell_rot.ravel() # .ravel permet de passer de deux dimension à une seule
                        Ell_rot_psf_arr[psf_i][psf_j][:,psf_k] = Ell_rot_psf[:,psf_k] 
                                  

                

                                    # ========================================================#
                                    # For the radial profile at a given orientation, theta_f  #     
                                    # ========================================================# 
            
            
            
          ## star
        im_s  = np.log10(sub_v_star_arr[0] + np.abs(np.min(sub_v_star_arr[0])) + 10) # intensity for the radial profile 
        imp_s = np.log10(sub_v_star_arr[1] + np.abs(np.min(sub_v_star_arr[1])) + 10) # polarized  intensity for the radial profile 
        x0_s, y0_s, x1_s, y1_s, x2_s, y2_s,z_s, zi1_s, zi2_s, xx1_s, yy1_s, xx2_s, yy2_s, zzi1_s, zzi2_s = erp(par_star_arr[0][0][0], par_star_arr[0][0][1], par_star_arr[0][0][2], par_star_arr[0][0][3], par_star_arr[0][0][4], im_s, 100)
        x0p_s, y0p_s, x1p_s, y1p_s, x2p_s, y2p_s, zp_s, zi1p_s, zi2p_s, xx1p_s, yy1p_s, xx2p_s, yy2p_s, zzi1p_s, zzi2p_s = erp(par_star_arr[1][0][0], par_star_arr[1][0][1], par_star_arr[1][0][2], par_star_arr[1][0][3], par_star_arr[1][0][4], imp_s, 100)
        
          ##psf    
        im_p  = np.log10(sub_v_psf_arr[0] + np.abs(np.min(sub_v_psf_arr[0])) + 10) # intensity for the radial profile 
        imp_p = np.log10(sub_v_psf_arr[1] + np.abs(np.min(sub_v_psf_arr[1])) + 10) # polarized  intensity for the radial profile 
        x0_p, y0_p, x1_p, y1_p, x2_p, y2_p,z_p, zi1_p, zi2_p, xx1_p, yy1_p, xx2_p, yy2_p, zzi1_p, zzi2_p = erp(par_psf_arr[0][0][0], par_psf_arr[0][0][1], par_psf_arr[0][0][2], par_psf_arr[0][0][3], par_psf_arr[0][0][4], im_p, 100)
        x0p_p, y0p_p, x1p_p, y1p_p, x2p_p, y2p_p, zp_p, zi1p_p, zi2p_p, xx1p_p, yy1p_p, xx2p_p, yy2p_p, zzi1p_p, zzi2p_p = erp(par_psf_arr[1][0][0], par_psf_arr[1][0][1], par_psf_arr[1][0][2], par_psf_arr[1][0][3], par_psf_arr[1][0][4], imp_p, 100)
        
        
        ##les plots
        fig = plt.figure('profiles comparison')
        plt.clf()
        #plt.subplot(3,2,1) # De l'intensité
        
        # plt.imshow(z_s, cmap ='inferno', vmin = Vmin_star_r[0][0], vmax = Vmax_star_r[0][0], origin='lower') # full image of intensity
        # plt.plot(Ell_rot_star_arr[0][0][0,:] , Ell_rot_star_arr[0][0][1,:]) # ellipse de l'intensité à 10% de l'intensité max
        # plt.plot([x0_s, x1_s], [y0_s, y1_s], 'ro-')  # tracé du demi-grand axe de l'ellipse
        # #plt.plot([x0, x1_], [y0, y1_], 'ro-')# tracé de l'autre demi-grand axe de l'ellipse
        # plt.plot([x0_s, x2_s], [y0_s, y2_s], 'ro-') # tracé du demi petit axe de l'ellipse
        # #plt.plot([x0, x2_], [y0, y2_], 'ro-') # tracé de l'autre demi petit axe de l'ellipse
        # plt.xlabel('Relative R.A.(pix)', size=10)
        # plt.ylabel('Relative Dec.(pix)', size=10)
        # plt.title('radial profile of  ('+star_name+ ')'  f'{lst_Frame_name[0]}'  ' at theta', fontsize=10)
        # vmin = Vmin_star_r[1][0] vmax =  Vmin_star_r[1][0],
       
        # plt.subplot(1,2,1) # De l'intensité polarisée  
        # plt.imshow(zp_s, cmap ='inferno', vmin = np.min(zp_s), vmax = np.max(zp_s),  origin='lower')# # full image of polarised intensity
        # plt.plot(Ell_rot_star_arr[1][0][0,:] , Ell_rot_star_arr[1][0][1,:])  # ellipse de l'intensitépolarisé à 10% de l'intensité max
        # plt.plot([x0p_s, x1p_s], [y0p_s, y1p_s], 'ro-')  # tracé du demi-grand axe de l'ellipse
        # #plt.plot([x0p, x1p_], [y0p, y1p_], 'ro-')-')# tracé de l'autre demi-grand axe de l'ellipse
        # plt.plot([x0p_s, x2p_s], [y0p_s, y2p_s], 'ro-')   # tracé du demi petit axe de l'ellipse
        # #plt.plot([x0p, x2p_], [y0p, y2p_], 'ro-') # tracé de l'autre demi petit axe de l'ellipse
        # plt.colorbar(label='ADU', location ='top', shrink = 0.6)
        # plt.xlabel('Relative R.A.(pix)', size=10)
        # plt.ylabel('Relative Dec.(pix)', size=10)
        # # plt.title('radial profile  (' + star_name + ')'  f'{lst_Frame_name[1]}' ' at theta + pi/2', fontsize=10)
        
    
    
    
        # im4 = plt.subplot(2,1,2)
        # # assign plot to a new object
        # im = im4.imshow(img2, aspect='equal')
        
        # # add the bar
        # cbar = plt.colorbar(im)
        
        # im4.axis('off')
    
    
    
        #t1 = np.arange(0.0, 3.0, 0.01)
        
        # pour les profiles radiaux
        fig.set_size_inches(18.5, 10, forward = True)
        ax1 = plt.subplot(212)
        ax1.margins(0.05)           # Default margin is 0.05, value 0 means fit
        line11, = ax1.plot(zi1p_s, label= 'star') # profile radiale de l'intensité polarisée suivant le demi petit axe(star) à theta
        line22, = ax1.plot(zi1p_p, label= 'psf') # profile radiale de l'intensité polarisée suivant le demi petit axe  à theta
        # line1 = 
        # line2 =
        #ax1.legend(handles=[line1, line2])
        ax1.set_xlabel('r(pix)')
        ax1.set_ylabel('I (hdu)')
        ax1.set_title(f'{star_name}')
        
        # pour l'etoile
        ax2 = plt.subplot(221)
        #im2 = ax2.imshow(zp_s, aspect='equal') 
        ax2.margins(2, 2)           # Values >0.0 zoom out
        im2=ax2.imshow(zp_s, cmap ='inferno', vmin = np.min(zp_s), vmax = np.max(zp_s), origin='lower')# # full image of polarised intensity
        ax2.plot(Ell_rot_star_arr[1][0][0,:] , Ell_rot_star_arr[1][0][1,:])  # ellipse de l'intensitépolarisé à 10% de l'intensité max
        ax2.plot([x0p_s, x1p_s], [y0p_s, y1p_s], 'ro-')  # tracé du demi-grand axe de l'ellipse
        ax2.plot([x0p_s, x2p_s], [y0p_s, y2p_s], 'ro-')   # tracé du demi petit axe de l'ellipse
        #ax2.set_xlim(-nSubDim/2 +x_sf, nSubDim/2+x_sf) 
        ax2.set_xlim(0, nSubDim) 
        ax2.set_yticks([0, nSubDim/2, nSubDim])
        ax2.set_xlabel('x(pix)', fontsize = 24)
        ax2.set_ylabel('y(pix)', fontsize = 24)
        cbar = plt.colorbar(im2)
        #plt.colorbar(label='ADU', shrink = 0.6)
        ax2.set_title('star') 
        
        # pour la psf
        ax3 = plt.subplot(222)
        #im3 = ax2.imshow(zp_p, aspect='equal')
        ax3.margins(2, 2)   # Values in (-0.5, 0.0) zooms in to center
        im3=ax3.imshow(zp_p, cmap ='inferno', vmin = np.min(zp_p), vmax = np.max(zp_p), origin='lower')# # full image of polarised intensity
        ax3.plot(Ell_rot_psf_arr[1][0][0,:] , Ell_rot_psf_arr[1][0][1,:])  # ellipse de l'intensitépolarisé à 10% de l'intensité max
        ax3.plot([x0p_p, x1p_p], [y0p_p, y1p_p], 'ro-')  # tracé du demi-grand axe de l'ellipse
        ax3.plot([x0p_p, x2p_p], [y0p_p, y2p_p], 'ro-')   # tracé du demi petit axe de l'ellipse
        ax3.set_xlim(0, nSubDim) 
        ax3.set_yticks([0, nSubDim/2, nSubDim]) # pour fixer lécart de la graduation sur les axes
        ax3.set_xlabel('x(pix)', fontsize = 24) 
        #ax3.set_ylabel('y(pix)', fontsize = 12)
        ax3.set_ylim(0, nSubDim) 
        cbar = plt.colorbar(im3)
        ax3.set_title('psf')
        
            
        # plt.subplot(3,2,3) # pour l'intensité de la psf
        # plt.imshow(z_p, cmap ='inferno', vmin = np.min(z_s), vmax = np.max(z_s), origin='lower') # full image of intensity
        # plt.plot(Ell_rot_psf_arr[0][0][0,:] , Ell_rot_psf_arr[0][0][1,:]) # ellipse de l'intensité à 10% de l'intensité max
        # plt.plot([x0_p, x1_p], [y0_p, y1_p], 'ro-')  # tracé du demi-grand axe de l'ellipse
        # #plt.plot([x0, x1_], [y0, y1_], 'ro-')# tracé de l'autre demi-grand axe de l'ellipse
        # plt.plot([x0_p, x2_p], [y0_p, y2_p], 'ro-') # tracé du demi petit axe de l'ellipse
        # #plt.plot([x0, x2_], [y0, y2_], 'ro-') # tracé de l'autre demi petit axe de l'ellipse
        # plt.xlabel('Relative R.A.(pix)', size=10)
        # plt.ylabel('Relative Dec.(pix)', size=10)
        # plt.title('radial profile of  ('+star_name+ ')'  f'{lst_Frame_name[0]}'  ' at theta', fontsize=10)
        
        # plt.subplot(3,2,4) # pour l'intensité polarisée de la psf
        # plt.imshow(zp_p, cmap ='inferno', vmin = np.min(zp_p), vmax = np.max(zp_p), origin='lower')# # full image of polarised intensity
        # plt.plot(Ell_rot_psf_arr[1][0][0,:] , Ell_rot_psf_arr[1][0][1,:])  # ellipse de l'intensitépolarisé à 10% de l'intensité max
        # plt.plot([x0p_p, x1p_p], [y0p_p, y1p_p], 'ro-')  # tracé du demi-grand axe de l'ellipse
        # #plt.plot([x0p, x1p_], [y0p, y1p_], 'ro-')-')# tracé de l'autre demi-grand axe de l'ellipse
        # plt.plot([x0p_p, x2p_p], [y0p_p, y2p_p], 'ro-')   # tracé du demi petit axe de l'ellipse
        # #plt.plot([x0p, x2p_], [y0p, y2p_], 'ro-') # tracé de l'autre demi petit axe de l'ellipse
        # plt.xlabel('Relative R.A.(pix)', size=10)
        # plt.ylabel('Relative Dec.(pix)', size=10)
        # plt.title('radial profile  (' + star_name + ')'  f'{lst_Frame_name[1]}' ' at theta + pi/2', fontsize=10)
        
        # plt.subplot(3,2,5)
        # plt.plot(zi1_s) # profile radiale de l'intensité suivant le demi grand axe (star) à theta
        # plt.plot(zi1_p) # profile radiale de l'intensité suivant le demi petit axe (psf)   à theta
        # #plt.plot(zi1p) # profile radiale de l'intensité polarisée suivant le grand axe
        # plt.legend(["star", "psf"])
        # plt.title('radial profile of intensity demi grand axe')
        # plt.xlabel('r (mas)', size=10)          
        # plt.ylabel(r'Intensity in log$_{10}$ scale', size=10)
        # plt.subplot(1,2,2)
        # plt.plot(zi1p_s) # profile radiale de l'intensité polarisée suivant le demi petit axe(star) à theta
        #plt.plot(zi2) # profile radiale de l'intensité suivant le demi petit axe
        # plt.plot(zi1p_p) # profile radiale de l'intensité polarisée suivant le demi petit axe  à theta
        # plt.legend(["star at " + str(int(par_star_arr[0][0][4]*180/np.pi)) + ' °', "psf at " + str(int(par_star_arr[0][0][4]*180/np.pi)) + ' °'])
        # plt.title('radial profile of polarised intensity demi petit axe')
        # plt.xlabel('r (pix)', size=10) 
        # plt.ylabel(r'Intensity in log$_{10}$ scale', size=10)
       
        plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+'/plots/radial/radial_profile_at_a_given_orientation.pdf', 
                       dpi=100, bbox_inches ='tight')
        plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+'/plots/radial/radial_profile_at_a_given_orientation.png', 
                dpi=100, bbox_inches ='tight')
        
        plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/radial_profile_at_given_orientation/'+ star_name +'radial_profile_at_a_given_orientation.pdf', dpi=100, bbox_inches ='tight')
        plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/radial_profile_at_given_orientation/'+ star_name +'radial_profile_at_a_given_orientation.png', dpi=100, bbox_inches ='tight')
        plt.tight_layout()
        plt.show()
    

# star_name  = 'SW_Col'
# obsmod = 'both'
# star=log_image(star_name, obsmod)
