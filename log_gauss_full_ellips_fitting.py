#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:45:23 2022

@author: nbadolo
"""

"""
## This function plot the gauss ellips fits of each star of the full log for a many given intensity values  
"""


# ====================================================#
# Pour les fits  gaussiens de tous les objets resolus #
# ====================================================#



#packages
import numpy as np
import os
from matplotlib import pyplot as plt
from math import pi, cos, sin, atan
from astropy.nddata import Cutout2D
from astropy.io import fits
from scipy.stats import linregress
import scipy.optimize as opt
from sklearn.linear_model import LinearRegression
from skimage import io, color, measure, draw, img_as_bool
import pylab as plt
#import matplotlib.pyplot as plt

#%%
def log_image(star_name, obsmod): 
#%%       
    fdir= '/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+ '/'
    fdir_star = fdir + 'star/'+obsmod+ '/' 
    fdir_psf = fdir +'psf/'+obsmod+ '/'
    lst_fltr_star1 = os.listdir(fdir_star) #fait la liste des filtres contenus dans alone ou both
    print(lst_fltr_star1)
    n_lst_fltr_star1 = len(lst_fltr_star1)
    lst_fltr_star2 = []
    nDimfigj = [3, 4, 5]
    nDimfigk = [6, 7, 8]
    # star opening
    for p in range(n_lst_fltr_star1):
        fdir_fltr_data_star = fdir_star + lst_fltr_star1[p]
        lst_fltr_data_star = os.listdir(fdir_fltr_data_star) # fait la liste du contenu de cahque filtre
        n_lst_fltr_data_star = len(lst_fltr_data_star)
        if n_lst_fltr_data_star != 0:
            lst_fltr_star2.append(lst_fltr_star1[p]) # retient les filtre qui contiennent des fichiers 
    n_lst_fltr_star2 = len(lst_fltr_star2)           # et les concatene a la liste vide lst_fltr_star2
    print(lst_fltr_star2)
    
    #psf openig
    for l in range(n_lst_fltr_star2):
       
        fdir_star_fltr = fdir_star + lst_fltr_star2[l] +'/'
        #fdir_psf_fltr = fdir_psf + lst_fltr_star2[l] + '/'
                
        fname1='zpl_p23_make_polar_maps-ZPL_SCIENCE_P23_REDUCED'
        fname2='-zpl_science_p23_REDUCED'
        file_I_star= fdir_star_fltr + fname1+'_I'+fname2+'_I.fits'
        file_PI_star= fdir_star_fltr +fname1+'_PI'+fname2+'_PI.fits'
        file_DOLP_star= fdir_star_fltr +fname1+'_DOLP'+fname2+'_DOLP.fits'
        #file_AOLP_star= fdir_star_fltr + fname1+'_AOLP'+fname2+'_AOLP.fits'




    #creating lists
        file_lst=[file_I_star, file_PI_star]
          #,file_I_psf,file_PI_psf,file_DOLP_psf,file_AOLP_psf]
          
        nFrames = len(file_lst)
        lst_Frame_name = ['Intensity', 'Pol. Intensity'] 

    #parameters
        nDim=1024
        nSubDim = 150 # plage de pixels que l'on veut afficher
        size = (nSubDim, nSubDim)
        nDimfigj=[9,10,11]
        nDimfigk=[0,1,2]
        lst_threshold = [0.0095, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1]
        n_threshold = len(lst_threshold)
        vmin0 = 3.5
        vmax0 = 15
        pix2mas = 3.4  #en mas/pix
        x_min = -pix2mas*nSubDim//2
        x_max = pix2mas*(nSubDim//2-1)
        y_min = -pix2mas*nSubDim//2
        y_max = pix2mas*(nSubDim//2-1)
        position = (nDim//2,nDim//2)
        size = (nSubDim, nSubDim)
        
     # lists and arrays   
        x, y = np.meshgrid(np.arange(nSubDim),np.arange(nSubDim)) #cree un tableau 
        sub_v_arr = np.zeros((nFrames, nSubDim, nSubDim))
        Ellips_arr = np.zeros((nFrames, n_threshold,nSubDim,nSubDim))
        Ellips_im_arr = np.zeros((nFrames, n_threshold,nSubDim,nSubDim))
        Ellips_arr2 = np.zeros((n_threshold, nSubDim, nSubDim))
        Ellips_ = np.zeros((n_threshold, nSubDim))
        Ell_rot_arr = np.zeros((nFrames, n_threshold, 2, nSubDim))
        par_arr = np.zeros((nFrames, n_threshold, 5)) # les paramètres de l'étoile
        Vmin = np.zeros((nFrames, n_threshold)) # pour l'image totale
        Vmax = np.zeros((nFrames, n_threshold))
        Vmin_r = np.zeros((nFrames, n_threshold))
        Vmax_r = np.zeros((nFrames, n_threshold))# pour l'image reelle tronquee à au pourcentage
        Vmin_w = np.zeros((nFrames, n_threshold))
        Vmax_w = np.zeros((nFrames, n_threshold)) # pour l'image binaire
        
        strs = [str(x*100) + ' %' for x in lst_threshold]


    #opening file
        fsize = [0,1]       
        n_fsize = len (fsize)
        fltr_arr = np.empty(n_fsize, dtype = str)
        for z in range(n_fsize) :
            for i in range(nFrames):
                  hdu = fits.open(file_lst[i])[0]  
                  data = hdu.data
                  intensity = data[z,:,:]
                  fltr1 = hdu.header.get('HIERARCH ESO INS3 OPTI5 NAME')   
                  fltr2 = hdu.header.get('HIERARCH ESO INS3 OPTI6 NAME')
                  fltr_arr[0] = fltr1
                  fltr_arr[1] = fltr2
                  cutout = Cutout2D(intensity, position = position, size = size)
                  zoom_hdu = hdu.copy()
                  sub_v = cutout.data
                  
                  sub_v_arr[i] = sub_v
                  
                  for j in range(n_threshold):
                      Ellips_arr2[i] = sub_v
                      Ellips_im = np.zeros_like(sub_v_arr[i]) #creation d'un tableau de meme forme que sub_v
                      Ellips_im[sub_v > lst_threshold[j]*np.max(sub_v)] = sub_v[sub_v > lst_threshold[j]*np.max(sub_v)]# on retient les points d'intensité égale à 5% de Imax 
                      Ellips_im_arr[i][j] = Ellips_im
                      
                      Ellips = np.zeros_like(sub_v)          #creation d'un tableau de meme forme que sub_v
                      Ellips[sub_v > lst_threshold[j]*np.max(sub_v)] = 1   # on retient les points d'intensité 
                      Ellips_arr[i][j] = Ellips                # égale à 5% de Imax et à tous ces points 
                                                             # on donne 1 comme valeur d'intensité
                
                        
                      Vmin[i][j] = np.min(np.log10(sub_v+np.abs(np.min(sub_v))+10))
                      Vmax [i][j] = np.max(np.log10(sub_v+np.abs(np.min(sub_v))+10))
                      
                      Vmin_r[i][j] = np.min(np.log10(Ellips_im+np.abs(np.min(Ellips_im))+10))
                      Vmax_r[i][j] = np.max(np.log10(Ellips_im+np.abs(np.min(Ellips_im))+10))  
                      
                      Vmin_w[i][j] = np.min(np.log10(Ellips+np.abs(np.min(Ellips))+10))
                      Vmax_w[i][j] = np.max(np.log10(Ellips+np.abs(np.min(Ellips))+10))  
                 
                  
                      im_white = Ellips_arr[i][j]
                      im_real = Ellips_im_arr[i][j]
                      
                      
                      # linear_reg = np.polyfit(X, Y, 1, full = False, cov = True)
                      # alpha_rad = atan(linear_reg[0][0])    # recupération de la pente de la regression
                      # alpha_deg = alpha_rad*180/pi
                      # aa = linear_reg[0][0]
                      # bb = linear_reg[0][1]
                      # xx = np.arange(nSubDim)
                      # yy = aa*xx + bb
                      
                      
                      #slope, intercept, r, p, se = linregress(Y, X)
                     #image = img_as_bool(io.imread('bubble.jpg')[..., 0])
                      regions = measure.regionprops(measure.label(im_white))
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
                          template = np.zeros_like(im_white)
                          template[coords] = 1
                          return -np.sum(template == im_white)
                        
                      x_f, y_f, a_f, b_f, theta_f = opt.fmin(cost, (x_i, y_i, a_i, b_i, theta_i))
                      
                      #def ellips(t, x_f, y_f, a_f, bb_f, theta_f):
                      theta_f = np.pi/2 -theta_f
                      par_arr[i][j] = [x_f, y_f, a_f, b_f, theta_f]
                      
                      theta_f_deg = theta_f*180/pi
                      Ell = np.array([a_f*np.cos(t) , b_f*np.sin(t)])  
                             #u,v removed to keep the same center location
                      M_rot = np.array([[cos(theta_f) , -sin(theta_f)],[sin(theta_f) , cos(theta_f)]]) 
                      
                             #2-D rotation matrix
                        
                      Ell_rot_ = np.zeros((2, nSubDim))
                      Ell_rot = np.zeros((2, nSubDim))
                      for k in range(Ell.shape[1]):
                          Ell_rot_[:,k] = np.dot(M_rot,Ell[:,k]) # fait le produit scal de la matrice de rotation par chaq couple parametriq 
                          Ell_rot[:,k] = Ell_rot_[:,k]
                          Ell_rot[0,k] = Ell_rot[0,k] + x_f
                          Ell_rot[1,k] = Ell_rot[1,k] + y_f
                      #return Ell_rot.ravel() # .ravel permet de passer de deux dimension à une seule
                          Ell_rot_arr[i][j][:,k] = Ell_rot[:,k] # ajout des cooordonnées du centre de l'ellipse (changement de repère)
                      
                     
                       ### plots   
                    #   plt.figure('white ellipse contour at ' + f'{strs[j]}' +' for ' + f'{lst_Frame_name[i]}'+' of '+ f'{star_name}')
                    #   plt.clf()
                    #   plt.imshow(np.log10(Ellips_arr[i][j]+np.abs(np.min(Ellips_arr[i][j]))+10), cmap ='inferno', vmin=Vmin_w[i][j], vmax=Vmax_w[i][j], origin='lower')
                    #     #plt.plot( u + Ell_rot[0,:] , v + Ell_rot[1,:],'darkorange' )  #rotated ellipse
                    #   plt.plot( Ell_rot[0,:] , Ell_rot[1,:],'darkorange' ) #rotated fit
                    #     #plt.grid(color='lightgray',linestyle='--')
                    #   plt.show()
                    #   # plt.title('white ellipse contour at ' + f'{strs[j]}' + ' for ' + f'{lst_Frame_name[i]}' +' of '+ f'{star_name}', fontsize=10)
                    #   # plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/SW_Col/plots/fits/log_scale/fully_automatic/' +'white_ellips_contour_at_' + strs[j] + '_for_' +
                    #   #             f'{lst_Frame_name[i]}' + '_' + f'{fltr_arr[z]}'+ '.pdf',  dpi=100, bbox_inches ='tight')
                    #   # plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/SW_Col/plots/fits/log_scale/fully_automatic/' +'white_ellips_contour_at_' + strs[j] + '_for_' + 
                    #   #             f'{lst_Frame_name[i]}' +  '_' + f'{fltr_arr[z]}'+ '.png', dpi=100, bbox_inches ='tight')
                    
                      
                    #   # plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/All_plots/radial_profile/log_scale/fully_automatic/' +'white_ellips_contour_at_' + strs[j] + '_for_' +
                    #   #             f'{lst_Frame_name[i]}' + '_' + f'{fltr_arr[z]}'+ '.pdf',  dpi=100, bbox_inches ='tight')
                    #   # plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/All_plots/radial_profile/log_scale/fully_automatic/' +'white_ellips_contour_at_' + strs[j] + '_for_' +
                    #   #             f'{lst_Frame_name[i]}' + '_' + f'{fltr_arr[z]}'+ '.png',  dpi=100, bbox_inches ='tight')
                      
                      
                    #   plt.figure('real image contour at ' + f'{strs[j]}')
                    #   plt.clf()
                    #   plt.imshow(np.log10(Ellips_im_arr[i][j] + np.abs(np.min(Ellips_im_arr[i][j]))+10), cmap ='inferno', vmin = Vmin_r[i][j], vmax = Vmax_r[i][j], origin='lower')
                    # #plt.plot( u + Ell_rot[0,:] , v + Ell_rot[1,:],'darkorange' )  #rotated ellipse
                    #   plt.plot( Ell_rot[0,:] , Ell_rot[1,:],'darkorange' ) #rotated fit
                    # #plt.grid(color='lightgray',linestyle='--')
                      
                    #   plt.show()
                    #   # plt.title('real image contour at ' + f'{strs[j]}' + ' for ' + f'{lst_Frame_name[i]}' +' of '+ f'{star_name}', fontsize=10)
                    #   # plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+ star_name +'/plots/fits/log_scale/fully_automatic/' +'real_image_contour_at_' + strs[j] + '_for_' + 
                    #   #             f'{lst_Frame_name[i]}' + '_' + f'{fltr_arr[z]}'+ '.pdf',  dpi=100, bbox_inches ='tight')
            
            
                    #   # plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+ star_name +'/plots/fits/log_scale/fully_automatic/' +'real_image_contour_at_' + strs[j] + '_for_' + 
                    #   #             f'{lst_Frame_name[i]}' + '_' + f'{fltr_arr[z]}'+ '.png', dpi=100, bbox_inches ='tight')
                    #   # plt.tight_layout()
            
                      plt.figure('full image and contour at ' + f'{strs[j]}')
                      plt.clf()
                      plt.imshow(np.log10(sub_v_arr[i]+np.abs(np.min(sub_v_arr[i]))+10), cmap ='inferno', vmin=Vmin_w[i][j], vmax=Vmax_r[i][j], origin='lower')
                      #plt.plot( u + Ell_rot[0,:] , v + Ell_rot[1,:],'darkorange' )  #rotated ellipse
                      plt.plot( Ell_rot[0,:] , Ell_rot[1,:],'darkorange') #rotated fit
                      plt.text(size[0]//2, 5*size[1]//6.,
                                f'{strs[j]}' + ' ' + f'{lst_Frame_name[i]}' + ' of ' + f'{star_name}', color='w',
                                  fontsize='large', ha='center')
                      plt.ylabel('Y [pix]', size=10)
                      plt.xlabel('X [pix]', size=10)
                      #plt.grid(color='lightgray',linestyle='--')
                      #plt.show()
                      #plt.title('full image and contour at ' + f'{strs[j]}' + ' for ' + f'{lst_Frame_name[i]} '+' of '+ f'{star_name}', fontsize=10)
                      plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+ star_name +'/plots/fits/log_scale/fully_automatic/' +'full_image_and_contour_at_' + strs[j] + 
                                   '_for_' + f'{lst_Frame_name[i]}' + '_' + f'{fltr_arr[z]}'+ '.pdf', dpi=100, bbox_inches ='tight')
                      plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+ star_name +'/plots/fits/log_scale/fully_automatic/' +'full_image_and_contour_at_' + strs[j] + 
                                   '_for_' + f'{lst_Frame_name[i]}' + '_' + f'{fltr_arr[z]}'+'.png', dpi=100, bbox_inches ='tight')
                      
                      
                      plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/radial_profile/log_scale/fully_automatic/'+ star_name +'_full_image_and_contour_at_' + strs[j] + 
                                   '_for_' + f'{lst_Frame_name[i]}' + '_' + f'{fltr_arr[z]}'+ '.pdf', dpi=100, bbox_inches ='tight')
                      plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/radial_profile/log_scale/fully_automatic/'+ star_name + '_full_image_and_contour_at_' + strs[j] + 
                                   '_for_' + f'{lst_Frame_name[i]}' + '_' + f'{fltr_arr[z]}'+'.png', dpi=100, bbox_inches ='tight')
                      plt.tight_layout()
                                            
                  plt.figure('full image and all the  contours' + f'{strs[j]}')
                  plt.clf()
                  plt.imshow(np.log10(sub_v_arr[i] + np.abs(np.min(sub_v_arr[i]))+10), cmap ='inferno', vmin=Vmin_r[i][j], vmax=Vmax_r[i][j], origin='lower' )
                  #plt.plot( u + Ell_rot[0,:] , v + Ell_rot[1,:],'darkorange' )  #rotated ellipse
                  for j in range(n_threshold): 
                  
                      plt.plot(Ell_rot_arr[i][j][0,:], Ell_rot_arr[i][j][1,:])
                      
                       #plt.plot( Ell_rot[0,:] , Ell_rot[1,:],'darkorange' ) #rotated fit
                       #plt.grid(color='lightgray',linestyle='--')
                     # plt.show()
                  plt.text(size[0]//2, 5*size[1]//6.,
                               f'{lst_Frame_name[i]}' + ' of '+ f'{star_name}',color='w',
                              fontsize='large', ha='center')
                  plt.ylabel('Y [pix]', size=10)
                  plt.xlabel('X [pix]', size=10)
                  #plt.title('full image and all the  contours ' + ' for ' + f'{lst_Frame_name[i]}'+' of '+ f'{star_name}', fontsize=10)
                  
                  plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+ star_name +'/plots/fits/log_scale/fully_automatic/' +'full_image_and_all_the_contours' +'_for_' +
                               f'{lst_Frame_name[i]}' + '_' + f'{fltr_arr[z]}'+ '.pdf', dpi=100, bbox_inches ='tight')                                        
                  plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+ star_name +'/plots/fits/log_scale/fully_automatic/' +'full_image_and_all_the_contours'  + '_for_' +
                               f'{lst_Frame_name[i]}' + '_' + f'{fltr_arr[z]}'+ '.png', dpi=100, bbox_inches ='tight')
                  
                 
                  plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/radial_profile/log_scale/fully_automatic/'+ star_name  +'_full_image_and_all_the_contours' +'_for_' +
                               f'{lst_Frame_name[i]}' + '_' + f'{fltr_arr[z]}'+ '.pdf', dpi=100, bbox_inches ='tight')
                  plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/radial_profile/log_scale/fully_automatic/'+ star_name +'_full_image_and_all_the_contours'  + '_for_' +
                               f'{lst_Frame_name[i]}' + '_' + f'{fltr_arr[z]}'+ '.png', dpi=100, bbox_inches ='tight')
                  plt.tight_layout()
    
    #return()

log_image('SW_Col', 'both')