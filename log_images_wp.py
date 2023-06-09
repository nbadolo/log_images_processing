#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:29:41 2022

@author: nbadolo
"""

"""
Code simplifié pour l'affichage simultané de tous les alone et both  des étoiles 
sans psf:  flux d'intensité. Code okay !!!
"""

import numpy as np
import os
import scipy 
from os.path import exists
from astropy.io import fits
from scipy import optimize
from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.pyplot import Figure, subplot
import webbrowser
#%% 
def log_image(star_name, obsmod):
#%%       
    fdir= '/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+ '/'
    fdir_star = fdir + 'star/'+obsmod+ '/' 
    fdir_psf = fdir +'psf/'+obsmod+ '/'
    lst_fltr_star1 = os.listdir(fdir_star)
    #print(lst_fltr_star1)
    n_lst_fltr_star1 = len(lst_fltr_star1)
    #print(n_lst_fltr_star1)
    lst_fltr_star2 = []
    nDimfigj = [3, 4, 5]
    nDimfigk = [6, 7, 8]
    
    ## Parameters
    nDim = 1024
    nSubDim = 100 # plage de pixels que l'on veut afficher
    size = (nSubDim, nSubDim)
    # nDimfigj = [3, 4, 5]
    # nDimfigk = [6, 7, 8]
    # vmin0 = 3.5
    # vmax0 = 15
    pix2mas = 3.4  #en mas/pix
    x_min = -pix2mas*nSubDim//2
    x_max = pix2mas*(nSubDim//2-1)
    y_min = -pix2mas*nSubDim//2
    y_max = pix2mas*(nSubDim//2-1)
    X, Y= np.meshgrid(np.linspace(-nSubDim/2,nSubDim/2-1,nSubDim), np.linspace(-nSubDim/2,nSubDim/2-1,nSubDim))
    X *= pix2mas
    Y *= pix2mas
    
    
    
  
    X_step = 10
    X_step_ = 50
    nx = ny = 20
    position = (nDim//2,nDim//2)
    size = (nSubDim, nSubDim)
    
    #Recherche des filtres contenant des données
    for p in range(n_lst_fltr_star1):
        fdir_fltr_data_star = fdir_star + lst_fltr_star1[p]
        lst_fltr_data_star = os.listdir(fdir_fltr_data_star) 
        n_lst_fltr_data_star = len(lst_fltr_data_star)
        if n_lst_fltr_data_star != 0:
            lst_fltr_star2.append(lst_fltr_star1[p])
    n_lst_fltr_star2 = len(lst_fltr_star2)
    #print(lst_fltr_star2)
    
    
    for l in range(n_lst_fltr_star2):
       
        fdir_star_fltr = fdir_star + lst_fltr_star2[l] +'/'
        fdir_psf_fltr = fdir_psf + lst_fltr_star2[l] + '/'
                
        fname1='zpl_p23_make_polar_maps-ZPL_SCIENCE_P23_REDUCED'
        fname2='-zpl_science_p23_REDUCED'
        file_I_star= fdir_star_fltr + fname1+'_I'+fname2+'_I.fits'
        file_PI_star= fdir_star_fltr +fname1+'_PI'+fname2+'_PI.fits'
        file_DOLP_star= fdir_star_fltr +fname1+'_DOLP'+fname2+'_DOLP.fits'
        file_AOLP_star= fdir_star_fltr + fname1+'_AOLP'+fname2+'_AOLP.fits'
        file_Q_star= fdir_star_fltr + fname1+'_Q'+fname2+'_Q.fits'
        file_U_star= fdir_star_fltr + fname1+'_U'+fname2+'_U.fits'
        
        file_lst = [file_I_star, file_PI_star, file_DOLP_star, file_AOLP_star, file_Q_star, file_U_star]
        nFrames = len(file_lst)
        file_lst2 = [file_I_star, file_PI_star, file_DOLP_star, file_AOLP_star]
        nFrames2 = len(file_lst2)
      
        
        mean_sub_v_arr = np.empty((nFrames,nSubDim//2-1))
        #mean_sub_v_arr_l =  np.empty((nFrames,nSubDim_l//2-1))
        
        sub_v_arr = np.empty((nFrames,nSubDim,nSubDim))
        #sub_v_arr_l = np.empty((nFrames,nSubDim_l,nSubDim_l))
        im_name_lst = ['I','PI','DOLP', 'AOLP']
        Vmin = np.zeros((nFrames))
        Vmax = np.zeros((nFrames))
        
        position = (nDim//2,nDim//2)
        size = (nSubDim, nSubDim)
        
        x, y = np.meshgrid(np.arange(nSubDim), np.arange(nSubDim)) #cree un tableau 
        
        R = np.sqrt((x-nSubDim/2)**2+(y-nSubDim/2)**2)
        r = np.linspace(1,nSubDim//2-1,nSubDim//2-1) # creation d'un tableau de distance radiale
        
        r_mas = pix2mas*r #  où r est en pixels et r_mas en millièmes d'arcseconde
        fsize = [0,1]       
        n_fsize = len (fsize)
        fltr_arr = np.empty(n_fsize, dtype = str)
        for z in range(n_fsize) :
            for i in range (nFrames):
                  hdu = fits.open(file_lst[i])[0]   
                  data = hdu.data   
                  i_v = data[0,:,:]
                  
                  fltr1 = hdu.header.get('HIERARCH ESO INS3 OPTI5 NAME')   
                  fltr2 = hdu.header.get('HIERARCH ESO INS3 OPTI6 NAME')
                  fltr_arr[0] = fltr1
                  fltr_arr[1] = fltr2
                  
                  cutout = Cutout2D(i_v, position=position, size=size)
                  zoom_hdu = hdu.copy()
                  sub_v = cutout.data
                  
                  f = lambda r : sub_v[(R >= r-0.5) & (R < r+0.5)].mean()   
                  mean_sub_v = np.vectorize(f)(r) 
                
                  mean_sub_v_arr[i] = mean_sub_v 
                  sub_v_arr[i] = sub_v
                  
                  
                  
                  DOLP = sub_v_arr[2]
                  ii = (sub_v_arr[4] == 0)
                  if True in ii:
                     sub_v_arr[4][ii] = sub_v_arr[4][ii] + 0.0001  # introduction d'un ofset pour les valeurs de Q == 0
                        
                  AOLP_2 = 0.5*np.arctan2(sub_v_arr[5], sub_v_arr[4])
                  U2 = DOLP*np.cos(-(AOLP_2 + np.pi/2))
                  V2 = DOLP*np.sin(-(AOLP_2 + np.pi/2))
            
            print(Vmin)
            print(Vmax)
            # linear scale
            plt.clf()
            fig = plt.figure(f'{star_name}' +'(' + f'{fltr_arr[z]}' + ' linear scale' + ')')
            fig.set_size_inches(12, 6, forward = True)
            for i in range (nFrames -2):
                  plt.subplot(2,2,i+1)
                  if  i < 3:
                      
                      plt.imshow(sub_v_arr[i], cmap='inferno', origin='lower',
                          vmin= np.min(sub_v_arr[i]), vmax= np.max(sub_v_arr[i]), extent = [x_min , x_max, y_min , y_max])
                          
                      plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                                    f'{star_name}' + '_' + f'{im_name_lst[i]}' + ' in '+ f'{fltr_arr[z]}' + ' band', color='w',
                                fontsize='large', ha='center')
                      plt.colorbar(label='ADU', shrink = 0.6)
                      
                  elif i == 3:
             
                      
                        plt.imshow(sub_v_arr[1], cmap ='inferno', origin='lower',vmin= np.min(sub_v_arr[1]), 
                                    vmax=np.max(sub_v_arr[1]), extent = [x_min , x_max, y_min , y_max])   
                        plt.colorbar(label='ADU', shrink = 0.6)       
                        q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2[::X_step,::X_step], V2[::X_step,::X_step], color='w', pivot='mid')
                        plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')                 
                        plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                                'Pol. vect.' + ' in '+ f'{fltr_arr[z]}' + ' band', color='y',
                                    fontsize='large', ha='center')
    
                  if i == 0:
                      plt.ylabel('Relative Dec.(mas)', size=10)
                  else:                 
                      if i == 2:
                          plt.ylabel('Relative Dec.(mas)', size=10)
                          plt.xlabel('Relative R.A.(mas)', size=10)
                      else:
                          if i == 3:
                              plt.xlabel('Relative R.A.(mas)', size=10)
                      
            plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+
                              '/plots/no_psf/' + star_name + '_' + f'{fltr_arr[z]}' + '_lin' + '.pdf', 
                              dpi=100, bbox_inches ='tight')
            
            plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+
                              '/plots/no_psf/' + star_name + '_' + f'{fltr_arr[z]}'  + '_lin' + '.png', 
                              dpi=100, bbox_inches ='tight')
            
            plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Intensities/linear_scale/'+ 
                        star_name +'_' + f'{fltr_arr[z]}' + '_lin' + '.pdf', 
                              dpi=100, bbox_inches ='tight')
            plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Intensities/linear_scale/'+ 
                        star_name +'_' + f'{fltr_arr[z]}' + '_lin' + '.png', 
                              dpi=100, bbox_inches ='tight')
            
            plt.tight_layout()
            
            
            
            
            # log scale        
            plt.clf()
            fig = plt.figure(f'{star_name}' +'(' + f'{fltr_arr[z]}' + ' log scale' + ')')
            fig.set_size_inches(12, 6, forward = True)
            for i in range (nFrames -2):
                  plt.subplot(2,2,i+1)
                  if  i < 3:                   
                      im = np.log10(sub_v_arr[i] + np.abs(np.min(sub_v_arr[i]) + 10)) # add an ofset because negatives values 
                      plt.imshow(im, cmap='inferno', origin='lower',
                            vmin = np.min(im), vmax= np.max(im), extent = [x_min , x_max, y_min , y_max])
                           
                      plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                                      f'{star_name}' + '_' + f'{im_name_lst[i]}' + ' in '+ f'{fltr_arr[z]}' + ' band', color='w',
                                  fontsize='large', ha='center')
                      plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
                  elif i == 3 :
                      im_ = np.log10(sub_v_arr[1] + np.abs(np.min(sub_v_arr[1]) + 10))
                      plt.imshow(im_, cmap ='inferno', origin='lower', 
                                        extent = [x_min , x_max, y_min , y_max])   
                      plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)       
                      q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2[::X_step,::X_step], V2[::X_step,::X_step], color='w', pivot='mid')
                      plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')                       
                      plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                        'Pol. vect.' + ' in '+ f'{fltr_arr[z]}' + ' band', color='y',
                                  fontsize='large', ha='center')
                  if i == 0:
                      plt.ylabel('Relative Dec.(mas)', size=10)
                  else:                 
                      if i == 2:
                          plt.ylabel('Relative Dec.(mas)', size=10)
                          plt.xlabel('Relative R.A.(mas)', size=10)
                      else:
                          if i == 3:
                              plt.xlabel('Relative R.A.(mas)', size=10)
                      
            plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+ star_name +
                              '/plots/no_psf/'+ star_name +'_' + f'{fltr_arr[z]}' + '_log' + '.pdf', 
                              dpi=100, bbox_inches ='tight')
            
            plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+
                              '/plots/no_psf/'+ star_name +'_' + f'{fltr_arr[z]}' + '_log' + '.png', 
                              dpi=100, bbox_inches ='tight')
            
            plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Intensities/log_scale/'+ 
                        star_name +'_' + f'{fltr_arr[z]}'  + '_log' + '.pdf', 
                              dpi=100, bbox_inches ='tight')
            plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Intensities/log_scale/'+ 
                        star_name +'_' + f'{fltr_arr[z]}' + '_log' + '.png', 
                              dpi=100, bbox_inches ='tight')
            
            plt.tight_layout()
            msg1='reduction okay for '+ star_name+'_Cam1'
          #return(msg1)
        print(msg1)
        
    return()

log_image('Y_Scl', 'both')