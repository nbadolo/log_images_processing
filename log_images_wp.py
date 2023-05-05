#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:29:41 2022

@author: nbadolo
"""

"""
Code simplifié pour l'affichage simultané de tous les alone et both  des étoiles 
sans psf:  flux d'intensité
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
    # Parameters
    nDim = 1024
    nSubDim = 200 # plage de pixels que l'on veut afficher
    size = (nSubDim, nSubDim)
    # nDimfigj = [3, 4, 5]
    # nDimfigk = [6, 7, 8]
    # vmin0 = 3.5
    # vmax0 = 15
    pix2mas = 6.8  #en mas/pix
    x_min = -pix2mas*nSubDim//2
    x_max = pix2mas*(nSubDim//2-1)
    y_min = -pix2mas*nSubDim//2
    y_max = pix2mas*(nSubDim//2-1)
    X, Y= np.meshgrid(np.linspace(-100,99,200), np.linspace(-100,99,200))
    X_, Y_= np.meshgrid(np.linspace(-nDim/2,nDim/2-1,nDim), np.linspace(-nDim/2,nDim/2-1,nDim))
  
    X *= pix2mas
    Y *= pix2mas
    X_ *= pix2mas
    Y_ *= pix2mas
  
    X_step = 10
    X_step_ = 50
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
        nDim = 1024
        #nSubDim = 30 # plage de pixels que l'on veut afficher en ehelle lineaire
        nSubDim = 100 # plage de pixels que l'on veut afficher en ehelle log
        size = (nSubDim, nSubDim)
        
          # nDimfigj = [3, 4, 5]
          # nDimfigk = [6, 7, 8]
    
        pix2mas = 3.4  # en mas/pix
       
        x_min = -pix2mas*nSubDim//2
        x_max = pix2mas*(nSubDim//2-1)
        y_min = -pix2mas*nSubDim//2
        y_max = pix2mas*(nSubDim//2-1)
        
        x_min_log = -pix2mas*nSubDim//2
        x_max_log = pix2mas*(nSubDim//2-1)
        y_min_log = -pix2mas*nSubDim//2
        y_max_log = pix2mas*(nSubDim//2-1)
        X, Y= np.meshgrid(np.linspace(-nSubDim/2,nSubDim/2-1,nSubDim), np.linspace(-nSubDim/2,nSubDim/2-1,nSubDim))
        X_, Y_= np.meshgrid(np.linspace(-nDim/2,nDim/2-1,nDim), np.linspace(-nDim/2,nDim/2-1,nDim))
        
        X *= pix2mas
        Y *= pix2mas
        X_ *= pix2mas
        Y_ *= pix2mas
        
        X_step = 10
        X_step_ = 50
        nx = ny = 20
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
        
        for i in range (nFrames):
              hdu = fits.open(file_lst[i])[0]   
              data = hdu.data   
              i_v = data[0,:,:]
              fltr = hdu.header.get('HIERARCH ESO INS3 OPTI5 NAME')     
              #print(fltr)                   
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
        fig = plt.figure(f'{star_name}' +'(' + f'{fltr}' + ' linear scale' + ')')
        fig.set_size_inches(12, 6, forward = True)
        for i in range (nFrames -2):
              plt.subplot(2,2,i+1)
              if  i < 3:
                  
                  plt.imshow(sub_v_arr[i], cmap='inferno', origin='lower',
                      vmin= np.min(sub_v_arr[i]), vmax= np.max(sub_v_arr[i]), extent = [x_min , x_max, y_min , y_max])
                      
                  plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                                f'{star_name}' + '_' + f'{im_name_lst[i]}' + ' in '+ f'{fltr}' + ' band', color='w',
                            fontsize='large', ha='center')
                  plt.colorbar(label='ADU', shrink = 0.6)
                  
              elif i == 3:
         
                  
                    plt.imshow(sub_v_arr[1], cmap ='inferno', origin='lower',vmin= np.min(sub_v_arr[1]), 
                                vmax=np.max(sub_v_arr[1]), extent = [x_min , x_max, y_min , y_max])   
                    plt.colorbar(label='ADU', shrink = 0.6)       
                    q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2[::X_step,::X_step], V2[::X_step,::X_step], color='w', pivot='mid')
                    plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')                 
                    plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                            'Pol. vect.' + ' in '+ f'{fltr}' + ' band', color='y',
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
                          '/plots/'+ star_name +'_' + fltr + '_lin' + '.pdf', 
                          dpi=100, bbox_inches ='tight')
        
        plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+
                          '/plots/'+ star_name +'_' + fltr  + '_lin' + '.png', 
                          dpi=100, bbox_inches ='tight')
        
        plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Intensities/linear_scale/'+ 
                    star_name +'_' + fltr + '_lin' + '.pdf', 
                          dpi=100, bbox_inches ='tight')
        plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Intensities/linear_scale/'+ 
                    star_name +'_' + fltr + '_lin' + '.png', 
                          dpi=100, bbox_inches ='tight')
        
        plt.tight_layout()
        
        
        # log scale        
        plt.clf()
        fig = plt.figure(f'{star_name}' +'(' + f'{fltr}' + ' log scale' + ')')
        fig.set_size_inches(12, 6, forward = True)
        for i in range (nFrames -2):
              plt.subplot(2,2,i+1)
              if  i < 3:                   
                  im = np.log10(sub_v_arr[i] + np.abs(np.min(sub_v_arr[i]) + 10)) # add an ofset because negatives values 
                  plt.imshow(im, cmap='inferno', origin='lower',
                        vmin = np.min(im), vmax= np.max(im), extent = [x_min , x_max, y_min , y_max])
                       
                  plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                                  f'{star_name}' + '_' + f'{im_name_lst[i]}' + ' in '+ f'{fltr}' + ' band', color='w',
                              fontsize='large', ha='center')
                  plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
              elif i == 3 :
                  im_ = np.log10(sub_v_arr[1] + np.abs(np.min(sub_v_arr[1]) + 10))
                  plt.imshow(im_, cmap ='inferno', origin='lower', 
                                    extent = [x_min_log , x_max_log, y_min_log , y_max_log])   
                  plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)       
                  q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2[::X_step,::X_step], V2[::X_step,::X_step], color='w', pivot='mid')
                  plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')                       
                  plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                    'Pol. vect.' + ' in '+ f'{fltr}' + ' band', color='y',
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
                          '/plots/'+ star_name +'_' + fltr + '_log' + '.pdf', 
                          dpi=100, bbox_inches ='tight')
        
        plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+
                          '/plots/'+ star_name +'_' + fltr + '_log' + '.png', 
                          dpi=100, bbox_inches ='tight')
        
        plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Intensities/log_scale/'+ 
                    star_name +'_' + fltr  + '_log' + '.pdf', 
                          dpi=100, bbox_inches ='tight')
        plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Intensities/log_scale/'+ 
                    star_name +'_' + fltr + '_log' + '.png', 
                          dpi=100, bbox_inches ='tight')
        
        plt.tight_layout()
        msg1='reduction okay for '+ star_name+'_Cam1'
          #return(msg1)
        print(msg1)
        
    for m in range(n_lst_fltr_star2):
          fdir_star_fltr = fdir_star + lst_fltr_star2[m] +'/'
          fdir_psf_fltr = fdir_psf + lst_fltr_star2[m] + '/'
        
          fname1='zpl_p23_make_polar_maps-ZPL_SCIENCE_P23_REDUCED'
          fname2='-zpl_science_p23_REDUCED'
          file_I_star = fdir_star_fltr + fname1+'_I'+fname2+'_I.fits'
          file_PI_star = fdir_star_fltr +fname1+'_PI'+fname2+'_PI.fits'
          file_DOLP_star = fdir_star_fltr +fname1+'_DOLP'+fname2+'_DOLP.fits'
          file_AOLP_star = fdir_star_fltr + fname1 + '_AOLP'+fname2+'_AOLP.fits'
          file_Q_star= fdir_star_fltr + fname1+'_Q'+fname2+'_Q.fits'
          file_U_star= fdir_star_fltr + fname1+'_U'+fname2+'_U.fits'
         
          file_lst_ = [file_I_star, file_PI_star, file_DOLP_star, file_AOLP_star,file_Q_star,file_U_star]
          nFrames_ = len(file_lst_)
        
          
        
          mean_sub_v_arr_ = np.empty((nFrames_,nSubDim//2-1)) # pour le profil moyen d'intensité
          sub_v_arr_ = np.empty((nFrames_,nSubDim,nSubDim))
          im_name_lst = ['I','PI','DOLP']
          Vmin2_ = np.empty((nFrames_))
          Vmax2_ = np.empty((nFrames_))
        
          x, y = np.meshgrid(np.arange(nSubDim), np.arange(nSubDim)) #cree un tableau 
        
          R = np.sqrt((x-nSubDim/2)**2+(y-nSubDim/2)**2)
          r = np.linspace(1,nSubDim//2-1,nSubDim//2-1)
        
          r_mas = pix2mas*r #  où r est en pixels et r_mas en millièmes d'arcseconde
         
         
          for i in range (nFrames_):
                hdu_ = fits.open(file_lst_[i])[0]   
                data_ = hdu_.data   
                i_v_ = data_[1,:,:]
                fltr_ = hdu_.header.get('HIERARCH ESO INS3 OPTI6 NAME')
               
                cutout_ = Cutout2D(i_v_, position = position, size=size)
                zoom_hdu = hdu_.copy()
                sub_v_ = cutout_.data
             
                f = lambda r : sub_v_[(R >= r-0.5) & (R < r+0.5)].mean()   
                mean_sub_v_ = np.vectorize(f)(r) 
             
                mean_sub_v_arr_[i] = mean_sub_v_ 
                sub_v_arr_[i] = sub_v_
               
                Vmin_ = np.min(sub_v_)
                Vmax_ = np.max(sub_v_)
               
                DOLP_ = sub_v_arr_[2]
                #AOLP_2_ = 0
                # ii = (sub_v_arr_[4] == 0)
                # if True in ii:
                #     sub_v_arr_[4][ii] = sub_v_arr_[4][ii] + 0.0001  # introduction d'un ofset pour les valeurs de Q == 0
                    
                AOLP_2_ = 0.5*np.arctan2(sub_v_arr_[5], sub_v_arr_[4]) 
                # AOLP_2_ =np.reshape(AOLP_2_,(nSubDim,nSubDim))
                U2_ = DOLP_*np.cos(-(AOLP_2_ + np.pi/2))
                V2_ = DOLP_*np.sin(-(AOLP_2_ + np.pi/2))
         
           ## linear scale      
          plt.clf()
          fig = plt.figure(f'{star_name}' +'(' + f'{fltr_}' + '_Cam2' + ')')
          fig.set_size_inches(12,6, forward = True)
          for i in range (nFrames_ -2):
                 plt.subplot(2,2,i+1)
                 if i < 3:
                             
                     plt.imshow(sub_v_arr_[i], cmap='inferno', origin='lower',
                     extent = [x_min , x_max, y_min , y_max])
                    
                     plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                               f'{star_name}' + '_' + f'{im_name_lst[i]}' + ' in '+ f'{fltr_}' + ' band', color='w',
                           fontsize='large', ha='center')
                     plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
                   
                 elif i == 3:
                   
                     plt.imshow(sub_v_arr_[1], cmap ='inferno', origin='lower',
                                   extent = [x_min , x_max, y_min , y_max])   
                     plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)       
                     q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2_[::X_step,::X_step], V2_[::X_step,::X_step], color='w', pivot='mid')
                     plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')
                     plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                      'Pol. vect.' + ' in '+ f'{fltr_}' + ' band', color='w',
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
                             '/plots/'+star_name+'_' + fltr_ +'_lin' + '.pdf', 
                             dpi=100, bbox_inches ='tight')
           
           
          plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+
                             '/plots/'+star_name+'_' + fltr_ +'_lin' + '.png', 
                             dpi=100, bbox_inches ='tight')
          plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Intensities/linear_scale/'+ star_name+
                       '_' + fltr_ + '_lin' + '.pdf', 
                             dpi=100, bbox_inches ='tight')
          plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Intensities/linear_scale/'+ 
                       star_name +'_' + fltr_ + '_lin' + '.png', 
                             dpi=100, bbox_inches ='tight')
         
          plt.tight_layout()
         
          ## log scale
          plt.clf()
          fig = plt.figure(f'{star_name}' +'(' + f'{fltr_}' + '_Cam2' + ')')
          fig.set_size_inches(12, 6, forward = True)
          for i in range (nFrames_ -2):
                plt.subplot(2,2,i+1)
                if  i < 3:
                   
                    # log scale    
                    im_ = np.log10(sub_v_arr_[i] + np.abs(np.min(sub_v_arr_[i]) + 10))# add an ofset because negatives values 
                    plt.imshow(im_, cmap='inferno', origin='lower',
                          vmin = np.min(im_), vmax= np.max(im_), extent = [x_min , x_max, y_min , y_max])
                        
                    plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                                    f'{star_name}' + '_' + f'{im_name_lst[i]}' + ' in '+ f'{fltr_}' + ' band', color='w',
                                fontsize='large', ha='center')
                    plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
                elif i == 3 :
               
                      im_= np.log10(sub_v_arr_[1] + np.abs(np.min(sub_v_arr_[1]) + 10))
                      plt.imshow(im_, cmap ='inferno', origin='lower',vmin = np.min(im_), 
                                    vmax= np.max(im_), extent = [x_min , x_max, y_min , y_max])   
                      plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)       
                      q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2_[::X_step,::X_step], V2_[::X_step,::X_step], color='w', pivot='mid')
                      plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')                       
                      plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                    'Pol. vect.' + ' in '+ f'{fltr_}' + ' band', color='y',
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
                            '/plots/'+ star_name +'_' + fltr_ + '_log' + '.pdf', 
                            dpi=100, bbox_inches ='tight')
         
          plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+
                            '/plots/'+ star_name +'_' + fltr_ + '_log' + '.png', 
                            dpi=100, bbox_inches ='tight')
         
          plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Intensities/log_scale/'+ 
                      star_name +'_' + fltr_ + '_log' + '.pdf', 
                            dpi=100, bbox_inches ='tight')
          plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Intensities/log_scale/'+ 
                      star_name +'_' + fltr_ + '_log' + '.png', 
                            dpi=100, bbox_inches ='tight')
         
          plt.tight_layout()
       
    msg2='reduction okay for '+ star_name +'_Cam2'
    print(msg2)
    msg= 'reduction okay for ' + star_name
    return(msg)

log_image('SW_Col', 'both')