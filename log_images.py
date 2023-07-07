#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 11:13:32 2022

@author: nbadolo
"""

"""
Code simplifié pour l'affichage simultané de toutes les étoiles(avec psf) de alone et both  ainsi que leur psf: flux 
et profile radial d'intensité
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
#star_name = 'SW_Col'
#obsmod = 'both'
def log_image(star_name, obsmod):             
#%%        
    
    ##Parameters
    nDim = 1024
    nSubDim = 50 # plage de pixels que l'on veut afficher
    size = (nSubDim, nSubDim)
    # nDimfigj = [3, 4, 5]
    # nDimfigk = [6, 7, 8]
    vmin0 = 3.5
    vmax0 = 15
    pix2mas = 3.4  #en mas/pix
    x_min = -pix2mas*nSubDim//2
    x_max = pix2mas*(nSubDim//2-1)
    y_min = -pix2mas*nSubDim//2
    y_max = pix2mas*(nSubDim//2-1)
    X, Y= np.meshgrid(np.linspace(-nSubDim/2,nSubDim/2-1,nSubDim), np.linspace(-nSubDim/2,nSubDim/2-1,nSubDim))
    #X_, Y_= np.meshgrid(np.linspace(-nDim/2,nDim/2-1,nDim), np.linspace(-nDim/2,nDim/2-1,nDim))
    
    X *= pix2mas
    Y *= pix2mas
    # X_ *= pix2mas
    # Y_ *= pix2mas
    
    X_step = 3
    X_step_ = 50
    
    position = (nDim//2,nDim//2)
    size = (nSubDim, nSubDim)
    
    x, y = np.meshgrid(np.arange(nSubDim), np.arange(nSubDim)) #cree un tableau 
    
    R = np.sqrt((x-nSubDim/2)**2+(y-nSubDim/2)**2)
    r = np.linspace(1,nSubDim//2-1,nSubDim//2-1) # creation d'un tableau de distance radiale    
    r_mas = pix2mas*r #  où r est en pixels et r_mas en millièmes d'arcsecondes
    
    nDimfigj = [3, 4, 5]
    nDimfigk = [6, 7, 8]
    
    fdir= '/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+ '/'
    fdir_star = fdir + 'star/'+obsmod+ '/' 
    fdir_psf = fdir +'psf/'+obsmod+ '/'
    
    lst_fltr_star = os.listdir(fdir_star)
    n_lst_fltr_star = len(lst_fltr_star)
    lst_fltr2_star = []
     
    
    #Recherche des filtres contenant des données   
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
        file_AOLP_star= fdir_star_fltr + fname1+'_AOLP'+fname2+'_AOLP.fits'
        file_Q_star= fdir_star_fltr + fname1+'_Q'+fname2+'_Q.fits'
        file_U_star= fdir_star_fltr + fname1+'_U'+fname2+'_U.fits'
        
    
        file_I_psf = fdir_psf_fltr + fname1+'_I'+fname2+'_I.fits'
        file_PI_psf = fdir_psf_fltr +fname1+'_PI'+fname2+'_PI.fits'
        file_DOLP_psf = fdir_psf_fltr + fname1+'_DOLP'+fname2+'_DOLP.fits'
        file_AOLP_psf = fdir_psf_fltr +fname1+'_AOLP'+fname2+'_AOLP.fits'
        file_Q_psf = fdir_psf_fltr + fname1+'_Q'+fname2+'_Q.fits'
        file_U_psf = fdir_psf_fltr + fname1+'_U'+fname2+'_U.fits'
      
        file_lst = [file_I_star,file_PI_star,file_DOLP_star,file_AOLP_star, file_Q_star,file_U_star,
                  file_I_psf,file_PI_psf,file_DOLP_psf,file_AOLP_psf,file_Q_psf, file_U_psf]
        
        file_lst2 = [file_I_star, file_PI_star, file_DOLP_star, file_AOLP_star, file_Q_star, file_U_star]
        file_lst3 = [file_I_psf, file_PI_psf, file_DOLP_psf, file_AOLP_psf, file_Q_psf, file_U_psf]          
        
        nFrames = len(file_lst)
        nFrames2 = len(file_lst2)
        nFrames3 = len(file_lst3)
        
        mean_sub_v_arr2 = np.empty((nFrames2,nSubDim//2-1))
        mean_sub_v_arr3 = np.empty((nFrames3,nSubDim//2-1))
        sub_v_arr2 = np.empty((nFrames2,nSubDim,nSubDim))
        sub_v_arr3 = np.empty((nFrames3,nSubDim,nSubDim))
        im_name_lst = ['I','PI','DOLP','AOLP',
                        'I','PI','DOLP','AOLP']
        Vmin2 = np.empty((nFrames2))
        Vmax2 = np.empty((nFrames2))
        
        Vmin3 = np.empty((nFrames3))
        Vmax3 = np.empty((nFrames3))
        fsize = [0,1]       
        n_fsize = len (fsize)
        fltr_arr = np.empty(n_fsize, dtype = str)
        for z in range(n_fsize) :
            
            for i in range (nFrames2):
                  hdu = fits.open(file_lst2[i])[0]   
                  data2 = hdu.data   
                  i_v2 = data2[z,:,:]
                  fltr1 = hdu.header.get('HIERARCH ESO INS3 OPTI5 NAME')   
                  fltr2 = hdu.header.get('HIERARCH ESO INS3 OPTI6 NAME')
                  fltr_arr[0] = fltr1
                  fltr_arr[1] = fltr2
                  #print(fltr)                   
                  cutout2 = Cutout2D(i_v2, position=position, size=size)
                  zoom_hdu = hdu.copy()
                  sub_v2 = cutout2.data
                
                  f2 = lambda r : sub_v2[(R >= r-0.5) & (R < r+0.5)].mean()   
                  mean_sub_v2 = np.vectorize(f2)(r) 
                
                  mean_sub_v_arr2[i] = mean_sub_v2 
                  sub_v_arr2[i] = sub_v2
                      
                  DOLP_star = sub_v_arr2[2]
                  ii = (sub_v_arr2[4] == 0)
                  if True in ii:
                      sub_v_arr2[4][ii] = sub_v_arr2[4][ii] + 0.0001  # introduction d'un ofset pour les valeurs de Q == 0
                        
                  AOLP_2_star = 0.5*np.arctan2(sub_v_arr2[5], sub_v_arr2[4])
                  U2 = DOLP_star*np.cos(-(AOLP_2_star + np.pi/2))
                  V2 = DOLP_star*np.sin(-(AOLP_2_star + np.pi/2))
                  
            for i in range (nFrames3):
                  hdu3 = fits.open(file_lst3[i])   
                  data3 = hdu3[0].data   
                  i_v3 = data3[z,:,:]
               
                  cutout3 = Cutout2D(i_v3, position=position, size=size)
                  zoom_hdu3 = hdu3.copy()
                  sub_v3 = cutout3.data
                
                  f3 = lambda r : sub_v3[(R >= r-0.5) & (R < r+0.5)].mean()   
                  mean_sub_v3 = np.vectorize(f3)(r) 
                  
                  mean_sub_v_arr3[i] = mean_sub_v3 
                  sub_v_arr3[i] = sub_v3
                
                  DOLP_psf = sub_v_arr3[2]
                  ii = (sub_v_arr3[4] == 0)
                  if True in ii:
                     sub_v_arr3[4][ii] = sub_v_arr3[4][ii] + 0.0001  # introduction d'un ofset pour les valeurs de Q == 0
                        
                  AOLP_2_psf = 0.5*np.arctan2(sub_v_arr3[5], sub_v_arr3[4])
                  U3 = DOLP_psf*np.cos(-(AOLP_2_psf + np.pi/2))
                  V3 = DOLP_psf*np.sin(-(AOLP_2_psf + np.pi/2))
                
            
            #plt.figure(f'{star_name}' +'(' + f'{fltr}' + '_Cam1' + ')' , figsize=(18.5,10.5))
            
            # linear scale
            plt.clf()
            fig = plt.figure(f'{star_name}' +'(' + f'{fltr_arr[z]}' + '_lin_Cam1' + ')')
            fig.set_size_inches(18.5, 10, forward = True)
            for t in range (nFrames2):
                  plt.subplot(3,3, t+1)
                  #if i != 2 and i!= 3:
                  if t < 2:
                      
                      plt.imshow(sub_v_arr2[t], cmap='inferno', origin='lower',
                          vmin= np.min(sub_v_arr2[t]), vmax= np.max(sub_v_arr2[t]), extent = [x_min , x_max, y_min , y_max])
                          
                      plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                                    f'{star_name}' + '_' + f'{im_name_lst[t]}' + ' in '+ f'{fltr_arr[z]}' + ' band', color='w',
                                fontsize='large', ha='center')
                      plt.colorbar(label='ADU', shrink = 0.6)
                
                  elif t == 2:
                      plt.imshow(sub_v_arr2[1], cmap ='inferno', origin='lower',vmin= np.min(sub_v_arr2[1]), 
                                  vmax=np.max(sub_v_arr2[1]), extent = [x_min , x_max, y_min , y_max])   
                      plt.colorbar(label='ADU', shrink = 0.6)       
                      q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2[::X_step,::X_step], V2[::X_step,::X_step], color='w', pivot='mid')
                      plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')                 
                      plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                              'Pol. vect.' + ' in '+ f'{fltr_arr[z]}' + ' band', color='y',
                                  fontsize='large', ha='center')
    
                          
                  if t == 0:
                      plt.ylabel('Relative Dec.(mas)', size=10)   
                              
            for j in range (len(nDimfigj)): 
                plt.subplot(3,3,(nDimfigj[j] + 1))
                if j < 2 :
                    plt.imshow(sub_v_arr3[j], cmap='inferno', origin='lower',
                        vmin= np.min(sub_v_arr3[j]), vmax= np.max(sub_v_arr3[j]), extent = [x_min , x_max, y_min , y_max])
                        
                    plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                                  f'{star_name}' + '_psf_' + f'{im_name_lst[j]}' + ' in '+ f'{fltr_arr[z]}' + ' band', color='w',
                              fontsize='large', ha='center')
                    plt.colorbar(label='ADU', shrink = 0.6)
                
                elif j == 2:
                    plt.imshow(sub_v_arr3[1], cmap ='inferno', origin='lower',vmin= np.min(sub_v_arr3[1]), 
                                vmax=np.max(sub_v_arr3[1]), extent = [x_min , x_max, y_min , y_max])   
                    plt.colorbar(label='ADU', shrink = 0.6)       
                    q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U3[::X_step,::X_step], V3[::X_step,::X_step], color='w', pivot='mid')
                    plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')                 
                    plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                            'Pol. vect.' + ' in '+ f'{fltr_arr[z]}' + ' band', color='y',
                                fontsize='large', ha='center')
                    
                if j == 0:
                    plt.ylabel('Relative Dec.(mas)', size=10)
                    plt.xlabel('Relative R.A.(mas)', size=10)
                else:
                    plt.xlabel('Relative R.A.(mas)', size=10)  
                    
            for k in range(len(nDimfigk)):      
                plt.subplot(3,3,nDimfigk[k] + 1)
                plt.plot(r_mas, mean_sub_v_arr2[k], color='darkorange',
                linewidth = 2, label= f'{star_name}') 
                plt.plot(r_mas, mean_sub_v_arr3[k],color='purple',
                    linewidth = 2, label = f'{star_name}' + '_psf') 
                plt.legend(loc=0) 
                plt.xlabel('r (mas)', size=10) 
                
                if k == 0:
                       plt.ylabel(r'Intensity', size=10)
                
            plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+
                              '/plots/star_psf/'+ star_name +'_' +  f'{fltr_arr[z]}' + '_lin' + '.pdf', 
                              dpi=100, bbox_inches ='tight')
            
            plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+
                              '/plots/star_psf/'+ star_name +'_' +  f'{fltr_arr[z]}'  + '_lin' + '.png', 
                              dpi=100, bbox_inches ='tight')         
            
            plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/star_psf/linear_scale/'+star_name+
                 '_' +  f'{fltr_arr[z]}' + '_lin' + '.pdf', dpi=100, bbox_inches ='tight')
            
            plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/star_psf/linear_scale/'+star_name+
                              '_' +  f'{fltr_arr[z]}' + '_lin' + '.png', dpi=100, bbox_inches ='tight')
            
            plt.tight_layout()      
                
                
                
            # log scale
            plt.clf()
            fig = plt.figure(f'{star_name}' +'(' +  f'{fltr_arr[z]}' + '_log_Cam1' + ')')
            fig.set_size_inches(18.5, 10, forward = True)
            for o in range (nFrames2 -2):
                plt.subplot(3,3, o+1)
                      #if i != 2 and i!= 3:
                if o < 2:
                    im_star = np.log10(sub_v_arr2[o] + np.abs(np.min(sub_v_arr2[o]) + 10)) # add an ofset because of negatives values         
                    
                    plt.imshow(im_star, cmap='inferno', origin='lower',
                          vmin = np.min(im_star), vmax= np.max(im_star), extent = [x_min , x_max, y_min , y_max])
                         
                    plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                                    f'{star_name}' + '_' + f'{im_name_lst[o]}' + ' in '+  f'{fltr_arr[z]}' + ' band', color='w',
                                fontsize='large', ha='center')
                    plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
        
        
                elif o == 2 :
                    im_star2 = np.log10(sub_v_arr2[1] + np.abs(np.min(sub_v_arr2[1]) + 10))
                    plt.imshow(im_star2, vmin = np.min(im_star2), vmax= np.max(im_star2), cmap ='inferno', origin='lower', 
                                      extent = [x_min , x_max, y_min , y_max])   
                    plt.colorbar(label = 'ADU in log$_{10}$ scale', shrink = 0.6)       
                    q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2[::X_step,::X_step], V2[::X_step,::X_step], color='w', pivot='mid')
                    plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')                       
                    plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                      'Pol. vect.' + ' in '+  f'{fltr_arr[z]}' + ' band', color='y',
                                fontsize='large', ha='center')        
        
        
        
                if o == 0:
                    plt.ylabel('Relative Dec.(mas)', size=10)   
        
        
                for p in range (len(nDimfigj)): 
                    plt.subplot(3,3,(nDimfigj[p] + 1))
                    if p < 2 :
                        im_psf = np.log10(sub_v_arr3[p] + np.abs(np.min(sub_v_arr3[p]) + 10)) # add an ofset because of negatives values         
                        
                        plt.imshow(im_psf, cmap='inferno', origin='lower',
                              vmin = np.min(im_psf), vmax= np.max(im_psf), extent = [x_min , x_max, y_min , y_max])
                              
                        plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                                       f'{star_name}' + '_psf_'  + f'{im_name_lst[p]}' + ' in '+  f'{fltr_arr[z]}' + ' band', color='w',
                                    fontsize='large', ha='center')
                        plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
    
                    elif p == 2 :
                        im_psf2 = np.log10(sub_v_arr3[1] + np.abs(np.min(sub_v_arr3[1]) + 10)) # add an ofset because of negatives values         
                        
                        plt.imshow(im_psf2, vmin = np.min(im_psf2), vmax= np.max(im_psf2), cmap ='inferno', origin='lower', 
                                          extent = [x_min , x_max, y_min , y_max])   
                        plt.colorbar(label = 'ADU in log$_{10}$ scale', shrink = 0.6)       
                        q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2[::X_step,::X_step], V2[::X_step,::X_step], color='w', pivot='mid')
                        plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')                       
                        plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                          'Pol. vect.' + ' in '+  f'{fltr_arr[z]}' + ' band', color='y',
                                    fontsize='large', ha='center')
            
                    if p == 0:
                        plt.ylabel('Relative Dec.(mas)', size=10)
                        plt.xlabel('Relative R.A.(mas)', size=10)
                    else:
                        plt.xlabel('Relative R.A.(mas)', size=10)  
            
            for q in range(len(nDimfigk)):      
                plt.subplot(3,3,nDimfigk[q] + 1)
                plt.plot(np.log10(r_mas), np.log10(mean_sub_v_arr2[q]), color='darkorange',
                linewidth = 2, label= f'{star_name}') 
                plt.plot(np.log10(r_mas), np.log10(mean_sub_v_arr3[q]),color='purple',
                    linewidth = 2, label = f'{star_name}' + '_psf') 
                plt.legend(loc=0) 
                plt.xlabel('log10[r (mas)]', size=10) 
                
                if q == 0:
                       plt.ylabel(r'Intensity in log$_{10}$ scale', size=10)
                
        
            plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+ star_name +
                                  '/plots/star_psf/'+ star_name +'_' +  f'{fltr_arr[z]}' + '_log' + '.pdf', 
                                  dpi=100, bbox_inches ='tight')
                
            plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+
                                  '/plots/star_psf/'+ star_name +'_' +  f'{fltr_arr[z]}' + '_log' + '.png', 
                                  dpi=100, bbox_inches ='tight')
                
            plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/star_psf/log_scale/'+ 
                            star_name +'_' +  f'{fltr_arr[z]}'  + '_log' + '.pdf', 
                                  dpi=100, bbox_inches ='tight')
            plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/star_psf/log_scale/'+ 
                            star_name +'_' +  f'{fltr_arr[z]}' + '_log' + '.png', 
                                  dpi=100, bbox_inches ='tight')    
        
            plt.tight_layout()
    
    #print('Au total, '+  str(n_lst_fltr3) +' étoiles ont une psf')    
    return()
        
#log_image('SW_Col', 'both')    
