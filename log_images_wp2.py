#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:19:25 2023

@author: nbadolo
"""

"""
Code simplifié pour les carte de couleurs des étoiles du deuxieme lot. Sans prendre en compte
leur psf:  
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


# Parameters 
nDim = 1024
nSubDim = 200 # plage de pixels que l'on veut afficher
size = (nSubDim, nSubDim)

pix2mas = 3.4  #en mas/pix
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
#%% 
def log_image(star_name, obsmod):
#%%       
    fdir= '/home/nbadolo/Bureau/Aymard/Donnees_sph/log/News_stars/'+star_name+ '/'
    fdir_star = fdir + 'star/'+obsmod+ '/' 
    fdir_psf = fdir +'psf/' +obsmod+ '/'
    lst_fltr_star1 = os.listdir(fdir_star)
    #print(lst_fltr_star1)
    n_lst_fltr_star1 = len(lst_fltr_star1)
    #print(n_lst_fltr_star1)
    lst_fltr_star2 = []
    nDimfigj = [3, 4, 5]
    nDimfigk = [6, 7, 8]
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
        
        file_lst2 = [file_I_star, file_PI_star, file_DOLP_star, file_AOLP_star]
        nFrames2 = len(file_lst2)
    
        mean_sub_v_arr2 = np.empty((nFrames2,nSubDim//2-1))
        sub_v_arr2 = np.empty((nFrames2,nSubDim,nSubDim))
        im_name_lst = ['I','PI','DOLP', 'AOLP']
        Vmin2 = np.empty((nFrames2))
        Vmax2 = np.empty((nFrames2))
        
        position = (nDim//2,nDim//2)
        size = (nSubDim, nSubDim)
        
        x, y = np.meshgrid(np.arange(nSubDim), np.arange(nSubDim)) #cree un tableau 
        
        R = np.sqrt((x-nSubDim/2)**2+(y-nSubDim/2)**2)
        r = np.linspace(1,nSubDim//2-1,nSubDim//2-1) # creation d'un tableau de distance radiale
        
        r_mas = pix2mas*r #  où r est en pixels et r_mas en millièmes d'arcseconde
        
        for i in range (nFrames2):
              hdu = fits.open(file_lst2[i])[0]   
              data2 = hdu.data   
              i_v2 = data2[0,:,:]
              fltr = hdu.header.get('HIERARCH ESO INS3 OPTI5 NAME')     
              #print(fltr)                   
              cutout2 = Cutout2D(i_v2, position=position, size=size)
              zoom_hdu = hdu.copy()
              sub_v2 = cutout2.data
            
              f = lambda r : sub_v2[(R >= r-0.5) & (R < r+0.5)].mean()   
              mean_sub_v = np.vectorize(f)(r) 
            
              mean_sub_v_arr2[i] = mean_sub_v 
              sub_v_arr2[i] = sub_v2
              if np.any(np.min(sub_v_arr2[i])<= 0): #i==3 or i==7:
                  Vmin2[i] = np.min(sub_v_arr2[i])
                  Vmax2[i] = np.max(sub_v_arr2[i])  
              else:
                  Vmin2[i] = np.min(np.log10(sub_v_arr2[i]))
                  Vmax2[i] = np.max(np.log10(sub_v_arr2[i]))  
         
              U2 = sub_v_arr2[2]*np.cos(np.pi*sub_v_arr2[3]/180)
              V2 = sub_v_arr2[2]*np.sin(np.pi*sub_v_arr2[3]/180)
              
        plt.clf()
        fig = plt.figure(f'{star_name}' +'(' + f'{fltr}' + '_Cam1' + ')')
        fig.set_size_inches(12, 6, forward = True)
        for i in range (nFrames2):
              plt.subplot(2,2,i+1)
              if  i!= 3:
                  if np.any(np.min(sub_v_arr2[i])<= 0):           
                      plt.imshow(sub_v_arr2[i], cmap='inferno', origin='lower',
                      vmin=Vmin2[i], vmax=Vmax2[i], extent = [x_min , x_max, y_min , y_max])
                      
                      plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                                f'{star_name}' + '_' + f'{im_name_lst[i]}', color='w',
                            fontsize='large', ha='center')
                      plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
                  else:
                        plt.imshow(np.log10(sub_v_arr2[i]), cmap='inferno', origin='lower',
                        vmin=Vmin2[i], vmax=Vmax2[i], extent = [x_min , x_max, y_min , y_max])
                       
                        plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                                  f'{star_name}' + '_' + f'{im_name_lst[i]}', color='w',
                              fontsize='large', ha='center')
                        plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
              else :
                  
                  if np.any(np.min(sub_v_arr2[1])<= 0):
                    
                      plt.imshow(sub_v_arr2[1], cmap ='inferno', origin='lower',vmin=Vmin2[1], 
                                  vmax=Vmax2[1], extent = [x_min , x_max, y_min , y_max])   
                      plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)       
                      q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2[::X_step,::X_step], V2[::X_step,::X_step])
                      plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')                 
                      plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                              f'{im_name_lst[1]}'+ '_&_Pol. vect', color='w',
                                  fontsize='large', ha='center')
              
                  else :
                    plt.imshow(np.log10(sub_v_arr2[1]), cmap ='inferno', origin='lower',vmin=Vmin2[1], 
                                    vmax=Vmax2[1], extent = [x_min , x_max, y_min , y_max])   
                    plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)       
                    q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2[::X_step,::X_step], V2[::X_step,::X_step])
                    plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')                       
                    plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                      f'{im_name_lst[1]}'+ '_&_Pol. vect', color='w',
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
                  
        plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/News_stars/'+star_name+
                          '/plots/'+ star_name +'_' + fltr + '_Cam1' + '.pdf', 
                          dpi=100, bbox_inches ='tight')
        
        
        plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/News_stars/'+star_name+
                          '/plots/'+ star_name +'_' + fltr + '_Cam1' + '.png', 
                          dpi=100, bbox_inches ='tight')
        plt.tight_layout()
        msg1='image display ended for '+ star_name+'_Cam1'
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
         
         file_lst2_ = [file_I_star, file_PI_star, file_DOLP_star, file_AOLP_star]
         nFrames2_ = len(file_lst2_)
        
         
         mean_sub_v_arr2_ = np.empty((nFrames2_,nSubDim//2-1))
         sub_v_arr2_ = np.empty((nFrames2_,nSubDim,nSubDim))
         im_name_lst = ['I','PI','DOLP']
         Vmin2_ = np.empty((nFrames2_))
         Vmax2_ = np.empty((nFrames2_))
        
         position = (nDim//2,nDim//2)
         size = (nSubDim, nSubDim)
        
         x, y = np.meshgrid(np.arange(nSubDim), np.arange(nSubDim)) #cree un tableau 
        
         R = np.sqrt((x-nSubDim/2)**2+(y-nSubDim/2)**2)
         r = np.linspace(1,nSubDim//2-1,nSubDim//2-1)
        
         r_mas=pix2mas*r #  où r est en pixels et r_mas en millièmes d'arcseconde
         
         
         for i in range (nFrames2_):
               hdu_ = fits.open(file_lst2_[i])[0]   
               data2_ = hdu_.data   
               i_v2_ = data2_[1,:,:]
               fltr_ = hdu_.header.get('HIERARCH ESO INS3 OPTI6 NAME')
               
               cutout2_ = Cutout2D(i_v2_, position=position, size=size)
               zoom_hdu = hdu_.copy()
               sub_v2_ = cutout2_.data
             
               f = lambda r : sub_v2_[(R >= r-0.5) & (R < r+0.5)].mean()   
               mean_sub_v_ = np.vectorize(f)(r) 
             
               mean_sub_v_arr2_[i] = mean_sub_v_ 
               sub_v_arr2_[i] = sub_v2_
               if np.any(np.min(sub_v_arr2_[i])<= 0): #i==3 or i==7:
                   Vmin2_[i] = np.min(sub_v_arr2_[i])
                   Vmax2_[i] = np.max(sub_v_arr2_[i])  
               else:
                   Vmin2_[i] = np.min(np.log10(sub_v_arr2_[i]))
                   Vmax2_[i] = np.max(np.log10(sub_v_arr2_[i]))  
          
               U2_ = sub_v_arr2_[2]*np.cos(np.pi*sub_v_arr2_[3]/180)
               V2_ = sub_v_arr2_[2]*np.sin(np.pi*sub_v_arr2_[3]/180)
               
         plt.clf()
         fig = plt.figure(f'{star_name}' +'(' + f'{fltr_}' + '_Cam2' + ')')
         fig.set_size_inches(12,6, forward = True)
         for i in range (nFrames2_):
               plt.subplot(2,2,i+1)
               if i!= 3:
                   if np.any(np.min(sub_v_arr2_[i])<= 0):           
                       plt.imshow(sub_v_arr2_[i], cmap='inferno', origin='lower',
                       vmin=Vmin2_[i], vmax=Vmax2_[i], extent = [x_min , x_max, y_min , y_max])
                       
                       plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                                 f'{star_name}' + '_' + f'{im_name_lst[i]}', color='w',
                             fontsize='large', ha='center')
                       plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
                   else:
                        plt.imshow(np.log10(sub_v_arr2_[i]), cmap='inferno', origin='lower',
                        vmin=Vmin2_[i], vmax=Vmax2_[i], extent = [x_min , x_max, y_min , y_max])
                        
                        plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                                  f'{star_name}' + '_' + f'{im_name_lst[i]}', color='w',
                              fontsize='large', ha='center')
                        plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
               else :
                   
                             
                   if np.any(np.min(sub_v_arr2_[1])<= 0):
                        plt.imshow(sub_v_arr2_[1], cmap ='inferno', origin='lower',vmin=Vmin2_[1], 
                                      vmax= Vmax2_[1], extent = [x_min , x_max, y_min , y_max])   
                        plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)       
                        q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2_[::X_step,::X_step], V2_[::X_step,::X_step])
                        plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')
                        plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                          f'{im_name_lst[1]}'+ '_&_Pol. vect', color='w',
                             fontsize='large', ha='center')
                   else :
                        plt.imshow(np.log10(sub_v_arr2_[1]), cmap ='inferno', origin='lower',vmin=Vmin2_[1], 
                                   vmax=Vmax2_[1], extent = [x_min , x_max, y_min , y_max])   
                        plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)       
                        q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2_[::X_step,::X_step], V2_[::X_step,::X_step])
                        plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')
                        plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                      f'{im_name_lst[1]}'+ '_&_Pol. vect', color='w',
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
            
         plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/News_stars/'+star_name+
                            '/plots/'+star_name+'_' + fltr_ +'_Cam2' + '.pdf', 
                            dpi=100, bbox_inches ='tight')
           
           
         plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/News_stars/'+star_name+
                            '/plots/'+star_name+'_' + fltr_ +'_Cam2' + '.png', 
                            dpi=100, bbox_inches ='tight')
         plt.tight_layout()
       
    msg2='image display ended for '+ star_name +'_Cam2'
    print(msg2)
    msg= 'image display ended for ' + star_name
    return(msg)

#log_image('SW_Col', 'both')