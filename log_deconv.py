#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:59:06 2022

@author: nbadolo
"""

"""
Code de déconvolution pour les etoiles mises a part, c'est à dire les objets resolu. Elles doivent avoir une psf chacune. 
"""

#Packages
import numpy as np
import os
from astropy.io import fits
from scipy import optimize, signal
from scipy.signal import convolve2d as conv2
from AymardPack import Margaux_RL_deconv
from skimage import color, data, restoration
from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt
import math
import matplotlib.colors as colors
from matplotlib.pyplot import Figure, subplot


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
        file_AOLP_star= fdir_star_fltr + fname1+'_AOLP'+fname2+'_AOLP.fits'
    
        file_I_psf = fdir_psf_fltr + fname1+'_I'+fname2+'_I.fits'
        file_PI_psf = fdir_psf_fltr +fname1+'_PI'+fname2+'_PI.fits'
        file_DOLP_psf = fdir_psf_fltr + fname1+'_DOLP'+fname2+'_DOLP.fits'
        file_AOLP_psf = fdir_psf_fltr +fname1+'_AOLP'+fname2+'_AOLP.fits'
      
        file_lst = [file_I_star,file_PI_star,file_DOLP_star,file_AOLP_star,
                  file_I_psf,file_PI_psf,file_DOLP_psf,file_AOLP_psf]
        
        file_lst2 = [file_I_star, file_PI_star, file_DOLP_star, file_AOLP_star]
        file_lst3 = [file_I_psf, file_PI_psf, file_DOLP_psf, file_AOLP_psf]          
        
        nFrames = len(file_lst)
        nFrames2 = len(file_lst2)
        nFrames3 = len(file_lst3)
        nFrames_d = nFrames2
    
        mean_sub_v_arr2 = np.empty((nFrames2,nSubDim//2-1))
        mean_sub_v_arr3 = np.empty((nFrames3,nSubDim//2-1))
        sub_v_arr2 = np.empty((nFrames2,nSubDim,nSubDim))
        sub_v_arr3 = np.empty((nFrames3,nSubDim,nSubDim))
        deconv_arr = np.empty((nFrames_d,nSubDim,nSubDim))
        deconv_arr2 = np.empty((nFrames_d,nSubDim,nSubDim))
        
        
        im_name_lst = ['I','PI','DOLP','AOLP',
                        'I','PI','DOLP','AOLP']
        
        Vmin2 = np.empty((nFrames2))
        Vmax2 = np.empty((nFrames2))
        
        Vmin3 = np.empty((nFrames3))
        Vmax3 = np.empty((nFrames3))
        
        Vmin_d = np.empty((nFrames_d))
        Vmax_d = np.empty((nFrames_d))
        
        Vmin_d2 = np.empty((nFrames_d))
        Vmax_d2 = np.empty((nFrames_d))
    
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
              
              
              
        for i in range (nFrames3):
              hdu3 = fits.open(file_lst3[i])   
              data3 = hdu3[0].data   
              i_v3 = data3[0,:,:]
           
              cutout3 = Cutout2D(i_v3, position=position, size=size)
              zoom_hdu3 = hdu3.copy()
              sub_v3 = cutout3.data
            
              f = lambda r : sub_v3[(R >= r-0.5) & (R < r+0.5)].mean()   
              mean_sub_v3 = np.vectorize(f)(r) 
            
              mean_sub_v_arr3[i] = mean_sub_v3
              sub_v_arr3[i] = sub_v3
              if np.any(np.min(sub_v_arr3[i])<= 0): #i==3 or i==7:
                  Vmin3[i] = np.min(sub_v_arr3[i])
                  Vmax3[i] = np.max(sub_v_arr3[i])  
              else:
                  Vmin3[i] = np.min(np.log10(sub_v_arr3[i]))
                  Vmax3[i] = np.max(np.log10(sub_v_arr3[i]))  
         
              U3 = sub_v_arr3[2]*np.cos(np.pi*sub_v_arr3[3]/180)
              V3 = sub_v_arr3[2]*np.sin(np.pi*sub_v_arr3[3]/180)
        mean_sub_v_arr =  mean_sub_v_arr2 + mean_sub_v_arr3   
        sub_v_arr = sub_v_arr2 + sub_v_arr3     
        # Vmin = Vmin2 + Vmin3
        # Vmax = Vmax2 + Vmax3
        
        for i in range (nFrames_d):   
            deconvolved_RL = Margaux_RL_deconv(sub_v_arr2[i], sub_v_arr3[0], 5) # deconvolution de toutes les cartes par l'int de la psf
            deconv_arr[i]  = deconvolved_RL
                        
            # deconvolved_RL2 = arl.Margaux_RL_deconv(sub_v_arr2[i], sub_v_arr3[i], 15) #  deconvolution int par int, int pol par int pol 
            # deconv_arr2[i]  = deconvolved_RL2
            
            if np.any(np.min(deconv_arr[i])<= 0): #i==3 or i==7:
                Vmin_d[i] = np.min(deconv_arr[i])
                Vmax_d[i] = np.max(deconv_arr[i])  
            else:
                Vmin_d[i] = np.min(np.log10(deconv_arr[i]))
                Vmax_d[i] = np.max(np.log10(deconv_arr[i]))  
                
            if np.any(np.min(deconv_arr2[i])<= 0): #i==3 or i==7:
                Vmin_d2[i] = np.min(deconv_arr2[i])
                Vmax_d2[i] = np.max(deconv_arr2[i])  
            else:
                Vmin_d2[i] = np.min(np.log10(deconv_arr2[i]))
                Vmax_d2[i] = np.max(np.log10(deconv_arr2[i]))
            
            print('deconvolution okay for ' + ' ' f'{star_name}' + ' '+ f'{fltr}'+ ' '+ f'{im_name_lst[i]}' )
            
        #plt.figure(f'{star_name}' +'(' + f'{fltr}' + '_Cam1' + ')' , figsize=(18.5,10.5))     
        plt.clf()
        fig = plt.figure('deconv_' + f'{star_name}' +'(' + f'{fltr}' + '_Cam1' + ')')
        fig.set_size_inches(18.5, 10, forward = True)
        for i in range (nFrames2):
              plt.subplot(3,3,i+1)
              if i != 2 and i!= 3:      # i!=3 car si non on aurait 4 fifures à afficher dans 3 cadres dispo
                  if np.any(np.min(deconv_arr[i])<= 0):           
                      plt.imshow(deconv_arr[i], cmap='inferno', origin='lower',
                      vmin=Vmin_d[i], vmax=Vmax_d[i], extent = [x_min , x_max, y_min , y_max])
                      
                      plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                              'deconv_'+f'{star_name}' + '_' + f'{im_name_lst[i]}', color='w',
                            fontsize='large', ha='center')
                      plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
                  else:
                        plt.imshow(np.log10(deconv_arr[i]), cmap='inferno', origin='lower',
                        vmin=Vmin_d[i], vmax = Vmax_d[i], extent = [x_min , x_max, y_min , y_max])
                       
                        plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                                 'deconv_' + f'{star_name}' + '_' + f'{im_name_lst[i]}', color='w',
                              fontsize='large', ha='center')
                        plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
              else :
                  if i == 2:                  
                      if np.any(np.min(deconv_arr[1])<= 0):
                          plt.imshow(deconv_arr[1], cmap ='inferno', origin='lower',vmin=Vmin_d[1], 
                                        vmax = Vmax_d[1], extent = [x_min , x_max, y_min , y_max])   
                          plt.colorbar(label='ADU in log$_{10}$ scale')       
                          q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2[::X_step,::X_step], V2[::X_step,::X_step])
                          plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')                 
                          plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                                   'deconv_'+ f'{im_name_lst[1]}'+ '_&_Pol. vect', color='w',
                                        fontsize='large', ha='center')
                    
                      else :
                          plt.imshow(np.log10(deconv_arr[1]), cmap ='inferno', origin='lower',vmin = Vmin_d[1], 
                                          vmax = Vmax_d[1], extent = [x_min , x_max, y_min , y_max])   
                          plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)       
                          q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2[::X_step,::X_step], V2[::X_step,::X_step])
                          plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')                       
                          plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                            'deconv_'+ f'{im_name_lst[1]}'+ '_&_Pol. vect', color='w',
                                fontsize='large', ha='center')
        
              if i == 0:
                  plt.ylabel('Relative Dec.(mas)', size=10)   
                          
        for j in range (len(nDimfigj)): 
            plt.subplot(3,3,(nDimfigj[j] + 1))
            if j == 2:
                print(nDimfigj[j])
                if np.any(np.min(sub_v_arr2[1])<= 0):
                    plt.imshow(sub_v_arr2[1], cmap ='inferno', origin='lower',vmin = Vmin2[1], 
                                        vmax=Vmax2[1], extent = [x_min , x_max, y_min , y_max])   
                    plt.colorbar(label='ADU in log$_{10}$ scale')       
                    q_ = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2[::X_step,::X_step], V2[::X_step,::X_step])
                    plt.quiverkey(q_, X = 0.1, Y = 1.03, U = 0.01, label='deg vect. n. scale 0.03', labelpos='E')
                    plt.text(size[0]//10, 2*pix2mas*size[1]//6., 
                                f'{star_name}' + '_' + f'{im_name_lst[1]}'+ '_&_Pol. vect', color='w',
                                        fontsize='large', ha='center')
                else :
                    plt.imshow(np.log10(sub_v_arr2[1]), cmap ='inferno', origin='lower',vmin=Vmin2[1], 
                                          vmax=Vmax2[1], extent = [x_min , x_max, y_min , y_max])   
                    plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)       
                    q_ = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2[::X_step,::X_step], V2[::X_step,::X_step])
                    plt.quiverkey(q_, X = 0.1, Y = 1.03, U = 0.01, label='deg vect. n. scale 0.03', labelpos='E')
                    plt.text(size[0]//10, 2*pix2mas*size[1]//6., 
                          f'{star_name}' + '_'+ f'{im_name_lst[1]}'+ '_&_Pol. vect', color='w',
                                  fontsize='large', ha='center')
            else:
                if np.any(np.min(sub_v_arr2[j])<= 0):           
                    plt.imshow(sub_v_arr2[j], cmap='inferno', origin='lower',
                    vmin = Vmin2[j], vmax = Vmax2[j], extent = [x_min , x_max, y_min , y_max])
                      
                    plt.text(size[0]//10, 2*pix2mas*size[1]//6., 
                                f'{star_name}' + '_' + f'{im_name_lst[j]}', color='w',
                              fontsize='large', ha='center')
                    plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
                else:
                    plt.imshow(np.log10(sub_v_arr2[j]), cmap='inferno', origin='lower',
                    vmin = Vmin2[j], vmax = Vmax2[j], extent = [x_min , x_max, y_min , y_max])
                       
                    plt.text(size[0]//10, 2*pix2mas*size[1]//6., 
                                 f'{star_name}' + '_' + f'{im_name_lst[j]}', color='w',
                                fontsize='large', ha='center')
                    plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
                    
            if j == 0:
                plt.ylabel('Relative Dec.(mas)', size=10)
                plt.xlabel('Relative R.A.(mas)', size=10)
            else:
                plt.xlabel('Relative R.A.(mas)', size=10)         
                           
        for k in range(len(nDimfigk)):      
              plt.subplot(3,3,nDimfigk[k] + 1)
              plt.plot(r_mas, np.log10(mean_sub_v_arr2[k]), color='darkorange',
                      linewidth = 2, label= f'{star_name}' +'_' + f'{im_name_lst[k]}' + '_profile') 
              plt.plot(r_mas, np.log10(mean_sub_v_arr3[k]),color='purple',
                      linewidth = 2, label = f'{star_name}' + '_' + f'{im_name_lst[k]}' + '_profile' + '_psf') 
              plt.legend(loc=0) 
              plt.xlabel('r (mas)', size=10) 
              if k == 0:
                  plt.ylabel(r'Intensity in log$_{10}$ scale', size=10)
        
        plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/'+star_name+
                        '/plots/' +'deconvoluted_' + star_name+'_' + fltr + '_Cam1' + '.pdf', 
                        dpi=100, bbox_inches ='tight')
        
        
        plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/'+star_name+
                        '/plots/' + 'deconvoluted_' + star_name+'_' + fltr + '_Cam1' + '.png', 
                        dpi=100, bbox_inches ='tight')
        plt.tight_layout()
    
        msg1='reduction okay for '+ star_name+'_Cam1'
        #return(msg1)
        print(msg1)
        
                           #########################
                           # Pour la déconvolution Intensite par intensite, int. pol par Int. pol
                           #########################
 
        
        # plt.clf()
        # fig = plt.figure('deconv_' + f'{star_name}' +'(' + f'{fltr}' + '_Cam1' + ')')
        # fig.set_size_inches(18.5, 10, forward = True)
        # for i in range (nFrames2):
        #       plt.subplot(3,3,i+1)
        #       if i != 2 and i!= 3:      # i!=3 car si non on aurait 4 figures à afficher dans 3 cadres dispo
        #           if np.any(np.min(deconv_arr2[i])<= 0):           
        #               plt.imshow(deconv_arr2[i], cmap='inferno', origin='lower',
        #               vmin = Vmin_d2[i], vmax = Vmax_d2[i], extent = [x_min , x_max, y_min , y_max])
                      
        #               plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
        #                       'deconv_'+f'{star_name}' + '_' + f'{im_name_lst[i]}', color='w',
        #                     fontsize='large', ha='center')
        #               plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
        #           else:
        #                 plt.imshow(np.log10(deconv_arr2[i]), cmap='inferno', origin='lower',
        #                 vmin=Vmin_d2[i], vmax = Vmax_d2[i], extent = [x_min , x_max, y_min , y_max])
                       
        #                 plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
        #                          'deconv_' + f'{star_name}' + '_' + f'{im_name_lst[i]}', color='w',
        #                       fontsize='large', ha='center')
        #                 plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
        #       else :
        #           if i == 2:                  
        #               if np.any(np.min(deconv_arr2[1])<= 0):
        #                   plt.imshow(deconv_arr2[1], cmap ='inferno', origin='lower',vmin = Vmin_d2[1], 
        #                                 vmax = Vmax_d2[1], extent = [x_min , x_max, y_min , y_max])   
        #                   plt.colorbar(label='ADU in log$_{10}$ scale')       
        #                   q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2[::X_step,::X_step], V2[::X_step,::X_step])
        #                   plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')                 
        #                   plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
        #                            'deconv_'+ f'{im_name_lst[1]}'+ '_&_Pol. vect', color='w',
        #                                 fontsize='large', ha='center')
                    
        #               else :
        #                   plt.imshow(np.log10(deconv_arr2[1]), cmap ='inferno', origin='lower',vmin = Vmin_d2[1], 
        #                                   vmax = Vmax_d2[1], extent = [x_min , x_max, y_min , y_max])   
        #                   plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)       
        #                   q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2[::X_step,::X_step], V2[::X_step,::X_step])
        #                   plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')                       
        #                   plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
        #                     'deconv_'+ f'{im_name_lst[1]}'+ '_&_Pol. vect', color='w',
        #                         fontsize='large', ha='center')
        
        #       if i == 0:
        #           plt.ylabel('Relative Dec.(mas)', size=10)   
                          
        # for j in range (len(nDimfigj)): 
        #     plt.subplot(3,3,(nDimfigj[j] + 1))
        #     if j == 2:
        #         print(nDimfigj[j])
        #         if np.any(np.min(sub_v_arr2[1])<= 0):
        #             plt.imshow(sub_v_arr2[1], cmap ='inferno', origin='lower',vmin=Vmin2[1], 
        #                                 vmax=Vmax2[1], extent = [x_min , x_max, y_min , y_max])   
        #             plt.colorbar(label='ADU in log$_{10}$ scale')       
        #             q_ = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2[::X_step,::X_step], V2[::X_step,::X_step])
        #             plt.quiverkey(q_, X = 0.1, Y = 1.03, U = 0.01, label='deg vect. n. scale 0.03', labelpos='E')
        #             plt.text(size[0]//10, 2*pix2mas*size[1]//6., 
        #                         f'{star_name}' + '_' + f'{im_name_lst[1]}'+ '_&_Pol. vect', color='w',
        #                                 fontsize='large', ha='center')
        #         else :
        #             plt.imshow(np.log10(sub_v_arr2[1]), cmap ='inferno', origin='lower',vmin=Vmin2[1], 
        #                                   vmax=Vmax2[1], extent = [x_min , x_max, y_min , y_max])   
        #             plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)       
        #             q_ = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2[::X_step,::X_step], V2[::X_step,::X_step])
        #             plt.quiverkey(q_, X = 0.1, Y = 1.03, U = 0.01, label='deg vect. n. scale 0.03', labelpos='E')
        #             plt.text(size[0]//10, 2*pix2mas*size[1]//6., 
        #                   f'{star_name}' + '_'+ f'{im_name_lst[1]}'+ '_&_Pol. vect', color='w',
        #                           fontsize='large', ha='center')
        #     else:
        #         if np.any(np.min(sub_v_arr2[j])<= 0):           
        #             plt.imshow(sub_v_arr2[j], cmap='inferno', origin='lower',
        #             vmin=Vmin2[j], vmax=Vmax2[j], extent = [x_min , x_max, y_min , y_max])
                      
        #             plt.text(size[0]//10, 2*pix2mas*size[1]//6., 
        #                         f'{star_name}' + '_' + f'{im_name_lst[j]}', color='w',
        #                       fontsize='large', ha='center')
        #             plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
        #         else:
        #             plt.imshow(np.log10(sub_v_arr2[j]), cmap='inferno', origin='lower',
        #             vmin=Vmin2[j], vmax=Vmax2[j], extent = [x_min , x_max, y_min , y_max])
                       
        #             plt.text(size[0]//10, 2*pix2mas*size[1]//6., 
        #                          f'{star_name}' + '_' + f'{im_name_lst[j]}', color='w',
        #                         fontsize='large', ha='center')
        #             plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
                    
        #     if j == 0:
        #         plt.ylabel('Relative Dec.(mas)', size=10)
        #         plt.xlabel('Relative R.A.(mas)', size=10)
        #     else:
        #         plt.xlabel('Relative R.A.(mas)', size=10)         
                           
        # for k in range(len(nDimfigk)):      
        #       plt.subplot(3,3,nDimfigk[k] + 1)
        #       plt.plot(r_mas, np.log10(mean_sub_v_arr2[k]), color='darkorange',
        #               linewidth = 2, label= f'{star_name}' + '_' + f'{im_name_lst[k]}' + '_profile') 
        #       plt.plot(r_mas, np.log10(mean_sub_v_arr3[k]),color='purple',
        #               linewidth = 2, label = f'{star_name}' + '_' + f'{im_name_lst[k]}' + '_profile' + '_psf') 
        #       plt.legend(loc=0) 
        #       plt.xlabel('r (mas)', size=10) 
        #       if k == 0:
        #           plt.ylabel(r'Intensity in log$_{10}$ scale', size=10)
        
        # plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/'+star_name+
        #                 '/plots/'+ '_f_by_f_'+ 'deconvoluted_' +  star_name+'_' + fltr + '_Cam1' + '.pdf', 
        #                 dpi=100, bbox_inches ='tight')
        
        
        # plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/'+star_name+
        #                 '/plots/' + '_f_by_f_'+ 'deconvoluted_' + star_name+'_' + fltr + '_Cam1' + '.png', 
        #                 dpi=100, bbox_inches ='tight')
        # plt.tight_layout()
        
    msg1='reduction okay for '+ star_name+'_Cam1'
        #return(msg1)
    print(msg1)
     
                           ####################################
                           # Pour la camera 2 de Zimpol
                           ######################################
    
    for m in range(n_lst_fltr3):
        fdir_star_fltr = fdir_star + lst_fltr3[m] +'/'
        fdir_psf_fltr = fdir_psf + lst_fltr3[m] + '/'
        
        fname1='zpl_p23_make_polar_maps-ZPL_SCIENCE_P23_REDUCED'
        fname2='-zpl_science_p23_REDUCED'
        file_I_star= fdir_star_fltr + fname1+'_I'+fname2+'_I.fits'
        file_PI_star= fdir_star_fltr +fname1+'_PI'+fname2+'_PI.fits'
        file_DOLP_star= fdir_star_fltr +fname1+'_DOLP'+fname2+'_DOLP.fits'
        file_AOLP_star= fdir_star_fltr + fname1+'_AOLP'+fname2+'_AOLP.fits'
    
        file_I_psf = fdir_psf_fltr + fname1+'_I'+fname2+'_I.fits'
        file_PI_psf = fdir_psf_fltr +fname1+'_PI'+fname2+'_PI.fits'
        file_DOLP_psf = fdir_psf_fltr + fname1+'_DOLP'+fname2+'_DOLP.fits'
        file_AOLP_psf = fdir_psf_fltr +fname1+'_AOLP'+fname2+'_AOLP.fits'
      
        file_lst = [file_I_star,file_PI_star,file_DOLP_star,file_AOLP_star,
                  file_I_psf,file_PI_psf,file_DOLP_psf,file_AOLP_psf]
        
        file_lst2_ = [file_I_star, file_PI_star, file_DOLP_star, file_AOLP_star]
        file_lst3_ = [file_I_psf, file_PI_psf, file_DOLP_psf, file_AOLP_psf]          
        
        nFrames = len(file_lst)
        nFrames2_ = len(file_lst2_)
        nFrames3_ = len(file_lst3_)
        nFrames_d_ = nFrames2_
    
    
        mean_sub_v_arr2_ = np.empty((nFrames2,nSubDim//2-1))
        mean_sub_v_arr3_ = np.empty((nFrames3,nSubDim//2-1))
        sub_v_arr2_ = np.empty((nFrames2,nSubDim,nSubDim))
        sub_v_arr3_ = np.empty((nFrames3,nSubDim,nSubDim))
        deconv_arr_ = np.empty((nFrames_d,nSubDim,nSubDim))
        
        deconv_arr_2 = np.empty((nFrames_d,nSubDim,nSubDim))
        
        im_name_lst = ['I','PI','DOLP','AOLP',
                        'I','PI','DOLP','AOLP']
        Vmin2_ = np.empty((nFrames3))
        Vmax2_ = np.empty((nFrames3))
        
        Vmin3_ = np.empty((nFrames3))
        Vmax3_ = np.empty((nFrames3))
        
        Vmin_d_ = np.empty((nFrames_d_))
        Vmax_d_ = np.empty((nFrames_d_))
        
        Vmin_d_2 = np.empty((nFrames_d_))
        Vmax_d_2 = np.empty((nFrames_d_))
    
        position = (nDim//2,nDim//2)
        size = (nSubDim, nSubDim)
        
        x, y = np.meshgrid(np.arange(nSubDim), np.arange(nSubDim)) #cree un tableau 
        
        R = np.sqrt((x-nSubDim/2)**2+(y-nSubDim/2)**2)
        r = np.linspace(1,nSubDim//2-1,nSubDim//2-1)
        
        r_mas = pix2mas*r #  où r est en pixels et r_mas en millièmes d'arcseconde
    
      
    
        for i in range (nFrames2_):
              hdu_ = fits.open(file_lst2_[i])[0]   
              data2_ = hdu_.data   
              i_v2_ = data2_[1,:,:]
              fltr_ = hdu_.header.get('HIERARCH ESO INS3 OPTI6 NAME')
              
              cutout2_ = Cutout2D(i_v2_, position = position, size = size)
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
              
        for i in range (nFrames3_):
              hdu3_ = fits.open(file_lst3_[i])   
              data3_ = hdu3_[0].data   
              i_v3_ = data3_[1,:,:]
           
              cutout3_ = Cutout2D(i_v3_, position=position, size=size)
              zoom_hdu3_ = hdu3_.copy()
              sub_v3_ = cutout3_.data
            
              f = lambda r : sub_v3_[(R >= r-0.5) & (R < r+0.5)].mean()   
              mean_sub_v3_ = np.vectorize(f)(r) 
            
              mean_sub_v_arr3_[i] = mean_sub_v3_
              sub_v_arr3_[i] = sub_v3_
              if np.any(np.min(sub_v_arr3_[i])<= 0): #i==3 or i==7:
                  Vmin3_[i] = np.min(sub_v_arr3_[i])
                  Vmax3_[i] = np.max(sub_v_arr3_[i])  
              else:
                  Vmin3_[i] = np.min(np.log10(sub_v_arr3_[i]))
                  Vmax3_[i] = np.max(np.log10(sub_v_arr3_[i]))  
         
              U3_ = sub_v_arr3_[2]*np.cos(np.pi*sub_v_arr3_[3]/180)
              V3_ = sub_v_arr3_[2]*np.sin(np.pi*sub_v_arr3_[3]/180)
        
        for i in range (nFrames_d_):
            deconvolved_RL_ = Margaux_RL_deconv(sub_v_arr2_[i], sub_v_arr3_[0], 5)
            #deconvolved_RL_ = restoration.richardson_lucy(sub_v_arr2[i], sub_v_arr3[0], 10)
            deconv_arr_[i]  = deconvolved_RL_
            
            #deconvolved_RL_2 = arl.Margaux_RL_deconv(sub_v_arr2_[i], sub_v_arr3_[i], 15)
            ##deconvolved_RL_ = restoration.richardson_lucy(sub_v_arr2[i], sub_v_arr3[0], 10)
            #deconv_arr_2[i]  = deconvolved_RL_2
            
            
            if np.any(np.min(deconv_arr_[i])<= 0): #i==3 or i==7:
                Vmin_d_[i] = np.min(deconv_arr_[i])
                Vmax_d_[i] = np.max(deconv_arr_[i])  
            else:
                Vmin_d_[i] = np.min(np.log10(deconv_arr_[i]))
                Vmax_d_[i] = np.max(np.log10(deconv_arr_[i]))  
            
            if np.any(np.min(deconv_arr_2[i])<= 0): #i==3 or i==7:
                Vmin_d_2[i] = np.min(deconv_arr_2[i])
                Vmax_d_2[i] = np.max(deconv_arr_2[i])  
            else:
                Vmin_d_2[i] = np.min(np.log10(deconv_arr_2[i]))
                Vmax_d_2[i] = np.max(np.log10(deconv_arr_2[i]))  
            
            
            print('deconvolution okay for ' + ' ' f'{star_name}' + ' '+ f'{fltr_}'+ ' '+ f'{im_name_lst[i]}' )
        #plt.figure(f'{star_name}' + '(' + f'{fltr_}' + '_Cam2' + ')', figsize=(18.5,10.5, forward=True))
        plt.clf()
        fig = plt.figure('deconv_' + f'{star_name}' +'(' + f'{fltr_}' + '_Cam2' + ')')
        fig.set_size_inches(18.5, 10.5, forward = True)
        for i in range (nFrames2_):
              plt.subplot(3,3,i+1)
              if i != 2 and i!= 3:
                  if np.any(np.min(deconv_arr_[i])<= 0):           
                      plt.imshow(deconv_arr_[i], cmap='inferno', origin='lower',
                      vmin=Vmin_d_[i], vmax = Vmax_d_[i], extent = [x_min , x_max, y_min , y_max])
                      
                      plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                                'deconv_' + f'{star_name}' + '_' + f'{im_name_lst[i]}', color='w',
                            fontsize='large', ha='center')
                      plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
                  else:
                        plt.imshow(np.log10(deconv_arr_[i]), cmap='inferno', origin='lower',
                        vmin = Vmin_d_[i], vmax = Vmax_d_[i], extent = [x_min , x_max, y_min , y_max])
                       
                        plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                                  'deconv_' + f'{star_name}' + '_' + f'{im_name_lst[i]}', color='w',
                              fontsize='large', ha='center')
                        plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
              else :
                  if i == 2:                  
                      if np.any(np.min(deconv_arr_[1])<= 0):
                          plt.imshow(deconv_arr_[1], cmap ='inferno', origin='lower',vmin=Vmin_d_[1], 
                                        vmax = Vmax_d_[1], extent = [x_min , x_max, y_min , y_max])   
                          plt.colorbar(label='ADU in log$_{10}$ scale')       
                          q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2_[::X_step,::X_step], V2_[::X_step,::X_step])
                          plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')
                          plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                            'deconv_' + f'{im_name_lst[1]}' + '_&_Pol. vect', color='w',
                                fontsize = 'large', ha ='center')
                      else :
                          plt.imshow(np.log10(deconv_arr_[1]), cmap ='inferno', origin='lower',vmin = Vmin_d_[1], 
                                          vmax = Vmax_d_[1], extent = [x_min , x_max, y_min , y_max])   
                          plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)       
                          q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2_[::X_step,::X_step], V2_[::X_step,::X_step])
                          plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')
                          plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
                            'deconv_'+ f'{im_name_lst[1]}' + '_&_Pol. vect', color='w',
                                fontsize ='large', ha = 'center')
                          
              if i == 0:
                  plt.ylabel('Relative Dec.(mas)', size=10)   
                          
        for j in range (len(nDimfigj)):   
            plt.subplot(3,3,(nDimfigj[j] + 1))
            if j == 2:
                if np.any(np.min(sub_v_arr2_[1])<= 0):
                    plt.imshow(sub_v_arr2_[1], cmap ='inferno', origin='lower',vmin=Vmin2_[1], 
                                        vmax = Vmax2_[1], extent = [x_min , x_max, y_min , y_max])   
                    plt.colorbar(label='ADU in log$_{10}$ scale')       
                    q_ = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2_[::X_step,::X_step], V2_[::X_step,::X_step])
                    plt.quiverkey(q_, X = 0.1, Y = 1.03, U = 0.01, label='deg vect. n. scale 0.03', labelpos='E')
                    plt.text(size[0]//10, 2*pix2mas*size[1]//6., 
                                f'{im_name_lst[1]}'+ '_&_Pol. vect', color='w',
                                        fontsize='large', ha='center')
                else :
                    plt.imshow(np.log10(sub_v_arr2_[1]), cmap ='inferno', origin='lower',vmin=Vmin2_[1], 
                                          vmax=Vmax2_[1], extent = [x_min , x_max, y_min , y_max])   
                    plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)       
                    q_ = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2_[::X_step,::X_step], V2_[::X_step,::X_step])
                    plt.quiverkey(q_, X = 0.1, Y = 1.03, U = 0.01, label='deg vect. n. scale 0.03', labelpos='E')
                        
                    plt.text(size[0]//10, 2*pix2mas*size[1]//6., 
                          f'{im_name_lst[1]}'+ '_&_Pol. vect', color='w',
                                  fontsize='large', ha='center')
            else:
                if np.any(np.min(sub_v_arr2_[j])<= 0):           
                    plt.imshow(sub_v_arr2_[j], cmap='inferno', origin='lower',
                    vmin = Vmin2_[j], vmax = Vmax2_[j], extent = [x_min , x_max, y_min , y_max])
                      
                    plt.text(size[0]//10, 2*pix2mas*size[1]//6., 
                    f'{star_name}' + '_' + f'{im_name_lst[j]}', color='w',
                              fontsize='large', ha='center')
                    plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
                else:
                    plt.imshow(np.log10(sub_v_arr2_[j]), cmap='inferno', origin='lower',
                    vmin = Vmin2_[j], vmax = Vmax2_[j], extent = [x_min , x_max, y_min , y_max])
                       
                    plt.text(size[0]//10, 2*pix2mas*size[1]//6., 
                    f'{star_name}' + '_' + f'{im_name_lst[j]}', color='w',
                                fontsize='large', ha='center')
                    plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
                    
            if j == 0:
                plt.ylabel('Relative Dec.(mas)', size=10)
                plt.xlabel('Relative R.A.(mas)', size=10)
            else:
                plt.xlabel('Relative R.A.(mas)', size=10)         
                           
        for k in range(len(nDimfigk)):      
              plt.subplot(3,3,nDimfigk[k] + 1)
              plt.plot(r_mas, np.log10(mean_sub_v_arr2_[k]), color='darkorange',
                      linewidth = 2, label= f'{star_name}' + '_' + f'{im_name_lst[k]}' + '_profile') 
              plt.plot(r_mas, np.log10(mean_sub_v_arr3_[k]),color='purple',
                      linewidth = 2, label = f'{star_name}' + '_' + f'{im_name_lst[k]}' + '_profile'+ '_psf') 
              plt.legend(loc=0) 
              plt.xlabel('r (mas)', size=10) 
              if k == 0: 
                  plt.ylabel(r'Intensity in log$_{10}$ scale', size=10)
        
        plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/'+star_name+
                        '/plots/' +'deconvoluted_' + star_name + '_' + fltr_ + '_Cam2' + '.pdf', 
                        dpi=100, bbox_inches ='tight')
        
        
        plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/'+star_name+
                        '/plots/' +'deconvoluted_' +star_name+'_' + fltr_ + '_Cam2' + '.png', 
                        dpi=100, bbox_inches ='tight')
        plt.tight_layout()
    
        
                              #########################
                              # Pour la déconvolution Intensite par intensite, int. pol par Int. pol
                              #########################
    
        # plt.clf()
        # fig = plt.figure('deconv_' + f'{star_name}' +'(' + f'{fltr_}' + '_Cam2' + ')')
        # fig.set_size_inches(18.5, 10.5, forward = True)
        # for i in range (nFrames2_):
        #       plt.subplot(3,3,i+1)
        #       if i != 2 and i!= 3:
        #           if np.any(np.min(deconv_arr_2[i])<= 0):           
        #               plt.imshow(deconv_arr_2[i], cmap='inferno', origin='lower',
        #               vmin=Vmin_d_2[i], vmax = Vmax_d_2[i], extent = [x_min , x_max, y_min , y_max])
                      
        #               plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
        #                         'deconv_' + f'{star_name}' + '_' + f'{im_name_lst[i]}', color='w',
        #                     fontsize='large', ha='center')
        #               plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
        #           else:
        #                 plt.imshow(np.log10(deconv_arr_2[i]), cmap='inferno', origin='lower',
        #                 vmin = Vmin_d_2[i], vmax = Vmax_d_2[i], extent = [x_min , x_max, y_min , y_max])
                       
        #                 plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
        #                           'deconv_' + f'{star_name}' + '_' + f'{im_name_lst[i]}', color='w',
        #                       fontsize='large', ha='center')
        #                 plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
        #       else :
        #           if i == 2:                  
        #               if np.any(np.min(deconv_arr_2[1])<= 0):
        #                   plt.imshow(deconv_arr_2[1], cmap ='inferno', origin='lower',vmin=Vmin_d_2[1], 
        #                                 vmax = Vmax_d_2[1], extent = [x_min , x_max, y_min , y_max])   
        #                   plt.colorbar(label='ADU in log$_{10}$ scale')       
        #                   q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2_[::X_step,::X_step], V2_[::X_step,::X_step])
        #                   plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')
        #                   plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
        #                     'deconv_' + f'{im_name_lst[1]}' + '_&_Pol. vect', color='w',
        #                         fontsize = 'large', ha ='center')
        #               else :
        #                   plt.imshow(np.log10(deconv_arr_2[1]), cmap ='inferno', origin='lower',vmin = Vmin_d_2[1], 
        #                                   vmax = Vmax_d_2[1], extent = [x_min , x_max, y_min , y_max])   
        #                   plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)       
        #                   q = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2_[::X_step,::X_step], V2_[::X_step,::X_step])
        #                   plt.quiverkey(q, X = 0.1, Y = 1.03, U = 0, label='', labelpos='E')
        #                   plt.text(size[0]//10, 2*pix2mas*size[1]//6.,
        #                     'deconv_'+ f'{im_name_lst[1]}' + '_&_Pol. vect', color='w',
        #                         fontsize ='large', ha = 'center')
                          
        #       if i == 0:
        #           plt.ylabel('Relative Dec.(mas)', size=10)   
                          
        # for j in range (len(nDimfigj)):   
        #     plt.subplot(3,3,(nDimfigj[j] + 1))
        #     if j == 2:
        #         if np.any(np.min(sub_v_arr2_[1])<= 0):
        #             plt.imshow(sub_v_arr2_[1], cmap ='inferno', origin='lower',vmin=Vmin2_[1], 
        #                                 vmax = Vmax2_[1], extent = [x_min , x_max, y_min , y_max])   
        #             plt.colorbar(label='ADU in log$_{10}$ scale')       
        #             q_ = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2_[::X_step,::X_step], V2_[::X_step,::X_step])
        #             plt.quiverkey(q_, X = 0.1, Y = 1.03, U = 0.01, label='deg vect. n. scale 0.03', labelpos='E')
        #             plt.text(size[0]//10, 2*pix2mas*size[1]//6., 
        #                         f'{im_name_lst[1]}'+ '_&_Pol. vect', color='w',
        #                                 fontsize='large', ha='center')
        #         else :
        #             plt.imshow(np.log10(sub_v_arr2_[1]), cmap ='inferno', origin='lower',vmin=Vmin2_[1], 
        #                                   vmax=Vmax2_[1], extent = [x_min , x_max, y_min , y_max])   
        #             plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)       
        #             q_ = plt.quiver(X[::X_step,::X_step],Y[::X_step,::X_step],U2_[::X_step,::X_step], V2_[::X_step,::X_step])
        #             plt.quiverkey(q_, X = 0.1, Y = 1.03, U = 0.01, label='deg vect. n. scale 0.03', labelpos='E')
                        
        #             plt.text(size[0]//10, 2*pix2mas*size[1]//6., 
        #                   f'{im_name_lst[1]}'+ '_&_Pol. vect', color='w',
        #                           fontsize='large', ha='center')
        #     else:
        #         if np.any(np.min(sub_v_arr2_[j])<= 0):           
        #             plt.imshow(sub_v_arr2_[j], cmap='inferno', origin='lower',
        #             vmin = Vmin2_[j], vmax = Vmax2_[j], extent = [x_min , x_max, y_min , y_max])
                      
        #             plt.text(size[0]//10, 2*pix2mas*size[1]//6., 
        #             f'{star_name}' + '_' + f'{im_name_lst[j]}', color='w',
        #                       fontsize='large', ha='center')
        #             plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
        #         else:
        #             plt.imshow(np.log10(sub_v_arr2_[j]), cmap='inferno', origin='lower',
        #             vmin = Vmin2_[j], vmax = Vmax2_[j], extent = [x_min , x_max, y_min , y_max])
                       
        #             plt.text(size[0]//10, 2*pix2mas*size[1]//6., 
        #             f'{star_name}' + '_' + f'{im_name_lst[j]}', color='w',
        #                         fontsize='large', ha='center')
        #             plt.colorbar(label='ADU in log$_{10}$ scale', shrink = 0.6)
                    
        #     if j == 0:
        #         plt.ylabel('Relative Dec.(mas)', size=10)
        #         plt.xlabel('Relative R.A.(mas)', size=10)
        #     else:
        #         plt.xlabel('Relative R.A.(mas)', size=10)         
                           
        # for k in range(len(nDimfigk)):      
        #       plt.subplot(3,3,nDimfigk[k] + 1)
        #       plt.plot(r_mas, np.log10(mean_sub_v_arr2_[k]), color='darkorange',
        #               linewidth = 2, label= f'{star_name}' + '_' + f'{im_name_lst[k]}' + '_profile') 
        #       plt.plot(r_mas, np.log10(mean_sub_v_arr3_[k]),color='purple',
        #               linewidth = 2, label = f'{star_name}' + '_' + f'{im_name_lst[k]}' + '_profile' + '_psf') 
        #       plt.legend(loc=0) 
        #       plt.xlabel('r (mas)', size=10) 
        #       if k == 0: 
        #           plt.ylabel(r'Intensity in log$_{10}$ scale', size=10)
        
        # plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/'+star_name+
        #                 '/plots/' + 'f_by_f_'+'deconvoluted_' + star_name + '_' + fltr_ + '_Cam2' + '.pdf', 
        #                 dpi=100, bbox_inches ='tight')
        
        
        # plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/'+star_name+
        #                 '/plots/'+ 'f_by_f_'+ 'deconvoluted_' +star_name+'_' + fltr_ + '_Cam2' + '.png', 
        #                 dpi=100, bbox_inches ='tight')
        # plt.tight_layout()
    
    msg2= 'reduction okay for ' + star_name +'_Cam2'

    msg= 'reduction okay for '+ star_name
    print(msg2)
    return(msg) 


