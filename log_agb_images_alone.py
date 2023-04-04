#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:26:11 2022

@author: nbadolo
"""

"""
Code simplifié pour l'affichage simultané de tous les alone  et de sa psf: flux 
et profile radial d'intensité'
"""

import numpy as np
import os 
from os.path import exists
from astropy.io import fits
from scipy import optimize
from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.pyplot import Figure, subplot
import webbrowser
#%%
"""
### Creation of data list
"""
# nDatalist= 34
# data_lst = np.empty(nDatalist)
# data_lst = ['17_Lep', 'bet_Gru', 'IRC _10420', 'L02_Pup', 'Mira', 'psi_Phe', 
#             'ups_Cet', 'V_AC_Cet', 'V_BW_Oct', 'V_CW_Cnc', 'V_DZ_Aqr', 'V_pi_01_Gru',
#             'V_R_Aqr','V_R_Hor', 'V_R_Peg', 'V_R_Scl', 'V_RT_Vir', 'V_S_Lep', 'V_S_Pav',
#             'V_SW_Col', 'V_T_Cet', 'V_T_Mic', 'V_V854_Cen', 'V_V1943_Sgr', 'V_V_Hya',
#             'V_W_Peg', 'V_Y_Oph', 'V_Y_Pav', 'V_Y_Scl', 'V_Z_Eri', 'V_Z_Peg', 'w_Pup']

#nData = len(data_lst)


#for i in range(nData):    
#%% 
def log_image(star_name):   
    
#%%        
    fdir= '/home/nbadolo/Bureau/Aymard/Donnees_sph/log/'+star_name+ '/'
    fdir_star = fdir + 'star/alone/' 
    fdir_psf = fdir +'psf/alone/'
    lst_fltr_star = os.listdir(fdir_star)
    print(lst_fltr_star)
    n_lst_fltr_star = len(lst_fltr_star)
    lst_fltr2_star = []
    for p in range(n_lst_fltr_star):
        
        fdir_fltr_data_star = fdir_star + lst_fltr_star[p]
        lst_fltr_data_star = os.listdir(fdir_fltr_data_star) 
        n_lst_fltr_data_star = len(lst_fltr_data_star)
        if n_lst_fltr_data_star != 0:
            lst_fltr2_star.append(lst_fltr_star[p])
    print(lst_fltr2_star)
    
    
    lst_fltr_psf = os.listdir(fdir_psf)
    n_lst_fltr_psf = len(lst_fltr_psf)
    lst_fltr2_psf = []
    for n in range(n_lst_fltr_psf):
        
        fdir_fltr_data_psf = fdir_psf + lst_fltr_psf[n]
        lst_fltr_data_psf = os.listdir(fdir_fltr_data_psf) 
        n_lst_fltr_data_psf = len(lst_fltr_data_psf)
        if n_lst_fltr_data_psf != 0:
            lst_fltr2_psf.append(lst_fltr_psf[n])
    print(lst_fltr2_psf)
    
    lst_fltr3 = list(set(lst_fltr2_star).intersection(lst_fltr2_psf))
    print(lst_fltr3)
    n_lst_fltr3 = len(lst_fltr3)
    for j in range(n_lst_fltr3):
        fdir_star_fltr = fdir_star + lst_fltr3[j] +'/'
        fdir_psf_fltr = fdir_psf + lst_fltr3[j] + '/'
        
        fname1='zpl_p23_make_polar_maps-ZPL_SCIENCE_P23_REDUCED'
        fname2='-zpl_science_p23_REDUCED'
        file_I_star= fdir_star_fltr + fname1+'_I'+fname2+'_I.fits'
        file_PI_star= fdir_star_fltr +fname1+'_PI'+fname2+'_PI.fits'
        file_DOLP_star= fdir_star_fltr +fname1+'_DOLP'+fname2+'_DOLP.fits'
        file_AOLP_star= fdir_star_fltr + fname1+'_AOLP'+fname2+'_AOLP.fits'
    
        file_I_psf= fdir_psf_fltr + fname1+'_I'+fname2+'_I.fits'
        file_PI_psf= fdir_psf_fltr +fname1+'_PI'+fname2+'_PI.fits'
        file_DOLP_psf= fdir_psf_fltr + fname1+'_DOLP'+fname2+'_DOLP.fits'
        file_AOLP_psf= fdir_psf_fltr +fname1+'_AOLP'+fname2+'_AOLP.fits'
      
        file_lst = [file_I_star, file_PI_star, file_DOLP_star, file_AOLP_star,
                  file_I_psf, file_PI_psf, file_DOLP_psf, file_AOLP_psf]
        nFrames = len(file_lst)
        
        """"""
        ## Parameters
        """"""
    
        nDim = 1024
        nSubDim = 200 # plage de pixels que l'on veut afficher
        size = (nSubDim, nSubDim)
        nDimfigj = [0,1,2]
        nDimfigk = [9,10,11]
        vmin0 = 3.5
        vmax0 = 15
        pix2mas = 3.4  #en mas/pix
        x_min = -pix2mas*nSubDim//2
        x_max = pix2mas*(nSubDim//2-1)
        y_min = -pix2mas*nSubDim//2
        y_max = pix2mas*(nSubDim//2-1)
        
        mean_sub_v_arr = np.empty((nFrames,nSubDim//2-1))
        sub_v_arr=np.empty((nFrames,nSubDim,nSubDim))
        im_name_lst = ['I','PI','DOLP','AOLP',
                        'I','PI','DOLP','AOLP']
        Vmin = np.empty((nFrames))
        Vmax = np.empty((nFrames))
    
        position = (nDim//2,nDim//2)
        size = (nSubDim, nSubDim)
        
        x, y = np.meshgrid(np.arange(nSubDim),np.arange(nSubDim)) #cree un tableau 
        
        R = np.sqrt((x-nSubDim/2)**2+(y-nSubDim/2)**2)
        r = np.linspace(1,nSubDim//2-1,nSubDim//2-1)
        
        r_mas=pix2mas*r #  où r est en pixels et r_mas en millièmes d'arcseconde
    
      # """
      # Filtre utilisé: I_PRIM 
      # """
    
        for i in range (nFrames):
              hdu = fits.open(file_lst[i])   
              data = hdu[0].data   
              i_v = data[0,:,:]
           
              cutout = Cutout2D(i_v, position=position, size=size)
              zoom_hdu = hdu.copy()
              sub_v = cutout.data
            
              f = lambda r : sub_v[(R >= r-0.5) & (R < r+0.5)].mean()   
              mean_sub_v = np.vectorize(f)(r) 
            
              mean_sub_v_arr[i] = mean_sub_v 
              sub_v_arr[i]=sub_v
              if i==3 or i==7:
                  Vmin[i]=np.min(sub_v_arr[i])
                  Vmax[i]=np.max(sub_v_arr[i])  
              else:
                  Vmin[i]=np.min(np.log10(sub_v_arr[i]))
                  Vmax[i]=np.max(np.log10(sub_v_arr[i]))  
          
    
        plt.figure(f'{star_name}' +'(' + f'{ lst_fltr3[j]}' + ')', figsize=(40,20.5))
        plt.clf()    
        for i in range (nFrames):   
              plt.subplot(3,4,i+1)
              if i==3 or i==7:
                  plt.imshow(sub_v_arr[i], cmap='inferno', origin='lower',
                vmin=Vmin[i], vmax=Vmax[i], extent = [x_min , x_max, y_min , y_max])
              else:
                  plt.imshow(np.log10(sub_v_arr[i]), cmap='inferno', origin='lower',
                vmin=Vmin[i], vmax=Vmax[i], extent = [x_min , x_max, y_min , y_max])   
            
              if i < 4 :
                  plt.text(-1.1*pix2mas*size[0]//6., 2*pix2mas*size[1]//6.,
                            f'{star_name}' + '_' + f'{im_name_lst[i]}', color='w',
                        fontsize='large', ha='center')
                  plt.colorbar(label='ADU in log$_{10}$ scale')
                  plt.clim(Vmin[i],Vmax[i])
              else:
                  plt.text(size[0]//10, 2*pix2mas*size[1]//6., 
                            f'{star_name}' + '_psf_' + f'{im_name_lst[i]}', color='w',
                          fontsize='large', ha='center')
                  plt.colorbar(label='ADU in log$_{10}$ scale')
                  plt.clim(Vmin[i],Vmax[i])
        plt.xlabel('Relative R.A.(mas)', size=10)
              # plt.ylabel('Relative Dec.(mas)', size=10)
           
        
        for k in range(len(nDimfigk)):      
              plt.subplot(3,4,nDimfigk[k])
              plt.plot(r_mas, np.log10(mean_sub_v_arr[k]), color='darkorange',
                      linewidth=2, label='Mira') 
              plt.plot(r_mas, np.log10(mean_sub_v_arr[k+4]),color='purple',
                      linewidth=2, label='HD204971') 
              plt.legend(loc=0) 
              plt.xlabel('r (mas)', size=10) 
              if k == 0:
                  plt.ylabel(r'Intensity in log$_{10}$ scale', size=10)
        
        # plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/'+star_name+
        #                 '/plots/'+star_name+'_' +lst_fltr3[j] + '.pdf', 
        #                 dpi=100, bbox_inches ='tight')
        
        
        # plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/'+star_name+
        #                 '/plots/'+star_name+'_' +lst_fltr3[j] + '.png', 
        #                 dpi=100, bbox_inches ='tight')
        plt.tight_layout()
    
    msg='reduction okay for '+ star_name
    return(msg)


#log_image('SW_Col')
    
