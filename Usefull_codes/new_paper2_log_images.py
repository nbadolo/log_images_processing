#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:51:26 2023

@author: nbadolo
"""

"""
Code simplifié pour l'affichage simultané de tous les alone et both  des étoiles 
sans tenir compte de leur psf:  flux d'intensité. Code okay au jour du 05 dec 2023.
"""

import numpy as np
import astropy.units as u
import os
import scipy 
from os.path import exists
from astropy.io import fits
from scipy import optimize
from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt
import matplotlib.font_manager as fmg
import matplotlib.colors as colors
from matplotlib.pyplot import Figure, subplot
import webbrowser
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cbook as cbook
from matplotlib_scalebar.scalebar import ScaleBar
from AymardPack import process_fits_image as pfi # Pour l'extraction du bruit et des pixels morts et chauds
#%%
##Parameters
nDim = 1024
nSubDim = 100 # plage de pixels que l'on veut afficher
size = (nSubDim, nSubDim)
label_size = 30 # taille des étiquettes de la graduetion
label_size2 = 25 # taille du nom des axes
label_size3 = 30 # taille du texte dans l'image
pix2mas = 3.4  #en mas/pix
x_min = -pix2mas*nSubDim//2
x_max = pix2mas*(nSubDim//2-1)
y_min = -pix2mas*nSubDim//2
y_max = pix2mas*(nSubDim//2-1)
X, Y= np.meshgrid(np.linspace(-nSubDim/2,nSubDim/2-1,nSubDim), np.linspace(-nSubDim/2,nSubDim/2-1,nSubDim))
X *= pix2mas
Y *= pix2mas
 
X_step = 5
X_step_ = 50
nx = ny = 20
position = (nDim//2,nDim//2)
#%% 
def log_image(star_name, obsmod):
#%%       
    fdir= '/home/nbadolo/Bureau/Aymard/Donnees_sph/First/'+star_name+ '/'
    #fdir= '/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log_+/'+star_name+ '/'
    fdir_star = fdir + 'star/'+obsmod+ '/' 
    #fdir_psf = fdir +'psf/'+obsmod+ '/'
    lst_fltr_star1 = os.listdir(fdir_star)
    #print(lst_fltr_star1)
    n_lst_fltr_star1 = len(lst_fltr_star1)
    #print(n_lst_fltr_star1)
    lst_fltr_star2 = []
    nDimfigj = [3, 4, 5]
    nDimfigk = [6, 7, 8]
    
   
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
        #fdir_psf_fltr = fdir_psf + lst_fltr_star2[l] + '/'
                
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
        # file_lst2 = [file_I_star, file_PI_star, file_DOLP_star, file_AOLP_star]
        # nFrames2 = len(file_lst2)
        
   
        sub_v_arr = np.empty((2, nFrames,nSubDim,nSubDim))
        AOLP_2_star_arr =  np.empty((2, nFrames,nSubDim,nSubDim))
        im_name_lst = ['I','PI','DOLP','AOLP','I_Q', 'I_U']
        # Vmin2 = np.empty((nFrames2))
        # Vmax2 = np.empty((nFrames2))
        
        fsize = [0,1]       
        n_fsize = len (fsize)
        #fltr_arr = np.empty((n_lst_fltr_star2, n_fsize), dtype = str)
        fltr_arr= []
        hduh = fits.open(file_lst[0])[0]
        star_name2 = hduh.header.get('OBJECT')
        fltr1 = hduh.header.get('HIERARCH ESO INS3 OPTI5 NAME')
        fltr2 = hduh.header.get('HIERARCH ESO INS3 OPTI6 NAME')
        fltr_arr.append(fltr1)
        fltr_arr.append(fltr2)
        print(fltr1)
        print(fltr2)
        print('les filtres sont :' + str(fltr_arr))
        # print(fltr_arr)
        # header = hdul[0].header
        # filter_names = [
        #     header.get('HIERARCH ESO INS3 OPTI5 NAME'), 
        #     header.get('HIERARCH ESO INS3 OPTI6 NAME')
        #]  
       
        for z in range(n_fsize) :
            for i in range (nFrames):
                  hdu2 = fits.open(file_lst[i])[0]   
                  data2 = hdu2.data   
                  i_v2 = data2[z,:,:]
                  # star_name2 = hdu2.header.get('OBJECT')
                  # fltr1 = hdu2.header.get('HIERARCH ESO INS3 OPTI5 NAME')   
                  # fltr2 = hdu2.header.get('HIERARCH ESO INS3 OPTI6 NAME')
                  # fltr_arr[l][0] = fltr1
                  # fltr_arr[l][1] = fltr2
                  #print(fltr)                   
                  cutout2 = Cutout2D(i_v2, position=position, size=size)
                  zoom_hdu = hdu2.copy()
                  sub_v2 = cutout2.data
                  #sub_v2 = pfi(sub_v2) # Extraction des pixels chauds et morts
                  sub_v_arr[z][i] = sub_v2
                                   
                  #print(np.max(sub_v_arr[z][2]))
                  jj = (sub_v_arr[z][2] < 0.2*np.max(sub_v_arr[z][2]))
                 
                  if True in jj :
                      sub_v_arr[z][2] == 0
                  DOLP_star = sub_v_arr[z][2]
                  
                  ii = (sub_v_arr[z][4] == 0)
                  if True in ii:
                      sub_v_arr[z][4][ii] = sub_v_arr[z][4][ii] + 0.0001  # introduction d'un ofset pour les valeurs de Q == 0
                        
                  AOLP_2_star = 0.5*np.arctan2(sub_v_arr[z][5], sub_v_arr[z][4]) # l'angle de polarisation recalculer en utilisant arctan2
                  U2 = DOLP_star*np.cos(-(AOLP_2_star + np.pi/2))
                  V2 = DOLP_star*np.sin(-(AOLP_2_star + np.pi/2))
                  AOLP_2_star_arr[z][i] = AOLP_2_star 
           
            # plt.figure()
            # image = plt.imread(cbook.get_sample_data('grace_hopper.png'))
            # plt.imshow(image)
            # scalebar = ScaleBar(0.2) # 1 pixel = 0.2 meter
            # plt.gca().add_artist(scalebar)
            # plt.show()      
            
            # --- Plot des images ---
            plt.clf()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # --- Image DOLP ---
            # --- Image DOLP ---
            im1 = ax1.imshow(
                sub_v_arr[z][2], cmap='inferno', origin='lower',
                vmin=np.min(sub_v_arr[z][2]), vmax=np.max(sub_v_arr[z][2]),
                extent=[x_min, x_max, y_min, y_max]
            )
            ax1.text(0.02, 0.95, f'{star_name2}', transform=ax1.transAxes, fontsize=12, fontweight='bold', color='white', va='top')
            ax1.text(0.02, 0.02, f'{fltr_arr[z]}', transform=ax1.transAxes, fontsize=12, fontweight='bold', color='white', va='bottom')

            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes('right', size='3%', pad=0.02)
            cmapProp = {'drawedges': True}
            im1_max = np.max(sub_v_arr[z][2])
            if 1e-3 < im1_max < 1e-2:
                cb1 = fig.colorbar(im1, cax=cax1, orientation='vertical', ticks=[1e-3,2e-3,3e-3,4e-3,5e-3], **cmapProp)
            elif 1e-2 < im1_max < 1e-1:
                cb1 = fig.colorbar(im1, cax=cax1, orientation='vertical', ticks=[1.0e-2,1.5e-2,2.0e-2,2.5e-2,3.0e-2,3.5e-2,4.0e-2], **cmapProp)
            elif 1e-1 < im1_max:
                cb1 = fig.colorbar(im1, cax=cax1, orientation='vertical', ticks=[1.0e-1,1.5e-1,2.0e-1,2.5e-1,3.0e-1,3.5e-1,4.0e-1], **cmapProp)
            else:
                cb1 = fig.colorbar(im1, cax=cax1, orientation='vertical', **cmapProp)
            #cb1.ax.set_ylabel('DOLP', fontsize=11, weight='bold', labelpad=1.5)
            cb1.ax.tick_params(labelsize=11)
            cb1.formatter.set_powerlimits((0, 0))
            cb1.ax.yaxis.get_offset_text().set(size=11)#, weight='bold')
            # for tick in cb1.ax.yaxis.get_major_ticks():
            #     tick.label2.set_fontweight('bold')

            ax1.set_xlabel('Relative RA(mas)', fontsize=11, weight='bold')
            ax1.set_ylabel('Relative Dec(mas)', fontsize=11, weight='bold',labelpad=1.5)
            ax1.tick_params(axis='both', labelsize=9, width=1.2)
            for label in ax1.get_xticklabels() + ax1.get_yticklabels():
                label.set_fontweight('bold')
            ax1.locator_params(axis='x', nbins=5)
            ax1.locator_params(axis='y', nbins=5)

            # --- Image log10(PI) ---
            im2 = ax2.imshow(
                np.log10(sub_v_arr[z][1]), cmap='inferno', origin='lower',
                vmin=0, vmax=np.max(np.log10(sub_v_arr[z][1])),
                extent=[x_min, x_max, y_min, y_max]
            )
            # Vecteurs de polarisation
            # q = ax2.quiver(
            #     X[::X_step, ::X_step], Y[::X_step, ::X_step],
            #     U2[::X_step, ::X_step], V2[::X_step, ::X_step],
            #     color='w', pivot='mid'
            # )
            #ax2.quiverkey(q, X=0.1, Y=1.03, U=0, label='', labelpos='E')
            ax2.text(0.02, 0.95, f'{star_name2}', transform=ax2.transAxes, fontsize=12, fontweight='bold', color='white', va='top')

            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes('right', size='3%', pad=0.018)
            im2_max = np.max(np.log10(sub_v_arr[z][1]))
            if 1 < im2_max < 5:
                cb2 = fig.colorbar(im2, cax=cax2, orientation='vertical', ticks=[1.0, 1.5, 2.0, 2.5, 3.0], **cmapProp)
            elif 1e1 < im2_max < 1e2:
                cb2 = fig.colorbar(im2, cax=cax2, orientation='vertical', ticks=[1e1,2e1,3e1,4e1,5e1], **cmapProp)
            elif 1e2 < im2_max < 10e2:
                cb2 = fig.colorbar(im2, cax=cax2, orientation='vertical', ticks=[1e2,2e2,3.3e2,44e2,5e2], **cmapProp)
            elif 1e3 < im2_max:
                cb2 = fig.colorbar(im2, cax=cax2, orientation='vertical', ticks=[1e3,2e3,3e3,4e3,5e3], **cmapProp)
            else:
                cb2 = fig.colorbar(im2, cax=cax2, orientation='vertical', **cmapProp)
            #cb2.ax.set_ylabel('log10(PI)', fontsize=11, weight='bold', labelpad=1.5)
            cb2.ax.tick_params(labelsize=11)
            cb2.formatter.set_powerlimits((0, 0))
            cb2.ax.yaxis.get_offset_text().set(size=11, weight='bold')
            #for tick in cb2.ax.yaxis.get_major_ticks():
            #    tick.label2.set_fontweight('bold')

            ax2.set_xlabel('Relative RA(mas)', fontsize=11, weight='bold')
            #ax2.set_ylabel('Relative Dec (mas)', fontsize=11, weight='bold',labelpad=1)
            ax2.tick_params(axis='both', labelsize=9, width=1.2)
            for label in ax2.get_xticklabels() + ax2.get_yticklabels():
                label.set_fontweight('bold')
            ax2.locator_params(axis='x', nbins=5)
            ax2.locator_params(axis='y', nbins=5)
            ax2.axes.yaxis.set_ticklabels([])  # Pas de labels y sur la 2e image

            plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10, wspace=0.15, hspace=0.35)
            plt.savefig(
                '/home/nbadolo/Bureau/Aymard/Donnees_sph/First/' + star_name +
                '/plots/no_psf/PI/' + star_name + '_' + f'{obsmod}_{fltr_arr[z]}_{z}' + '_log_vect.pdf',
                dpi=100, bbox_inches='tight', pad_inches=0.01
            )
            plt.savefig(
                '/home/nbadolo/Bureau/Aymard/Donnees_sph/First/' + star_name +
                '/plots/no_psf/PI/' + star_name + '_' + f'{obsmod}_{fltr_arr[z]}_{z}' + '_log_vect.png',
                dpi=100, bbox_inches='tight', pad_inches=0.01
            )
            plt.close(fig) 
    return()
star=log_image('V854_Cen', 'alone')  
star=log_image('V854_Cen', 'both')  
