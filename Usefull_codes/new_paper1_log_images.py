#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:47:34 2023

@author: nbadolo
"""

"""
Code simplifié pour l'affichage simultané de toutes les étoiles(avec psf) de alone et both  ainsi que leur psf: flux 
et profile radial d'intensité. Code okay à la date du 05 dec 2023!!!
"""

import numpy as np
import os
import scipy 
from os.path import exists
from astropy.io import fits
from scipy import optimize
from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.colors as colors
from matplotlib.pyplot import Figure, subplot
import webbrowser
from mpl_toolkits.axes_grid1 import make_axes_locatable
from AymardPack import process_fits_image as pfi # Pour l'extraction du bruit et des pixels morts et chauds
#%% 
#star_name = 'SW_Col'
#obsmod = 'both'

txt_folder = 'sphere_files'
file_path = '/home/nbadolo/Bureau/Aymard/Donnees_sph/' + txt_folder + '/'
file_name = 'no_psf_star_lst.txt'
#no_psf_star_lst = open("{}/{}".format(file_path, file_name), "w")
#no_psf_star_lst.write("{}\n".format('Star name', 'Mode'))
#%%
def log_image(star_name, obsmod):             
#%%        
    
    ##Parameters
    nDim = 1024
    nSubDim = 100 # plage de pixels que l'on veut afficher
    size = (nSubDim, nSubDim)
    
    label_size_ = 40 # taille des étiquettes de la graduation
    label_size2_ = 35 # taille du nom des axes
    label_size3_ = 60 # taille du texte dans l'image sans psf
    
    
    label_size = 30 # taille des étiquettes de la graduation 
    label_size2 = 25 # taille du nom des axes
    label_size3 = 30 # taille du texte dans l'image
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
    
    #fdir= '/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'+star_name+ '/'
    fdir= '/home/nbadolo/Bureau/Aymard/Donnees_sph/First/'+star_name+ '/'
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
    n_lst_fltr2 = len(lst_fltr2_star)
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
    print(lst_fltr3)
    n_lst_fltr3 = len(lst_fltr3)
    #print(n_lst_fltr3)
    fsize = [0,1]       
    n_fsize = len (fsize)
    
    if n_lst_fltr3 == 0: # si l'étoile et sa psf n'ont pas de data en commun pour le même filtre
           #no_psf_star_lst.write("{}\n".format(f'{star_name}', obsmod))
           
           if n_lst_fltr2 == 0 :
               print (f'Any data for  {star_name} in ' + obsmod + ' mode')
           else :
               print( f'No common data for {star_name} and his psf in '+ obsmod + ' mode')
               
               for l in range(n_lst_fltr2) :
                   
                   fdir_star_fltr = fdir_star + lst_fltr2_star[l] +'/'
                   
                   fname1 ='zpl_p23_make_polar_maps-ZPL_SCIENCE_P23_REDUCED'
                   fname2 ='-zpl_science_p23_REDUCED'
                   file_I_star = fdir_star_fltr + fname1+'_I'+fname2+'_I.fits'
                   file_PI_star = fdir_star_fltr +fname1+'_PI'+fname2+'_PI.fits'
                   file_DOLP_star = fdir_star_fltr +fname1+'_DOLP'+fname2+'_DOLP.fits'
                   file_AOLP_star = fdir_star_fltr + fname1+'_AOLP'+fname2+'_AOLP.fits'
                   file_Q_star = fdir_star_fltr + fname1+'_Q'+fname2+'_Q.fits'
                   file_U_star = fdir_star_fltr + fname1+'_U'+fname2+'_U.fits'
                   
                   file_lst = [file_I_star, file_PI_star, file_DOLP_star, file_AOLP_star, file_Q_star, file_U_star]
                   nFrames = len(file_lst)
                   sub_v_arr = np.empty((n_fsize, nFrames,nSubDim,nSubDim))
                   im_name_lst = ['I','PI','DOLP','AOLP',
                                   'I','PI','DOLP','AOLP']
                   #fltr_arr = np.empty((n_lst_fltr2, n_fsize), dtype = str)
                   fltr_arr= []
                   hduh = fits.open(file_lst[0])[0]
                   star_name_im = hduh.header.get('OBJECT')
                   fltr1 = hduh.header.get('HIERARCH ESO INS3 OPTI5 NAME')
                   fltr2 = hduh.header.get('HIERARCH ESO INS3 OPTI6 NAME')
                   print(star_name_im)
                   print(fltr1)
                   print(fltr2)
                   print(np.shape(fltr1))
                   fltr_arr.append(fltr1)
                   fltr_arr.append(fltr2)
                   print(fltr_arr)
                   
                   for z in range(n_fsize) :
                       
                        for i in range (nFrames):
                             hdu = fits.open(file_lst[i])[0]   
                             data = hdu.data   
                             i_v = data[z,:,:]
                                           
                             cutout = Cutout2D(i_v, position=position, size=size)
                             zoom_hdu = hdu.copy()
                             sub_v = cutout.data
                             sub_v = pfi(sub_v) # Extraction des pixels chauds
                             
                             f2 = lambda r : sub_v[(R >= r-0.5) & (R < r+0.5)].mean()   
                             mean_sub_v2 = np.vectorize(f2)(r) 
                           
                             #mean_sub_v_arr2[z][i] = mean_sub_v2 
                             sub_v_arr[z][i] = sub_v
                       
                        
                        
                        
                        plt.clf()
                        fig, ax1 = plt.subplots(figsize=(6, 5))

                        im1 = ax1.imshow(
                            sub_v_arr[z][0],
                            cmap='inferno',
                            origin='lower',
                            vmin=np.min(sub_v_arr[z][0]),
                            vmax=np.max(sub_v_arr[z][0]),
                            extent=[x_min, x_max, y_min, y_max]
                        )

                        # Titres et filtres (coordonnées relatives, bien placés)
                        ax1.text(0.02, 0.95, f'{star_name_im}', transform=ax1.transAxes, fontsize=12, fontweight='bold', color='white', va='top')
                        ax1.text(0.02, 0.02, f'{fltr_arr[z]}', transform=ax1.transAxes, fontsize=12, fontweight='bold', color='white', va='bottom')
                        
                        # Colorbar parfaitement collée au bord droit
                        divider = make_axes_locatable(ax1)
                        cax = divider.append_axes('right', size='3%', pad=0.018)  # pad très petit
                        cmapProp = {'drawedges': True}
                        im1_max = np.max(sub_v_arr[z][0])
                        if im1_max < 1e4:
                            cb1 = fig.colorbar(im1, cax=cax, orientation='vertical', ticks=[1e3,2e3,3e3,4e3,5e3], **cmapProp)
                        elif 1e4 < im1_max < 1e5:
                            cb1 = fig.colorbar(im1, cax=cax, orientation='vertical', ticks=[1e4,2e4,3e4,4e4,5e4], **cmapProp)
                        elif 1e5 < im1_max < 1e6:
                            cb1 = fig.colorbar(im1, cax=cax, orientation='vertical', ticks=[1e5,2e5,3e5,4e5,5e5], **cmapProp)
                        else:
                            cb1 = fig.colorbar(im1, cax=cax, orientation='vertical', **cmapProp)
                        cb1.ax.tick_params(labelsize=12)
                        cb1.formatter.set_powerlimits((0, 0))
                        cb1.ax.yaxis.get_offset_text().set(size=12, weight='bold')
                        for tick in cb1.ax.yaxis.get_major_ticks():
                            tick.label2.set_fontweight('bold')

                        # Axes et ticks
                        ax1.set_xlabel('Relative RA (mas)', fontsize=11, weight='bold')
                        ax1.set_ylabel('Relative Dec (mas)', fontsize=11, weight='bold', labelpad=1.5)
                        ax1.tick_params(axis='both', labelsize=9, width=1.2)
                        # ax1.set_xticks([-150, 0, 150])
                        # ax1.set_yticks([-150, -75, 0, 75, 150])
                        # ax1.set_yticklabels(ax1.get_yticks(), weight='bold')
                        # ax1.set_xticklabels(ax1.get_xticks(), rotation=0, weight='bold')

                        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
                            label.set_fontweight('bold')
                        ax1.locator_params(axis='x', nbins=5)
                        ax1.locator_params(axis='y', nbins=5)

                        

                        #plt.subplots_adjust(left=0.13, right=0.93, top=0.97, bottom=0.10)  # marges serrées
                        plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10)
                        plt.savefig(
                            '/home/nbadolo/Bureau/Aymard/Donnees_sph/First/' + star_name +
                            '/plots/no_psf/I/' + star_name + '_' + f'{obsmod}_{fltr_arr[z]}_{z}' + '_lin' + '.pdf',
                            dpi=150, bbox_inches='tight', pad_inches=0.01
                        )
                        plt.savefig(
                            '/home/nbadolo/Bureau/Aymard/Donnees_sph/First/' + star_name +
                            '/plots/no_psf/I/' + star_name + '_' + f'{obsmod}_{fltr_arr[z]}_{z}' + '_lin' + '.png',
                            dpi=150, bbox_inches='tight', pad_inches=0.01
                        )
                        plt.close(fig)
    else : # l'étoile a une psf
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
            
            # mean_sub_v_arr2 = np.empty((2, nFrames2,nSubDim//2-1))
            # mean_sub_v_arr3 = np.empty((2, nFrames3,nSubDim//2-1))
            sub_v_arr2 = np.empty((2, nFrames2,nSubDim,nSubDim))
            sub_v_arr3 = np.empty((2, nFrames3,nSubDim,nSubDim))
            im_name_lst = ['I','PI','DOLP','AOLP',
                            'I','PI','DOLP','AOLP']
            Vmin2 = np.empty((nFrames2))
            Vmax2 = np.empty((nFrames2))
            
            Vmin3 = np.empty((nFrames3))
            Vmax3 = np.empty((nFrames3))
            
            
            #fltr_arr = np.empty((n_lst_fltr3, n_fsize), dtype = str)
            fltr_arr = []
            hduh = fits.open(file_lst[0])[0]
            star_name2 = hduh.header.get('OBJECT')
            fltr1 = hduh.header.get('HIERARCH ESO INS3 OPTI5 NAME')
            fltr2 = hduh.header.get('HIERARCH ESO INS3 OPTI6 NAME')
            fltr_arr.append(fltr1)
            fltr_arr.append(fltr2)
            print(fltr_arr)
            for z in range(n_fsize) : 
                
                for i in range (nFrames2):  # pour l'étoile
                      hdu2 = fits.open(file_lst2[i])[0]   
                      data2 = hdu2.data   
                      i_v2 = data2[z,:,:]
                                    
                      cutout2 = Cutout2D(i_v2, position=position, size=size)
                      zoom_hdu = hdu2.copy()
                      sub_v2 = cutout2.data
                      sub_v2 = pfi(sub_v2) # Extraction des pixels chauds
                    
                      f2 = lambda r : sub_v2[(R >= r-0.5) & (R < r+0.5)].mean()   
                      mean_sub_v2 = np.vectorize(f2)(r) 
                    
                      #mean_sub_v_arr2[z][i] = mean_sub_v2 
                      sub_v_arr2[z][i] = sub_v2
                          
                      # DOLP_star = sub_v_arr2[z][2]
                      # ii = (sub_v_arr2[z][4] == 0)
                      # if True in ii:
                      #     sub_v_arr2[z][4][ii] = sub_v_arr2[z][4][ii] + 0.0001  # introduction d'un ofset pour les valeurs de Q == 0
                            
                      # AOLP_2_star = 0.5*np.arctan2(sub_v_arr2[z][5], sub_v_arr2[z][4])
                      # U2 = DOLP_star*np.cos(-(AOLP_2_star + np.pi/2))
                      # V2 = DOLP_star*np.sin(-(AOLP_2_star + np.pi/2))
                      
                for i in range (nFrames3):  # pour la psf
                      hdu3 = fits.open(file_lst3[i])[0]  
                      data3 = hdu3.data   
                      i_v3 = data3[z,:,:]
                      psf_name = hdu3.header.get('OBJECT')
                      cutout3 = Cutout2D(i_v3, position=position, size=size)
                      zoom_hdu3 = hdu3.copy()
                      sub_v3 = cutout3.data
                      sub_v3 = pfi(sub_v3) # Extraction des pixels chauds
                    
                      f3 = lambda r : sub_v3[(R >= r-0.5) & (R < r+0.5)].mean()   
                      mean_sub_v3 = np.vectorize(f3)(r) 
                      
                      #mean_sub_v_arr3[z][i] = mean_sub_v3 
                      sub_v_arr3[z][i] = sub_v3
                    
                      # DOLP_psf = sub_v_arr3[z][2]
                      # ii = (sub_v_arr3[z][4] == 0)
                      # if True in ii:
                      #    sub_v_arr3[z][4][ii] = sub_v_arr3[z][4][ii] + 0.0001  # introduction d'un ofset pour les valeurs de Q == 0
                            
                      # AOLP_2_psf = 0.5*np.arctan2(sub_v_arr3[z][5], sub_v_arr3[z][4])
                      # U3 = DOLP_psf*np.cos(-(AOLP_2_psf + np.pi/2))
                      # V3 = DOLP_psf*np.sin(-(AOLP_2_psf + np.pi/2))
                    
                
                # linear scale
                plt.clf()
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # --- Image étoile ---
                im1 = ax1.imshow(
                    sub_v_arr2[z][0],
                    cmap='inferno',
                    origin='lower',
                    vmin=np.min(sub_v_arr2[z][0]),
                    vmax=np.max(sub_v_arr2[z][0]),
                    extent=[x_min, x_max, y_min, y_max]
                )
                ax1.text(0.02, 0.95, f'{star_name2}', transform=ax1.transAxes, fontsize=12, fontweight='bold', color='white', va='top')
                ax1.text(0.02, 0.02, f'{fltr_arr[z]}', transform=ax1.transAxes, fontsize=12, fontweight='bold', color='white', va='bottom')

                divider1 = make_axes_locatable(ax1)
                cax1 = divider1.append_axes('right', size='3%', pad=0.018)
                cmapProp = {'drawedges': True}
                im1_max = np.max(sub_v_arr2[z][0])
                if im1_max < 1e4:
                    cb1 = fig.colorbar(im1, cax=cax1, orientation='vertical', ticks=[1e3,2e3,3e3,4e3,5e3], **cmapProp)
                elif 1e4 < im1_max < 1e5:
                    cb1 = fig.colorbar(im1, cax=cax1, orientation='vertical', ticks=[1e4,2e4,3e4,4e4,5e4], **cmapProp)
                elif 1e5 < im1_max < 1e6:
                    cb1 = fig.colorbar(im1, cax=cax1, orientation='vertical', ticks=[1e5,2e5,3e5,4e5,5e5], **cmapProp)
                else:
                    cb1 = fig.colorbar(im1, cax=cax1, orientation='vertical', **cmapProp)
                cb1.ax.tick_params(labelsize=12)
                cb1.formatter.set_powerlimits((0, 0))
                cb1.ax.yaxis.get_offset_text().set(size=12, weight='bold')
                for tick in cb1.ax.yaxis.get_major_ticks():
                    tick.label2.set_fontweight('bold')

                ax1.set_xlabel('Relative RA (mas)', fontsize=11, weight='bold')
                ax1.set_ylabel('Relative Dec (mas)', fontsize=11, weight='bold', labelpad=1.5)
                ax1.tick_params(axis='both', labelsize=9, width=1.2)
                for label in ax1.get_xticklabels() + ax1.get_yticklabels():
                    label.set_fontweight('bold')
                ax1.locator_params(axis='x', nbins=5)
                ax1.locator_params(axis='y', nbins=5)

                # --- Image PSF ---
                im2 = ax2.imshow(
                    sub_v_arr3[z][0],
                    cmap='inferno',
                    origin='lower',
                    vmin=np.min(sub_v_arr3[z][0]),
                    vmax=np.max(sub_v_arr3[z][0]),
                    extent=[x_min, x_max, y_min, y_max]
                )
                ax2.text(0.02, 0.95, f'{psf_name}', transform=ax2.transAxes, fontsize=12, fontweight='bold', color='white', va='top')

                divider2 = make_axes_locatable(ax2)
                cax2 = divider2.append_axes('right', size='3%', pad=0.018)
                im2_max = np.max(sub_v_arr3[z][0])
                if im2_max < 1e4:
                    cb2 = fig.colorbar(im2, cax=cax2, orientation='vertical', ticks=[1e3,2e3,3e3,4e3,5e3], **cmapProp)
                elif 1e4 < im2_max < 1e5:
                    cb2 = fig.colorbar(im2, cax=cax2, orientation='vertical', ticks=[1e4,2e4,3e4,4e4,5e4], **cmapProp)
                elif 1e5 < im2_max < 1e6:
                    cb2 = fig.colorbar(im2, cax=cax2, orientation='vertical', ticks=[1e5,2e5,3e5,4e5,5e5], **cmapProp)
                else:
                    cb2 = fig.colorbar(im2, cax=cax2, orientation='vertical', **cmapProp)
                cb2.ax.tick_params(labelsize=12)
                cb2.formatter.set_powerlimits((0, 0))
                cb2.ax.yaxis.get_offset_text().set(size=12, weight='bold')
                for tick in cb2.ax.yaxis.get_major_ticks():
                    tick.label2.set_fontweight('bold')

                ax2.set_xlabel('Relative RA (mas)', fontsize=11, weight='bold')
                #ax2.set_ylabel('Relative Dec (mas)', fontsize=11, weight='bold',labelpad=1.5)
                ax2.tick_params(axis='both', labelsize=9, width=1.2)
                for label in ax2.get_xticklabels() + ax2.get_yticklabels():
                    label.set_fontweight('bold')
                ax2.locator_params(axis='x', nbins=5)
                ax2.locator_params(axis='y', nbins=5)
                ax2.axes.yaxis.set_ticklabels([])  # Pas de labels y sur la PSF

                plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10, wspace=0.02)
                plt.savefig(
                    '/home/nbadolo/Bureau/Aymard/Donnees_sph/First/' + star_name +
                    '/plots/star_psf/' + star_name + '_' + f'{obsmod}_{fltr_arr[z]}_{z}' + '_lin' + '.pdf',
                    dpi=150, bbox_inches='tight', pad_inches=0.01
                )
                plt.savefig(
                    '/home/nbadolo/Bureau/Aymard/Donnees_sph/First/' + star_name +
                    '/plots/star_psf/' + star_name + '_' + f'{obsmod}_{fltr_arr[z]}_{z}' + '_lin' + '.png',
                    dpi=150, bbox_inches='tight', pad_inches=0.01
                )
                plt.close(fig)
                    
    return()
        
star=log_image('V854_Cen', 'alone')  
star=log_image('V854_Cen', 'both')
star=log_image('V1943_Sgr', 'both')  
