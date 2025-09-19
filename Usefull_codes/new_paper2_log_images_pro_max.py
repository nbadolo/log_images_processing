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
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# Ajoute ce bloc tout en haut du fichier (après les imports)

star_filters = {
    'Y_Pav':    ('both', ['V_N_R']),
    'R_Hor':    ('both', ['V_N_R']),
    'R_Scl':    ('both', ['V_N_R']),
    'Y_Scl':    ('both', ['V_N_R']),
    'R_Crt':    ('alone', ['N_R']),
    'SW_Col':   ('both', ['V_N_R']),
    'W_Hya':    ('both',  ['Cnt820_Cnt748']),
    'SW_Vir':   ('alone', ['N_R']),
    'V_Hya':    ('both', ['V_N_R']),
    'Alpha_Her':('both', ['V_Cnt748']),
    'R_Hya':     ('alone', ['CntHa']),
    'Chi_Cyg':  ('both', ['V_Cnt748']),
    'Z_Eri':     ('both', ['V_N_R']),
    'R_Peg':     ('both', ['V_N_R']),
    'BW_Oct':    ('both', ['V_N_R']),
    'AC_Cet':    ('both', ['V_N_R']),
    'DZ_Aqr':    ('both', ['V_N_R']),
    'Z_Peg':     ('both', ['V_N_R']),
    'W_Peg':     ('both', ['V_N_R']),
    'RT_Vir':   ('alone', ['Cnt820']),
    'RX_Lep':   ('alone', ['CntHa']),
    'Beta_Gru': ('both', ['V_N_R']), 
    'T_Mic':    ('both', ['V_N_R']),
    'R_Dor':    ('both',  ['Cnt820_Cnt748']),
    'AK_Hya':   ('alone', ['N_R']),
    'R_Leo':    ('both', ['V_Cnt748']),
    'BK_Vir':   ('alone', ['Cnt820']),
    'T_Cet':    ('both',  ['CntHa_B_Ha']),
    'U_Del':    ('alone', ['CntHa']),
    'U_Her':    ('alone', ['VBB']),
    'W_Aql':    ('alone', ['VBB']),
    'V_PsA':    ('alone', ['N_R']),
    'R_Aql':    ('alone', ['N_R']),
    'S_Pav':    ('alone', ['N_R']),
    'GY_Aql':   ('alone', ['VBB']),
    'SV_Aqr':   ('alone', ['VBB']),
    'Ups_Cet':  ('both', ['V_N_R']),
    'V1943_Sgr':('both', ['V_N_R']),
    'Psi_Phe':  ('both', ['V_N_R']),
    'S_Lep':  ('alone', ['I_PRIM']),
    '17_Lep':   ('alone', ['I_PRIM']),
    'L02_Pup':  ('both', ['V_N_R']),
    'CW_Cnc': ('alone', ['I_PRIM']),
    'Mira':  ('both',  ['CntHa_B_Ha']),
    'Pi.01_Gru':('both',  ['V_N_R']),
}


# Listes des étoiles par classe de résolution
clearly_resolved = [
    "AK_Hya", "R_Hya", "U_Her", "S_Pav", "Mira", "W_Aql", "R_Crt", "R_Leo",
    "R_Dor", "BK_Vir", "V_PsA", "SW_Col", "GY_Aql", "SW_Vir", "RT_Vir",
    "W_Hya", "L02_Pup", "W_Peg"
]

marginally_resolved = [
    "V_Hya", "BW_Oct", "Pi.01_Gru", "V1943_Sgr", "Y_Pav", "Chi_Cyg",
    "R_Scl", "U_Del", "T_Mic", "Z_Peg", "Y_Scl"
]

all_resolved = clearly_resolved + marginally_resolved

unresolved = [
    "Z_Eri", "Alpha_Her", "T_Cet", "R_Peg", "R_Hor", "AC_Cet", "R_Aql",
    "Ups_Cet", "RX_Lep", "Psi_Phe", "S_Lep", "Beta_Gru", "DZ_Aqr",
    "SV_Aqr", "17_Lep", "CW_Cnc"
]
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
folname = 'large_log_+'
fontsize_title=14
fontsize_label=14
fontsize_tick=14
fontsize_colorbar=14
def log_image(star_name, obsmod, make_azim=False):
    if star_name not in star_filters:
        print(f"{star_name} n'est pas dans le dictionnaire star_filters.")
        return

    mode_dict, filters_dict = star_filters[star_name]
    # Si le mode demandé ne correspond pas, on ignore
    if obsmod != mode_dict and mode_dict != 'both':
        print(f"{star_name} n'a pas de données pour le mode {obsmod}.")
        return

    fdir= f'/home/nbadolo/Bureau/Aymard/Donnees_sph/{folname}/'+star_name+ '/'
    fdir_star = fdir + 'star/'+obsmod+ '/'

    # Recherche des filtres contenant des données, mais seulement ceux du dictionnaire
    lst_fltr_star2 = []
    for fltr in filters_dict:
        fdir_fltr_data_star = fdir_star + fltr
        if os.path.exists(fdir_fltr_data_star):
            lst_fltr_data_star = os.listdir(fdir_fltr_data_star)
            n_lst_fltr_data_star = len(lst_fltr_data_star)
            if n_lst_fltr_data_star != 0:
                lst_fltr_star2.append(fltr)
    n_lst_fltr_star2 = len(lst_fltr_star2)
    
    
    for l in range(n_lst_fltr_star2):
        print(f"    [{l+1}/{n_lst_fltr_star2}] Filtre '{lst_fltr_star2[l]}' en cours...")
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
                  #DOLP_star = sub_v_arr[z][2]
                  DOLP_star = np.clip(sub_v_arr[z][2] - 0.004, 0, None)  # Correction instrumentale de Beuzit et al. (2019)
                  
                  ii = (sub_v_arr[z][4] == 0)
                  if True in ii:
                      sub_v_arr[z][4][ii] = sub_v_arr[z][4][ii] + 0.0001  # introduction d'un ofset pour les valeurs de Q == 0

                  Q_data = np.nan_to_num(sub_v_arr[z][5], nan=0.0, posinf=0.0, neginf=0.0)
                  U_data = np.nan_to_num(sub_v_arr[z][4], nan=0.0, posinf=0.0, neginf=0.0)
                  AOLP_2_star = 0.5 * np.arctan2(Q_data, U_data)

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
            vmin1 = 0
            vmax1 = np.max(sub_v_arr[z][2])
            im1 = ax1.imshow(sub_v_arr[z][2], cmap='plasma', origin='lower', vmin=vmin1, vmax=vmax1, extent=[x_min, x_max, y_min, y_max])
            ax1.text(0.02, 0.95, f'{star_name2}', transform=ax1.transAxes, fontsize=fontsize_label, color='white', va='top')
            ax1.text(0.02, 0.02, f'{fltr_arr[z]}', transform=ax1.transAxes, fontsize=fontsize_label, color='white', va='bottom')

            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes('right', size='5%', pad=0.03)
            cb1 = fig.colorbar(im1, cax=cax1, orientation='vertical')
            cb1.ax.tick_params(labelsize=fontsize_colorbar)

            cmapProp = {'drawedges': True}
            
            cb1.formatter.set_powerlimits((0, 0))
            cb1.ax.yaxis.get_offset_text().set(size=fontsize_colorbar)
            # for tick in cb1.ax.yaxis.get_major_ticks():
            #     tick.label2.set_fontweight('bold')

            ax1.set_xlabel('Relative RA (mas)', fontsize=fontsize_label)
            ax1.set_ylabel('Relative Dec (mas)', fontsize=fontsize_label,labelpad=1.5)
            ax1.tick_params(axis='both', labelsize=fontsize_tick, width=1.2)
            # for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            #     label.set_fontweight('bold')
            ax1.locator_params(axis='x', nbins=5)
            ax1.locator_params(axis='y', nbins=5)

            # --- Image log10(PI) ---

            im2 = ax2.imshow(np.log10(sub_v_arr[z][1] + np.abs(np.min(sub_v_arr[z][1])) + 10), 
                cmap='inferno', origin='lower', extent=[x_min, x_max, y_min, y_max]
            )

            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes('right', size='5%', pad=0.03)
            cb2 = fig.colorbar(im2, cax=cax2, orientation='vertical')
            cb2.ax.tick_params(labelsize=fontsize_colorbar)
            # Vecteurs de polarisation
            # q = ax2.quiver(
            #     X[::X_step, ::X_step], Y[::X_step, ::X_step],
            #     U2[::X_step, ::X_step], V2[::X_step, ::X_step],
            #     color='w', pivot='mid'
            # )
            #ax2.quiverkey(q, X=0.1, Y=1.03, U=0, label='', labelpos='E')
            ax2.text(0.02, 0.95, f'{star_name2}', transform=ax2.transAxes, fontsize=fontsize_label, color='white', va='top')
            cb2.formatter.set_powerlimits((0, 0))
            #cb2.ax.yaxis.get_offset_text().set(size=fontsize_colorbar)
            #for tick in cb2.ax.yaxis.get_major_ticks():
            #    tick.label2.set_fontweight('bold')

            ax2.set_xlabel('Relative RA (mas)', fontsize=fontsize_label)
            #ax2.set_ylabel('Relative Dec (mas)', fontsize=fontsize_label, weight='bold',labelpad=1)
            ax2.tick_params(axis='both', labelsize=fontsize_tick, width=1.2)
            # for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            #     label.set_fontweight('bold')
            ax2.locator_params(axis='x', nbins=5)
            ax2.locator_params(axis='y', nbins=5)
            ax2.axes.yaxis.set_ticklabels([])  # Pas de labels y sur la 2e image

            plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10, wspace=0.15, hspace=0.35)
            #PI_png='/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/gallery/Pol/PIL_png/'
            PI_png='/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/PI_for_these/'
            PI_save_dir = f'/home/nbadolo/Bureau/Aymard/Donnees_sph/{folname}/' + star_name + '/plots/no_psf/PI/'
            if not os.path.exists(PI_save_dir):
                os.makedirs(PI_save_dir)
            if not os.path.exists(PI_png):
                os.makedirs(PI_png)
            # plt.savefig(
            #     PI_save_dir + star_name + '_' + f'{obsmod}_{fltr_arr[z]}_{z}' + '_log_vect.pdf',
            #     dpi=300, bbox_inches='tight', pad_inches=0.01
            # )
            # plt.savefig(
            #     PI_save_dir + star_name + '_' + f'{obsmod}_{fltr_arr[z]}_{z}' + '_log_vect.png',
            #     dpi=300, bbox_inches='tight', pad_inches=0.01
            # )
            plt.savefig(
                PI_png + star_name + '_' + f'{obsmod}_{fltr_arr[z]}_{z}' + '_log_vect.png',
                dpi=300, bbox_inches='tight', pad_inches=0.01
            )
            plt.savefig(
                PI_png + star_name + '_' + f'{obsmod}_{fltr_arr[z]}_{z}' + '_log_vect.pdf',
                dpi=300, bbox_inches='tight', pad_inches=0.01
            )
            # plt.savefig(
            #     PI_save_dir + star_name + '_' + f'{obsmod}_{fltr_arr[z]}_{z}' + '_log_vect.eps',
            #     format='eps', dpi=300, bbox_inches='tight', pad_inches=0.01
            # )
            #plt.show()
            #plt.close(fig)
            # === Bloc supplémentaire : PIL et DoLP azimutaux (Schmid/de Boer) ===
            # Q = sub_v_arr[z][5], U = sub_v_arr[z][4], I = sub_v_arr[z][0]
            if make_azim:
                # --- Calcul de DoLP azimutal et PIL azimutal ---
                Q = sub_v_arr[z][5]
                U = sub_v_arr[z][4]
                I = sub_v_arr[z][0]
                ny, nx = Q.shape
                Y, X = np.indices((ny, nx))
                x0 = ny // 2
                y0 = nx // 2
                phi0 = 0  # offset en radians

                phi = np.arctan2(x0 - X, Y - y0) + phi0
                Q_phi = -Q * np.cos(2 * phi) - U * np.sin(2 * phi)
                U_phi =  Q * np.sin(2 * phi) - U * np.cos(2 * phi)
                PIL = np.abs(Q_phi)
                #DoLP_az = PIL / (I + 1e-10)
                DoLP_az = np.clip(PIL / (I + 1e-10) - 0.004, 0, None) # Correction instrumentale de Beuzit et al. (2019)
                
                # --- Création des figures pour DoLP azimutal et PIL azimutal ---
                fig2, (ax4, ax3) = plt.subplots(1, 2, figsize=(12, 5))  # DoLP à gauche, PIL à droite

                # --- DoLP azimutal (à gauche) ---
                im4 = ax4.imshow(DoLP_az, cmap='inferno', origin='lower',
                                extent=[x_min, x_max, y_min, y_max],
                                vmin=np.min(DoLP_az), vmax=np.max(DoLP_az))
                ax4.text(0.02, 0.95, f'{star_name2}', transform=ax4.transAxes, fontsize=fontsize_label, color='white', va='top')
                ax4.text(0.02, 0.02, f'{fltr_arr[z]}', transform=ax4.transAxes, fontsize=fontsize_label, color='white', va='bottom')
                divider4 = make_axes_locatable(ax4)
                cax4 = divider4.append_axes('right', size='5%', pad=0.03)
                cb4 = fig2.colorbar(im4, cax=cax4, orientation='vertical')
                cb4.ax.tick_params(labelsize=fontsize_colorbar)
                #cb4.formatter = ScalarFormatter(useMathText=True)
                cb4.formatter.set_powerlimits((0, 0))
                cb4.update_ticks()
                cb4.ax.yaxis.get_offset_text().set(size=fontsize_colorbar)
                #cb4.set_ticks(np.linspace(np.nanmin(DoLP_az), np.nanmax(DoLP_az), 6))
                # cb4.ax.set_ylabel('DoLP (azimutal)', fontsize=11, weight='bold')
                # ax4.set_title('DoLP azimutal (PIL/I)', fontsize=12, weight='bold')
                ax4.set_xlabel('Relative RA (mas)', fontsize=fontsize_label)
                ax4.set_ylabel('Relative Dec (mas)', fontsize=fontsize_label, labelpad=1.5)
                ax4.tick_params(axis='both', labelsize=fontsize_tick, width=1.2)
                # Limite le nombre de ticks sur les axes
                # for label in ax4.get_xticklabels() + ax4.get_yticklabels():
                #     label.set_fontweight('bold')
                ax4.locator_params(axis='x', nbins=5)
                ax4.locator_params(axis='y', nbins=5)
                # # --- PIL azimutal (à droite) ---
                # im3 = ax3.imshow(PIL, cmap='inferno', origin='lower',
                #                 extent=[x_min, x_max, y_min, y_max])
                # ax3.text(0.02, 0.95, f'{star_name2}', transform=ax3.transAxes, fontsize=12, fontweight='bold', color='white', va='top')
                # divider3 = make_axes_locatable(ax3)
                # cax3 = divider3.append_axes('right', size='3%', pad=0.02)
                # cb3 = fig2.colorbar(im3, cax=cax3, orientation='vertical')
                # cb3.ax.tick_params(labelsize=11)
                # #cb3.formatter = ScalarFormatter(useMathText=True)
                # cb3.formatter.set_powerlimits((0, 0))
                # cb3.update_ticks()
                # cb3.ax.yaxis.get_offset_text().set(size=11)
                # #cb3.set_ticks(np.linspace(np.nanmin(PIL), np.nanmax(PIL), 6))
                # # cb3.ax.set_ylabel('PIL (azimutal)', fontsize=11, weight='bold')
                # PIL azimutal (à droite, en échelle log)

                offset = 1e-10 # pour éviter les problèmes de log(0)
                logPIL = np.log10(PIL + offset)
                pil_vmin = np.nanmin(logPIL)
                pil_vmax = np.nanmax(logPIL)

                # forcer la colorbar à commencer à 0 (PIL=1):
                vmin = 0
                vmax = pil_vmax

                im3 = ax3.imshow(logPIL, cmap='inferno', origin='lower',
                                extent=[x_min, x_max, y_min, y_max],
                                vmin=vmin, vmax=vmax)
                ax3.text(0.02, 0.95, f'{star_name2}', transform=ax3.transAxes, fontsize=fontsize_label, color='white', va='top')
                divider3 = make_axes_locatable(ax3)
                cax3 = divider3.append_axes('right', size='5%', pad=0.03)
                cb3 = fig2.colorbar(im3, cax=cax3, orientation='vertical')
                cb3.ax.tick_params(labelsize=fontsize_colorbar)
                #cb3.formatter = ScalarFormatter(useMathText=True)
                cb3.formatter.set_powerlimits((0, 0))
                ticks_pil = np.linspace(vmin, vmax, 5)
                cb3.set_ticks(ticks_pil)
                cb3.ax.set_yticklabels([f"{tick:.1f}" for tick in ticks_pil])
                #cb3.update_ticks()
                cb3.ax.yaxis.get_offset_text().set(size=fontsize_colorbar)
                # cb3.ax.set_ylabel('log$_{10}$(PIL$_{\\phi}$)', fontsize=11, weight='bold')
                # ax3.set_title('log$_{10}$(PIL azimutal)', fontsize=12, weight='bold')
                # ax3.set_title('PIL azimutal (|Q_phi|)', fontsize=12, weight='bold')
                ax3.set_xlabel('Relative RA (mas)', fontsize=fontsize_label)
                #ax3.set_ylabel('Relative Dec (mas)', fontsize=fontsize_label, weight='bold')
                ax3.tick_params(axis='both', labelsize=fontsize_tick, width=1.2)
                # Limite le nombre de ticks sur les axes
                # for label in ax3.get_xticklabels() + ax3.get_yticklabels():
                #     label.set_fontweight('bold')
                ax3.locator_params(axis='x', nbins=5)
                ax3.locator_params(axis='y', nbins=5)
                print(f"      → Sauvegarde des figures pour {star_name}, filtre {fltr}, image {z}")
                plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10, wspace=0.15, hspace=0.35)
                Azim_save_dir = f'/home/nbadolo/Bureau/Aymard/Donnees_sph/{folname}/' + star_name + '/plots/no_psf/Azim/'
                if not os.path.exists(Azim_save_dir):
                    os.makedirs(Azim_save_dir)
                # Azim_png_dir='/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/gallery/Pol/Azim_png/'
                Azim_png_dir='/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Azim_for_these/'
                if not os.path.exists(Azim_png_dir):
                    os.makedirs(Azim_png_dir)
                # plt.savefig(
                #     Azim_save_dir + f'{star_name}_{obsmod}_{fltr_arr[z]}_{z}_azimutal.pdf',
                #     dpi=300, bbox_inches='tight', pad_inches=0.01
                # )
                # plt.savefig(
                #     Azim_save_dir + f'{star_name}_{obsmod}_{fltr_arr[z]}_{z}_azimutal.png',
                #     dpi=300, bbox_inches='tight', pad_inches=0.01
                # )
                plt.savefig(
                    Azim_png_dir + f'{star_name}_{fltr_arr[z]}_{z}_azimutal.png',
                    dpi=300, bbox_inches='tight', pad_inches=0.01
                )
                plt.savefig(
                    Azim_png_dir + f'{star_name}_{fltr_arr[z]}_{z}_azimutal.pdf',
                    dpi=300, bbox_inches='tight', pad_inches=0.01
                )
                # plt.savefig(
                #     Azim_save_dir + f'{star_name}_{obsmod}_{fltr_arr[z]}_{z}_azimutal.eps',
                #     format='eps', dpi=300, bbox_inches='tight', pad_inches=0.01
                # )
                #plt.show()
                #plt.close(fig2)
            print(f"  → Fin du traitement pour {star_name}\n")
    print("\nTraitement terminé pour toutes les étoiles.\nLes figures sont sauvegardées dans :\n  - /home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/I_with_psf_for_these/\n  - /home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/PI_for_these/", flush=True)
    if make_azim:
        return fig, fig2
    else:
        return fig, None

# star=log_image('V854_Cen', 'alone')
# star=log_image('V854_Cen', 'both')
# Traitement principal
total_stars = len(star_filters)
for idx, (star, (mode, filters)) in enumerate(star_filters.items(), start=1):
    print(f"[{idx}/{total_stars}] Traitement étoile : {star} | Mode : {mode} | Filtres : {filters}")
    make_azim = star in unresolved
    figs = log_image(star, mode, make_azim=make_azim)
    if figs is not None:
        fig, fig2 = figs
        if make_azim:
            print(f"Figures PI et azimutale générées pour {star}")
        else:
            print(f"Figure PI générée pour {star}")



# # Cartes PIL pour toutes les étoiles
# total_pil = len(all_resolved + unresolved)
# for idx, star in enumerate(all_resolved + unresolved, start=1):
#     mode, filters = star_filters[star]
#     fig, _ = log_image(star, mode, make_azim=False)
#     print(f"[{idx}/{total_pil}] Carte PIL générée pour {star}")

# # Cartes azimutales pour les non résolues
# total_azim = len(unresolved)
# for idx, star in enumerate(unresolved, start=1):
#     mode, filters = star_filters[star]
#     _, fig2 = log_image(star, mode, make_azim=True)
#     print(f"[{idx}/{total_azim}] Carte azimutale générée pour {star}")

# # Fonction pour afficher une galerie d'images à partir d'un dossier
# def show_gallery_from_files(image_folder, title=''):
#     # Format A&A pleine page
#     width_cm = 17.8
#     height_cm = 24
#     dpi = 300
#     n_rows = 7
#     n_cols = 2
#     figsize = (width_cm / 2.54, height_cm / 2.54)  # conversion cm -> pouces

#     image_files = sorted(glob.glob(os.path.join(image_folder, "**", "*.png"), recursive=True))
#     images_per_page = n_rows * n_cols
#     total = len(image_files)
#     page = 0

#     for start in range(0, total, images_per_page):
#         page += 1
#         end = min(start + images_per_page, total)
#         fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
#         axes = axes.flatten()
#         for i, img_path in enumerate(image_files[start:end]):
#             img = mpimg.imread(img_path)
#             axes[i].imshow(img)
#             axes[i].axis('off')
#         for i in range(end-start, images_per_page):
#             axes[i].axis('off')
#         plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.15, hspace=0.01)
#         gallery_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/gallery/Pol/'
#         if not os.path.exists(gallery_dir):
#             os.makedirs(gallery_dir)
#         fig.savefig(f"{gallery_dir}png/{title.replace(' ', '_')}_page_{page}.png", bbox_inches='tight', dpi=dpi)
#         fig.savefig(f"{gallery_dir}pdf/{title.replace(' ', '_')}_page_{page}.pdf", bbox_inches='tight', dpi=dpi)
#         fig.savefig(f"{gallery_dir}eps/{title.replace(' ', '_')}_page_{page}.eps", bbox_inches='tight', dpi=dpi)
#         plt.close(fig)

# # Galerie PI
# show_gallery_from_files(
#     '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/gallery/Pol/PIL_png/',
#     title='PI_gallery'
# )

# # Galerie azimutale
# show_gallery_from_files(
#     '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/gallery/Pol/Azim_png/',
#     title='Azim_gallery'
# )

