#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse des profils radiaux à orientation donnée pour étoile et PSF.
Traitement d'images FITS, extraction de sous-images, seuillage, ajustement d'ellipse,
calcul et tracé des profils radiaux, sauvegarde des figures.
"""

import os
import numpy as np
from numpy import nan
from astropy.io import fits
from astropy.nddata import Cutout2D
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.colors as colors
from skimage import measure, draw
import scipy.optimize as opt
from math import pi, cos, sin
from AymardPack import Margaux_RL_deconv 
from AymardPack import EllRadialProf as erp
from AymardPack import DelHotPix

# === PARAMÈTRES GLOBAUX ===
nDim = 1024
nSubDim = 100  # Taille de la sous-image extraite
size = (nSubDim, nSubDim)
lst_threshold = [0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]
n_threshold = len(lst_threshold)
pix2mas = 3.4  # Conversion pixel -> mas
position = (nDim // 2, nDim // 2)

# === DOSSIERS DE SORTIE POUR LES LOGS ===
txt_folder = 'sphere_txt_file'
file_path = '/home/nbadolo/Bureau/Aymard/Donnees_sph/' + txt_folder + '/'
file_name = 'no_common_data_lst.txt'
os.makedirs(file_path, exist_ok=True)
no_common_data_lst = open(os.path.join(file_path, file_name), "w")
no_common_data_lst.write("Star name, Mode\n")

def log_image(star_name, obsmod):
    """
    Traite les images d'une étoile et de sa PSF pour extraire et comparer les profils radiaux.
    Sauvegarde les figures dans les dossiers appropriés.
    """
    # === DÉFINITION DES CHEMINS ===
    base_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log_+/'
    star_dir = os.path.join(base_dir, star_name, 'star', obsmod)
    psf_dir = os.path.join(base_dir, star_name, 'psf', obsmod)

    # === LISTE DES FILTRES DISPONIBLES ===
    print("star_dir:", star_dir)
    print("psf_dir:", psf_dir)
    print("star_dir content:", os.listdir(star_dir))
    print("psf_dir content:", os.listdir(psf_dir))
    lst_fltr2_star = [f for f in os.listdir(star_dir) if os.path.isdir(os.path.join(star_dir, f))]
    lst_fltr2_psf = [f for f in os.listdir(psf_dir) if os.path.isdir(os.path.join(psf_dir, f))]
    print("Filtres star:", lst_fltr2_star)
    print("Filtres psf:", lst_fltr2_psf)
    
    lst_fltr3 = list(set(lst_fltr2_star).intersection(lst_fltr2_psf))
    print("Filtres communs:", lst_fltr3)
    n_lst_fltr3 = len(lst_fltr3)

    if n_lst_fltr3 == 0:
        print(f'No common data for {star_name} and its psf')
        no_common_data_lst.write(f"{star_name},{obsmod}\n")
        return (star_name, obsmod)

    for l in range(n_lst_fltr3):
        # === CHEMINS DES DONNÉES ===
        star_fltr_dir = os.path.join(star_dir, lst_fltr3[l])
        psf_fltr_dir = os.path.join(psf_dir, lst_fltr3[l])

        # === NOMS DES FICHIERS FITS ===
        fname1 = 'zpl_p23_make_polar_maps-ZPL_SCIENCE_P23_REDUCED'
        fname2 = '-zpl_science_p23_REDUCED'
        file_I_star = os.path.join(star_fltr_dir, f"{fname1}_I{fname2}_I.fits")
        file_PI_star = os.path.join(star_fltr_dir, f"{fname1}_PI{fname2}_PI.fits")
        file_I_psf = os.path.join(psf_fltr_dir, f"{fname1}_I{fname2}_I.fits")
        file_PI_psf = os.path.join(psf_fltr_dir, f"{fname1}_PI{fname2}_PI.fits")

        file_star_lst = [file_I_star, file_PI_star]
        file_psf_lst = [file_I_psf, file_PI_psf]
        nFrames_star = len(file_star_lst)
        nFrames_psf = len(file_psf_lst)

        # === INITIALISATION DES TABLEAUX ===
        sub_v_star_arr = np.zeros((nFrames_star, nSubDim, nSubDim))
        sub_v_psf_arr = np.zeros((nFrames_psf, nSubDim, nSubDim))
        par_star_arr = np.zeros((nFrames_star, n_threshold, 5))
        par_psf_arr = np.zeros((nFrames_psf, n_threshold, 5))
        Ell_rot_star_arr = np.zeros((nFrames_star, n_threshold, 2, nSubDim))
        Ell_rot_psf_arr = np.zeros((nFrames_psf, n_threshold, 2, nSubDim))

        # === TRAITEMENT DES IMAGES ÉTOILE ===
        for str_i, file_star in enumerate(file_star_lst):
            if not os.path.exists(file_star):
                print(f"Fichier manquant (star): {file_star}")
                continue
            hdu_str = DelHotPix(file_star)
            data_str = hdu_str[0].data
            intensity_str = data_str[1, :, :]
            cutout_str = Cutout2D(intensity_str, position=position, size=size)
            sub_v_star = cutout_str.data
            sub_v_star_arr[str_i] = sub_v_star

            for str_j, threshold in enumerate(lst_threshold):
                # Seuillage et création de l'image binaire
                Ellips_star = (sub_v_star > threshold * np.max(sub_v_star)).astype(float)
                im_star_white = Ellips_star

                # Extraction des propriétés de la région la plus brillante
                regions_str = measure.regionprops(measure.label(im_star_white))
                if not regions_str:
                    continue
                bubble_str = regions_str[0]
                ys_i, xs_i = bubble_str.centroid
                as_i = bubble_str.major_axis_length / 2.
                bs_i = bubble_str.minor_axis_length / 2.
                thetas_i = pi / 4
                t = np.linspace(0, 2 * pi, nSubDim)

                # Ajustement de l'ellipse par optimisation
                def cost(params_s):
                    x0s, y0s, a0s, b0s, thetas = params_s
                    coords_s = draw.ellipse(y0s, x0s, a0s, b0s, shape=None, rotation=thetas)
                    template_star = np.zeros_like(im_star_white)
                    template_star[coords_s] = 1
                    return -np.sum(template_star == im_star_white)

                x_sf, y_sf, a_sf, b_sf, theta_sf = opt.fmin(cost, (xs_i, ys_i, as_i, bs_i, thetas_i), disp=False)
                theta_sf = np.pi / 2 - theta_sf
                par_star_arr[str_i][str_j] = [x_sf, y_sf, a_sf, b_sf, theta_sf]

                # Calcul des coordonnées de l'ellipse ajustée
                Ell_star = np.array([a_sf * np.cos(t), b_sf * np.sin(t)])
                M_rot_star = np.array([[cos(theta_sf), -sin(theta_sf)], [sin(theta_sf), cos(theta_sf)]])
                Ell_rot_star = np.dot(M_rot_star, Ell_star)
                Ell_rot_star[0, :] += x_sf
                Ell_rot_star[1, :] += y_sf
                Ell_rot_star_arr[str_i][str_j] = Ell_rot_star

        # === TRAITEMENT DES IMAGES PSF ===
        for psf_i, file_psf in enumerate(file_psf_lst):
            if not os.path.exists(file_psf):
                print(f"Fichier manquant (psf): {file_psf}")
                continue
            hdu_psf = DelHotPix(file_psf)
            data_psf = hdu_psf[0].data
            intensity_psf = data_psf[0, :, :]
            cutout_psf = Cutout2D(intensity_psf, position=position, size=size)
            sub_v_psf = cutout_psf.data
            sub_v_psf_arr[psf_i] = sub_v_psf

            for psf_j, threshold in enumerate(lst_threshold):
                Ellips_psf = (sub_v_psf > threshold * np.max(sub_v_psf)).astype(float)
                im_psf_white = Ellips_psf

                regions_psf = measure.regionprops(measure.label(im_psf_white))
                if not regions_psf:
                    continue
                bubble_psf = regions_psf[0]
                yp_i, xp_i = bubble_psf.centroid
                ap_i = bubble_psf.major_axis_length / 2.
                bp_i = bubble_psf.minor_axis_length / 2.
                thetap_i = pi / 4
                t = np.linspace(0, 2 * pi, nSubDim)

                def cost(params_p):
                    x0p, y0p, ap, bp, thetap = params_p
                    coords_p = draw.ellipse(y0p, x0p, ap, bp, shape=None, rotation=thetap)
                    template_psf = np.zeros_like(im_psf_white)
                    template_psf[coords_p] = 1
                    return -np.sum(template_psf == im_psf_white)

                x_pf, y_pf, a_pf, b_pf, theta_pf = opt.fmin(cost, (xp_i, yp_i, ap_i, bp_i, thetap_i), disp=False)
                theta_pf = np.pi / 2 - theta_pf
                par_psf_arr[psf_i][psf_j] = [x_pf, y_pf, a_pf, b_pf, theta_pf]

                Ell_psf = np.array([a_pf * np.cos(t), b_pf * np.sin(t)])
                M_rot_psf = np.array([[cos(theta_pf), -sin(theta_pf)], [sin(theta_pf), cos(theta_pf)]])
                Ell_rot_psf = np.dot(M_rot_psf, Ell_psf)
                Ell_rot_psf[0, :] += x_pf
                Ell_rot_psf[1, :] += y_pf
                Ell_rot_psf_arr[psf_i][psf_j] = Ell_rot_psf

        # === PROFILS RADIAUX ET FIGURES ===
        # Extraction des profils radiaux pour l'intensité et l'intensité polarisée
        im_s = np.log10(sub_v_star_arr[0] + np.abs(np.min(sub_v_star_arr[0])) + 10)
        imp_s = np.log10(sub_v_star_arr[1] + np.abs(np.min(sub_v_star_arr[1])) + 10)
        im_p = np.log10(sub_v_psf_arr[0] + np.abs(np.min(sub_v_psf_arr[0])) + 10)
        imp_p = np.log10(sub_v_psf_arr[1] + np.abs(np.min(sub_v_psf_arr[1])) + 10)

        # Profils radiaux (utilise la fonction erp)
        x0_s, y0_s, x1_s, y1_s, x2_s, y2_s, z_s, zi1_s, zi2_s, *_ = erp(*par_star_arr[0][0], im_s, 100)
        x0p_s, y0p_s, x1p_s, y1p_s, x2p_s, y2p_s, zp_s, zi1p_s, zi2p_s, *_ = erp(*par_star_arr[1][0], imp_s, 100)
        x0_p, y0_p, x1_p, y1_p, x2_p, y2_p, z_p, zi1_p, zi2_p, *_ = erp(*par_psf_arr[0][0], im_p, 100)
        x0p_p, y0p_p, x1p_p, y1p_p, x2p_p, y2p_p, zp_p, zi1p_p, zi2p_p, *_ = erp(*par_psf_arr[1][0], imp_p, 100)

        print("zp_s min/max:", np.min(zp_s), np.max(zp_s))
        print("zp_p min/max:", np.min(zp_p), np.max(zp_p))
        print("sub_v_star_arr[0] min/max:", np.min(sub_v_star_arr[0]), np.max(sub_v_star_arr[0]))
        print("sub_v_star_arr[1] min/max:", np.min(sub_v_star_arr[1]), np.max(sub_v_star_arr[1]))
        print("sub_v_psf_arr[0] min/max:", np.min(sub_v_psf_arr[0]), np.max(sub_v_psf_arr[0]))
        print("sub_v_psf_arr[1] min/max:", np.min(sub_v_psf_arr[1]), np.max(sub_v_psf_arr[1]))

        # === PLOTS ===
        # Vérifie si la PSF a des données valides
        # ...calcul des profils radiaux...

        # Vérifie si la PSF a des données valides
        psf_data_exists = np.any(sub_v_psf_arr[0]) and np.any(sub_v_psf_arr[1])

        plt.clf()
        if psf_data_exists:
            fig = plt.figure('profiles comparison', figsize=(18.5, 10))
            # Profils radiaux polarisés (demi-petit axe) : étoile + PSF
            ax1 = plt.subplot(212)
            ax1.plot(zi1p_s, label='star')
            ax1.plot(zi1p_p, label='psf')
            ax1.set_xlabel('r(pix)')
            ax1.set_ylabel('I (hdu)')
            ax1.set_title(f'{star_name}')
            ax1.legend()

            # Image polarisée étoile
            ax2 = plt.subplot(221)
            im2 = ax2.imshow(zp_s, cmap='inferno', vmin=np.min(zp_s), vmax=np.max(zp_s), origin='lower')
            ax2.plot(Ell_rot_star_arr[1][0][0, :], Ell_rot_star_arr[1][0][1, :])
            ax2.plot([x0p_s, x1p_s], [y0p_s, y1p_s], 'ro-')
            ax2.plot([x0p_s, x2p_s], [y0p_s, y2p_s], 'ro-')
            ax2.set_xlim(0, nSubDim)
            ax2.set_yticks([0, nSubDim / 2, nSubDim])
            ax2.set_xlabel('x(pix)', fontsize=24)
            ax2.set_ylabel('y(pix)', fontsize=24)
            plt.colorbar(im2, ax=ax2)
            ax2.set_title('star')

            # Image polarisée PSF
            ax3 = plt.subplot(222)
            im3 = ax3.imshow(zp_p, cmap='inferno', vmin=np.min(zp_p), vmax=np.max(zp_p), origin='lower')
            ax3.plot(Ell_rot_psf_arr[1][0][0, :], Ell_rot_psf_arr[1][0][1, :])
            ax3.plot([x0p_p, x1p_p], [y0p_p, y1p_p], 'ro-')
            ax3.plot([x0p_p, x2p_p], [y0p_p, y2p_p], 'ro-')
            ax3.set_xlim(0, nSubDim)
            ax3.set_yticks([0, nSubDim / 2, nSubDim])
            ax3.set_xlabel('x(pix)', fontsize=24)
            ax3.set_ylim(0, nSubDim)
            plt.colorbar(im3, ax=ax3)
            ax3.set_title('psf')
        else:
            print("Pas de données PSF valides : seul le plot étoile sera affiché.")
            fig = plt.figure('star only', figsize=(10, 8))
            ax2 = plt.gca()
            im2 = ax2.imshow(zp_s, cmap='inferno', vmin=np.min(zp_s), vmax=np.max(zp_s), origin='lower')
            ax2.plot(Ell_rot_star_arr[1][0][0, :], Ell_rot_star_arr[1][0][1, :])
            ax2.plot([x0p_s, x1p_s], [y0p_s, y1p_s], 'ro-')
            ax2.plot([x0p_s, x2p_s], [y0p_s, y2p_s], 'ro-')
            ax2.set_xlim(0, nSubDim)
            ax2.set_yticks([0, nSubDim / 2, nSubDim])
            ax2.set_xlabel('x(pix)', fontsize=24)
            ax2.set_ylabel('y(pix)', fontsize=24)
            plt.colorbar(im2, ax=ax2)
            ax2.set_title('star')

        plt.tight_layout()
        plt.show()

# Exemple d'appel :
star_name = 'SW_Col'
obsmod = 'both'
log_image(star_name, obsmod)