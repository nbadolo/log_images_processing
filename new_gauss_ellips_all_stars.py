#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Python pour l’analyse morphologique automatique d’images polarisées d’étoiles.

Ce code :
- Parcourt tous les dossiers d’étoiles dans un répertoire donné.
- Pour chaque étoile, parcourt tous les filtres disponibles (si applicable).
- Ouvre les images FITS de polarisation (PI).
- Extrait une sous-image centrée sur l’étoile.
- Cherche automatiquement le meilleur seuil pour segmenter la région principale.
- Calcule les propriétés morphologiques de la région (ellipse de meilleure correspondance).
- Fait un fit automatique de l’ellipse (centre, axes, angle) en maximisant le Dice coefficient.
- Affiche et sauvegarde pour chaque image :
    - L’image log(PI) avec le contour de l’ellipse ajustée et le centre.
    - Le profil radial moyen normalisé avec le seuil optimal (contraste).
- Compile les résultats morphologiques dans un DataFrame global pour toutes les étoiles et les sauvegarde dans un seul CSV.
"""

import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['figure.max_open_warning'] = 0  # Désactive l'avertissement
matplotlib.use('Agg')  # Mode non-interactif pour éviter l'accumulation de figures
from math import pi, cos, sin
from astropy.nddata import Cutout2D
from astropy.io import fits
import scipy.optimize as opt
from skimage import measure, draw
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

def radial_profile(image):
    """Calcule le profil radial moyen d'une image 2D en fonction du pixel le plus brillant."""
    y, x = np.indices(image.shape)
    max_intensity_index = np.unravel_index(np.argmax(image), image.shape)
    center_y, center_x = max_intensity_index
    center = np.array([center_x, center_y])
    radius = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    radius = radius.astype(int)
    radial_sum = np.bincount(radius.ravel(), weights=image.ravel())
    radial_count = np.bincount(radius.ravel())
    radial_mean = radial_sum / radial_count
    nans = np.isnan(radial_mean)
    not_nans = ~nans
    radial_mean[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), radial_mean[not_nans])
    return radial_mean

def log_image(folder_name, star_name, obsmod):
    """
    Analyse morphologique automatique pour une étoile donnée.
    Retourne une liste de dictionnaires de résultats pour concaténation globale.
    """
    # Répertoires et paramètres
    # Répertoires et paramètres
    fdir = f'/home/nbadolo/Bureau/Aymard/Donnees_sph/{folder_name}/{star_name}/'
    fdir_star = os.path.join(fdir, obsmod)
    fits_files = [f for f in os.listdir(fdir_star) if f.endswith('.fits')]
    print(f"Fichiers FITS trouvés pour {star_name} :", fits_files)

    # Paramètres globaux
    nDim = 1024
    nSubDim = 100
    size = (nSubDim, nSubDim)
    lst_threshold = np.linspace(0.001, 0.1, 100)  # 100 seuils de 0.1% à 10%
    pix2mas = 3.4
    position = (nDim // 2, nDim // 2)
    x_min = -pix2mas * nSubDim // 2
    x_max = pix2mas * (nSubDim // 2 - 1)
    y_min = -pix2mas * nSubDim // 2
    y_max = pix2mas * (nSubDim // 2 - 1)

    results = []

    for fits_file in fits_files:
        file_PI_star = os.path.join(fdir_star, fits_file)
        outdir = os.path.join(fdir, "plots/fits/log_scale/fully_automatic/")
        os.makedirs(outdir, exist_ok=True)
        
        if not os.path.exists(file_PI_star):
            print(f"Fichier manquant : {file_PI_star}")
            continue

        hdu = fits.open(file_PI_star)
        data = hdu[0].data
        header = hdu[0].header
        star_name2 = header.get('OBJECT', star_name)
        fltr1 = header.get('HIERARCH ESO INS3 OPTI5 NAME', 'Filtre1 inconnu')
        fltr2 = header.get('HIERARCH ESO INS3 OPTI6 NAME', 'Filtre2 inconnu')
        fltr_arr = [fltr1, fltr2]
        n_fsize = data.shape[0]  # nombre de plans dans le cube (souvent 2)

        for z in range(n_fsize):
            intensity = data[z, :, :]
            cutout = Cutout2D(intensity, position=position, size=size)
            sub_v = cutout.data

            best_cost = 1.0
            best_threshold = None
            best_params = None

            # Boucle automatique sur tous les seuils de lst_threshold
            for threshold in lst_threshold:
                Ellips = np.zeros_like(sub_v)
                Ellips[sub_v > threshold * np.max(sub_v)] = 1
                regions = measure.regionprops(measure.label(Ellips))
                if not regions:
                    continue

                max_pos = np.unravel_index(np.argmax(sub_v), sub_v.shape)
                region_max = None
                for region in regions:
                    if region.coords is not None and any(np.array_equal(max_pos, coord) for coord in region.coords):
                        region_max = region
                        break
                if region_max is None:
                    region_max = regions[0]

                y_i, x_i = region_max.centroid
                a_i = region_max.major_axis_length / 2.
                b_i = region_max.minor_axis_length / 2.
                theta_i = pi / 4
                t = np.linspace(0, 2 * pi, nSubDim)
                def cost(params):
                    x0, y0, a, b, theta = params
                    a = min(a, nSubDim/2 - 2)
                    b = min(b, nSubDim/2 - 2)
                    try:
                        coords = draw.ellipse(y0, x0, a, b, shape=Ellips.shape, rotation=theta)
                        template = np.zeros_like(Ellips)
                        template[coords] = 1
                        intersection = np.sum((template == 1) & (Ellips == 1))
                        size_sum = np.sum(template) + np.sum(Ellips)
                        dice = 2 * intersection / size_sum if size_sum > 0 else 0
                        return 1 - dice
                    except Exception as e:
                        print(f"Erreur draw.ellipse pour nSubDim={nSubDim}, a={a}, b={b}: {e}")
                        return 1

                x_f, y_f, a_f, b_f, theta_f = opt.fmin(cost, (x_i, y_i, a_i, b_i, theta_i), disp=False)
                fit_cost = cost([x_f, y_f, a_f, b_f, theta_f])

                if fit_cost < best_cost:
                    best_cost = fit_cost
                    best_threshold = threshold
                    best_params = (x_f, y_f, a_f, b_f, theta_f)

            if best_params is None:
                continue

            x_f, y_f, a_f, b_f, theta_f = best_params
            t = np.linspace(0, 2 * pi, nSubDim)
            Ell = np.array([a_f * np.cos(t), b_f * np.sin(t)])
            theta_f = np.pi / 2 - theta_f
            M_rot = np.array([[cos(theta_f), -sin(theta_f)], [sin(theta_f), cos(theta_f)]])
            Ell_rot = np.dot(M_rot, Ell)
            Ell_rot[0, :] += x_f
            Ell_rot[1, :] += y_f

            nSubDim = sub_v.shape[0]
            x_mas = (np.arange(nSubDim) - nSubDim // 2) * pix2mas
            y_mas = (np.arange(nSubDim) - nSubDim // 2) * pix2mas
            x_contour_mas = (Ell_rot[0, :] - nSubDim // 2) * pix2mas
            y_contour_mas = (Ell_rot[1, :] - nSubDim // 2) * pix2mas
            x_centroid_mas = (x_f - nSubDim // 2) * pix2mas
            y_centroid_mas = (y_f - nSubDim // 2) * pix2mas
            diameter_mas = 2 * a_f * pix2mas
            diameter_minor_mas = 2 * b_f * pix2mas
            diameter_err_mas = 2 * pix2mas  # erreur simple : ±1 pixel sur chaque demi-axe
            diameter_minor_err_mas = 2 * pix2mas

            # Ajout du résultat pour cette image
            results.append({
                'star': star_name,
                'filter': fltr_arr[z],
                'frame_type': f'Pol_Intensity_{z}',
                'diameter_major_mas': diameter_mas,
                'diameter_major_err_mas': diameter_err_mas,
                'diameter_minor_mas': diameter_minor_mas,
                'diameter_minor_err_mas': diameter_minor_err_mas,
                'axis_ratio': a_f / b_f,
                'center_x_mas': x_centroid_mas,
                'center_y_mas': y_centroid_mas,
                'theta_deg': np.degrees(theta_f),
                'fit_cost': best_cost,
                'contrast': best_threshold
            })
            print(f"Traitement : filtre={fltr_arr[z]}, seuil={best_threshold:.4f}, fit_cost={best_cost:.4f}")

            # Affichage et sauvegarde des figures
            # Image log(PI) + ellipse
            plt.figure(figsize=(6, 5))
            ax = plt.gca()
            im = ax.imshow(
                np.log10(sub_v + np.abs(np.min(sub_v)) + 10),
                cmap='inferno',
                origin='lower',
                extent=[x_min+1, x_max, y_min+1, y_max]
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Log$_{10}$(PI)', fontsize=14, weight='normal')
            ax.plot(x_contour_mas, y_contour_mas, color='cyan', linewidth=2, linestyle='--')
            ax.scatter([x_centroid_mas], [y_centroid_mas], color='red', marker='x')
            ax.set_xlabel("Relative RA (mas)", fontsize=14, weight='normal')
            ax.set_ylabel("Relative Dec (mas)", fontsize=14, weight='normal')
            ax.tick_params(axis='both', labelsize=14, width=1.2)
            ax.locator_params(axis='x', nbins=5)
            ax.locator_params(axis='y', nbins=5)
            ax.text(0.02, 0.95, f'{star_name2}', transform=ax.transAxes, fontsize=14, color='white', va='top', weight='normal')
            ax.text(0.02, 0.02, f'{fltr_arr[z]}', transform=ax.transAxes, fontsize=14, color='white', va='bottom', weight='normal')
            plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10)
            plt.savefig(os.path.join(outdir, f'unique_max_contour_for_Pol_Intensity_{star_name}_{obsmod}_{fltr_arr[z]}_{z}.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # Profil radial moyen normalisé
            profile = radial_profile(sub_v)
            profile_norm = profile / np.max(profile)
            r_pix = np.arange(len(profile_norm))
            r_mas = r_pix * pix2mas
            plt.figure(figsize=(6, 5))
            plt.plot(r_mas, profile_norm, color='#1f77b4', lw=2, label='Radial profile')
            plt.axhline(best_threshold, color='#d62728', ls='--', lw=2, label=f'$h$ (contrast) = {best_threshold:.3f}')
            plt.xlabel('Separation (mas)', fontsize=14, weight='normal')
            plt.ylabel('Normalized PI', fontsize=14, weight='normal')
            #plt.title(f'{star_name2} - {fltr_arr[z]}', fontsize=14)
            plt.tick_params(axis='both', labelsize=14, width=1.2)
            plt.legend(fontsize=14, loc='upper right')
            plt.xlim(0, diameter_mas / 2)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f'profile_radial_PI_{star_name}_{obsmod}_{fltr_arr[z]}_{z}.png'), dpi=300, bbox_inches='tight')
            plt.close()

    # Retourne la liste des résultats pour concaténation globale
    return results

def process_all_stars(folder_name, obsmod):
    """
    Parcourt tous les sous-dossiers (étoiles) du dossier principal,
    appelle log_image pour chaque étoile, concatène tous les résultats
    et sauvegarde un unique CSV global.
    """
    base_dir = f"/home/nbadolo/Bureau/Aymard/Donnees_sph/{folder_name}/"
    
    all_results = []
    for star_name in os.listdir(base_dir):
        star_path = os.path.join(base_dir, star_name)
        star_obs_dir = os.path.join(star_path, obsmod)
        if os.path.isdir(star_obs_dir):
            print(f"Traitement de l'étoile : {star_name}")
            try:
                results = log_image(folder_name, star_name, obsmod)
                if results:
                    all_results.extend(results)
            except Exception as e:
                print(f"Erreur pour {star_name} : {e}")
    # Sauvegarde du DataFrame global
    if all_results:
        df_all = pd.DataFrame(all_results)
        outdir = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian/Output/large_log_+/Csv"
        os.makedirs(outdir, exist_ok=True)
        csv_path = os.path.join(outdir, f'morpho_results_all_{obsmod}.csv')
        df_all.to_csv(csv_path, index=False)
        print(f"Résultats morphologiques globaux sauvegardés dans : {csv_path}")

# Exemple d'appel batch
if __name__ == "__main__":
    process_all_stars("Gaussian/Input/large_log_+", "Pol_intensity")