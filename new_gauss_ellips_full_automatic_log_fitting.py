#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:45:23 2022

@author: nbadolo
"""

"""
Script Python pour l’analyse morphologique automatique d’images polarisées d’étoiles.

Ce code :
- Parcourt les dossiers d’observations pour chaque étoile et chaque filtre.
- Ouvre les images FITS de polarisation (PI).
- Extrait une sous-image centrée sur l’étoile.
- Cherche automatiquement le meilleur seuil pour segmenter la région principale.
- Calcule les propriétés morphologiques de la région (ellipse de meilleure correspondance).
- Fait un fit automatique de l’ellipse (centre, axes, angle) en maximisant le Dice coefficient.
- Gère les cas où l’ellipse sortirait du cadre (robuste pour petits nSubDim).
- Affiche et sauvegarde pour chaque image :
    - L’image log(PI) avec le contour de l’ellipse ajustée et le centre.
    - Les axes, labels, colorbar et annotations homogènes pour publication.
- Compile les résultats morphologiques dans un fichier CSV pour chaque étoile et mode d’observation.

Ce script est conçu pour produire des figures et des mesures directement exploitables pour la publication scientifique.
"""

# Importation des bibliothèques nécessaires
import numpy as np
import os
from matplotlib import pyplot as plt
from math import pi, cos, sin
from astropy.nddata import Cutout2D
from astropy.io import fits
import scipy.optimize as opt
from skimage import measure, draw
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Fonction principale pour l'analyse morphologique des images polarisées
def log_image(folder_name, star_name, obsmod):
    # Répertoires et paramètres
    fdir = f'/home/nbadolo/Bureau/Aymard/Donnees_sph/{folder_name}/{star_name}/'
    fdir_star = fdir + 'star/' + obsmod + '/'
    lst_fltr_star1 = [d for d in os.listdir(fdir_star) if os.path.isdir(os.path.join(fdir_star, d))]
    lst_fltr_star2 = []
    for fltr in lst_fltr_star1:
        if len(os.listdir(os.path.join(fdir_star, fltr))) > 0:
            lst_fltr_star2.append(fltr)
    print("Filtres trouvés :", lst_fltr_star2)

    # Paramètres globaux
    nDim = 1024
    nSubDim = 100
    size = (nSubDim, nSubDim)
    #lst_threshold = [0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]
    lst_threshold = np.linspace(0.005, 0.1, 50)  # 50 seuils de 0.5% à 10% 
    pix2mas = 3.4
    position = (nDim // 2, nDim // 2)
    label_size3 = 12
    # Calcul des limites en mas
    # pour l'affichage
    x_min = -pix2mas * nSubDim // 2
    x_max = pix2mas * (nSubDim // 2 - 1)
    y_min = -pix2mas * nSubDim // 2
    y_max = pix2mas * (nSubDim // 2 - 1)

    results = []

    for fltr in lst_fltr_star2:
        fdir_star_fltr = os.path.join(fdir_star, fltr)
        outdir = f'/home/nbadolo/Bureau/Aymard/Donnees_sph/{folder_name}/{star_name}/plots/fits/log_scale/fully_automatic/'
        os.makedirs(outdir, exist_ok=True)
        fname1 = 'zpl_p23_make_polar_maps-ZPL_SCIENCE_P23_REDUCED'
        fname2 = '-zpl_science_p23_REDUCED'
        file_PI_star = os.path.join(fdir_star_fltr, fname1 + '_PI' + fname2 + '_PI.fits')
        if not os.path.exists(file_PI_star):
            print(f"Fichier manquant : {file_PI_star}")
            continue

        hdu = fits.open(file_PI_star)
        data = hdu[0].data
        header = hdu[0].header
        # Récupère les deux filtres du header
        star_name2 = header.get('OBJECT')
        fltr1 = header.get('HIERARCH ESO INS3 OPTI5 NAME', 'Filtre1 inconnu')
        fltr2 = header.get('HIERARCH ESO INS3 OPTI6 NAME', 'Filtre2 inconnu')
        fltr_arr = [fltr1, fltr2]
        n_fsize = data.shape[0]  # nombre de plans dans le cube (souvent 2)

        # Affichage des informations
        for z in range(n_fsize):
            intensity = data[z, :, :]
            cutout = Cutout2D(intensity, position=position, size= size )
            sub_v = cutout.data

            best_cost = 1.0
            best_threshold = None
            best_params = None
            best_region = None

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
                # Initialisation des paramètres
                def cost(params):
                    x0, y0, a, b, theta = params
                    # Astuce : on force a et b à rester dans le cadre
                    a = min(a, nSubDim/2 - 2)
                    b = min(b, nSubDim/2 - 2)
                    try:
                        coords = draw.ellipse(y0, x0, a, b, shape=Ellips.shape, rotation=theta)
                        template = np.zeros_like(Ellips)
                        template[coords] = 1
                        intersection = np.sum((template == 1) & (Ellips == 1))
                        size_sum = np.sum(template) + np.sum(Ellips)
                        dice = 2 * intersection / size_sum if size_sum > 0 else 0
                        return 1 - dice  # à minimiser
                    except Exception as e:
                        print(f"Erreur draw.ellipse pour nSubDim={nSubDim}, a={a}, b={b}: {e}")
                        return 1  # Pénalise fortement ce fit

                x_f, y_f, a_f, b_f, theta_f = opt.fmin(cost, (x_i, y_i, a_i, b_i, theta_i), disp=False)
                fit_cost = cost([x_f, y_f, a_f, b_f, theta_f])

                if fit_cost < best_cost:
                    best_cost = fit_cost
                    best_threshold = threshold
                    best_params = (x_f, y_f, a_f, b_f, theta_f)
                    best_region = region_max

            # Si aucun fit n'a été trouvé, passe à l'image suivante
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

            results.append({
                'star': star_name,
                'filter': fltr_arr[z],
                'frame_type': f'Pol_Intensity_{z}',
                'diameter_major_mas': diameter_mas,
                'diameter_minor_mas': diameter_minor_mas,
                'center_x_mas': x_centroid_mas,
                'center_y_mas': y_centroid_mas,
                'theta_deg': np.degrees(theta_f),
                'fit_cost': best_cost,
                'threshold': best_threshold
            })
            print(f"Traitement : filtre={fltr_arr[z]}, seuil={best_threshold:.4f}, fit_cost={best_cost:.4f}")

            # ... (le reste du code pour l'affichage et la sauvegarde de la figure) ...
            # ...existing code...

            # Plot unique contour stylé (log uniquement)
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
            cbar.set_label('Log$_{10}$(PI)', fontsize=12, fontweight='bold')
            # for t in cbar.ax.get_yticklabels(): # pour mettre en gras les labels de la colorbar
            #     t.set_fontweight('bold')

            ax.plot(x_contour_mas, y_contour_mas, color='cyan', linewidth=2, linestyle='--')
            ax.scatter([x_centroid_mas], [y_centroid_mas], color='red', marker='x')
            # ax.set_xlabel('RA (mas)', fontweight='bold')
            # ax.set_ylabel('DEC (mas)', fontweight='bold')
            # Ajustement des axes
            ax.set_xlabel("Relative RA(mas)", fontsize=11, fontweight='bold')
            ax.set_ylabel("Relative Dec (mas)", fontsize=11, fontweight='bold')
            ax.tick_params(axis='both', labelsize=9, width=1.2)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
            ax.locator_params(axis='x', nbins=5)
            ax.locator_params(axis='y', nbins=5)
            # Annotations texte pour le nom de l'étoile et le filtre
            ax.text(0.02, 0.95, f'{star_name2}', transform=ax.transAxes, fontsize=12, fontweight='bold', color='white', va='top')
            ax.text(0.02, 0.02, f'{fltr_arr[z]}', transform=ax.transAxes, fontsize=12, fontweight='bold', color='white', va='bottom')

            plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10)
            plt.savefig(os.path.join(outdir, f'unique_max_contour_for_Pol_Intensity_{star_name}_{obsmod}_{fltr_arr[z]}_{z}.png'), dpi=100, bbox_inches='tight')
            #plt.show()
            plt.close()

    # Sauvegarde du DataFrame
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(outdir, f'morpho_results_{star_name}_{obsmod}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Résultats morphologiques sauvegardés dans : {csv_path}")

# Exemple d'appel de la fonction
log_image('First','V854_Cen', 'alone')
log_image('First','V854_Cen', 'both')