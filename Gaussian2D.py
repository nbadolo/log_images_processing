#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 17:00:23 2025

@author: nbadolo
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable  # à ajouter en haut du script si pas encore fait


# === PARAMÈTRES GLOBAUX ===
PIXEL_TO_MAS = 3.4              # Conversion pixel → milli-arcsecond
READ_NOISE = 0.01               # Bruit de lecture estimé (chi² réduit)
WINDOW_PIXELS = 60              # Taille du zoom autour de l'étoile pour le fit

# === GAUSSIENNE 2D CIRCULAIRE ===
def circular_gaussian_2d(coords, A, x0, y0, sigma):
    """
    Fonction de gaussienne 2D circulaire (même sigma dans toutes les directions).
    :param coords: tuple de meshgrid aplatis (X, Y)
    :return: tableau aplati de la gaussienne
    """
    x, y = coords
    return A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)).ravel()

# === FIT SUR UNE IMAGE UNIQUE ===

def process_single_frame_2d(frame, object_name, filtre, frame_index, save_path):
    y_max, x_max = np.unravel_index(np.argmax(frame), frame.shape)
    y1, y2 = y_max - WINDOW_PIXELS, y_max + WINDOW_PIXELS
    x1, x2 = x_max - WINDOW_PIXELS, x_max + WINDOW_PIXELS
    sub_img = frame[y1:y2, x1:x2].astype(float)

    sub_img -= np.min(sub_img)
    sub_img /= np.max(sub_img)

    Y, X = np.meshgrid(np.arange(sub_img.shape[0]), np.arange(sub_img.shape[1]), indexing='ij')
    initial_guess = (1.0, sub_img.shape[1] / 2, sub_img.shape[0] / 2, 5.0)

    try:
        params, _ = curve_fit(circular_gaussian_2d, (X, Y), sub_img.ravel(), p0=initial_guess)
    except Exception as e:
        print(f"⚠️ Fit échoué pour {object_name}, frame {frame_index} : {e}")
        return None

    A, x0, y0, sigma = params
    FWHM_pix = 2 * np.sqrt(2 * np.log(2)) * sigma
    FWHM_mas = FWHM_pix * PIXEL_TO_MAS

    fit_2d = circular_gaussian_2d((X, Y), *params).reshape(sub_img.shape)
    residuals = sub_img - fit_2d
    sigma_data = np.sqrt(sub_img + READ_NOISE**2)
    chi2_red = np.sum((residuals / sigma_data) ** 2) / (sub_img.size - 4)

            # === AFFICHAGE ===
    extent_mas = np.array([
        -sub_img.shape[1] / 2, sub_img.shape[1] / 2,
        -sub_img.shape[0] / 2, sub_img.shape[0] / 2
    ]) * PIXEL_TO_MAS

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sub_img, origin='lower', cmap='inferno', extent=extent_mas)

    # Contours du fit
    contour_levels = np.linspace(np.min(fit_2d), np.max(fit_2d), 5)
    x_shifted = (X - x0) * PIXEL_TO_MAS
    y_shifted = (Y - y0) * PIXEL_TO_MAS
    ax.contour(x_shifted, y_shifted, fit_2d, levels=contour_levels, colors='white', linewidths=1.2)

    # Annotations dans la figure
    ax.text(0.02, 0.95, object_name, transform=ax.transAxes,
            fontsize=12, fontweight='bold', color='white', ha='left', va='top')
    ax.text(0.02, 0.02, filtre, transform=ax.transAxes,
            fontsize=12, fontweight='bold', color='white', ha='left', va='bottom')

    # ax.text(0.05, 0.92, f"FWHM = {FWHM_mas:.2f} mas", transform=ax.transAxes,
    #         fontsize=10, fontweight='normal', color='white')
    # ax.text(0.05, 0.85, f"$\\chi^2_{{réduit}}$ = {chi2_red:.4f}", transform=ax.transAxes,
    #         fontsize=10, fontweight='normal', color='white')

    # Axes et ticks
    ax.set_xlabel("Relative RA(mas)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Relative Dec (mas)", fontsize=11, fontweight='bold')
    ax.tick_params(axis='both', labelsize=9, width=1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)

    # Colorbar compacte à droite
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('I / Imax', fontsize=10, fontweight='bold')
    for t in cbar.ax.get_yticklabels():
        t.set_fontweight('bold')

    plt.tight_layout()


    os.makedirs(save_path, exist_ok=True)
    fig_filename = f"{object_name.replace(' ', '_')}_{filtre.replace(' ', '_')}_frame{frame_index+1}.png"
    plt.savefig(os.path.join(save_path, fig_filename))
    plt.show()
    plt.close()

    return {
        'Etoile': object_name,
        'Filtre': filtre,
        'FWHM_mas': round(FWHM_mas, 2),
        'Sigma_pix': round(sigma, 2),
        'Chi2_reduit': round(chi2_red, 4)
    }



# === TRAITEMENT D'UN FICHIER FITS ===
def process_fits_file(filepath, fallback_name, save_path):
    results = []
    try:
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            object_name = header.get('OBJECT', fallback_name)
            filtre1 = header.get('HIERARCH ESO INS3 OPTI5 NAME', 'filtre1')
            filtre2 = header.get('HIERARCH ESO INS3 OPTI6 NAME', 'filtre2')

            if data.ndim == 3 and data.shape[0] == 2:
                filters = [filtre1, filtre2]
                for i in range(2):
                    result = process_single_frame_2d(data[i], object_name, filters[i], i, save_path)
                    if result:
                        results.append(result)
            else:
                print(f"⚠️ Format inattendu pour {filepath}")
    except Exception as e:
        print(f"❌ Erreur lecture {filepath} : {e}")
    return results

# === TRAITEMENT D'UN DOSSIER ===
def process_directory(fits_root_folder, output_root_folder):
    os.makedirs(output_root_folder, exist_ok=True)
    all_results = []

    for star_folder in os.listdir(fits_root_folder):
        star_path = os.path.join(fits_root_folder, star_folder)
        intensity_path = os.path.join(star_path, "Intensity")

        if not os.path.isdir(intensity_path):
            continue

        fits_files = [f for f in os.listdir(intensity_path) if f.endswith(".fits")]
        if not fits_files:
            print(f"⛔ Aucun .fits trouvé dans {intensity_path}")
            continue

        fits_path = os.path.join(intensity_path, fits_files[0])
        save_path = os.path.join(output_root_folder, star_folder)

        print(f"📁 Traitement de : {star_folder}")
        results = process_fits_file(fits_path, fallback_name=star_folder, save_path=save_path)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_root_folder, 'resultats_fwhm.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Résultats sauvegardés dans : {csv_path}")
    print(df)

# === LANCEMENT ===
folder_name = "test/"
input_path = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian/Input/" + folder_name
out_path = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian/Output/" + folder_name
process_directory(input_path, out_path)

