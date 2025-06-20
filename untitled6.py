#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 13:32:01 2025

@author: nbadolo
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from AymardPack import process_fits_image as pfi  # Nettoyage hot/dead pixels

# === PARAMÈTRES GLOBAUX ===
PIXEL_TO_MAS = 3.4       # Conversion pixel vers milli-arcsecondes
READ_NOISE = 0.01        # Bruit de lecture estimé
WINDOW_PIXELS = 30       # Taille de la fenêtre centrée sur le pic

# === MODÈLE : GAUSSIENNE 2D CIRCULAIRE ===
def circular_gaussian_2d(coords, A, x0, y0, sigma):
    x, y = coords
    return A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)).ravel()

# === TRAITEMENT D'UNE IMAGE 2D INDIVIDUELLE ===
def process_single_frame_2d(frame, object_name, filtre, frame_index, save_path):
    if frame.ndim != 2:
        print(f"⚠️ Image inattendue (non 2D) pour {object_name} ({filtre})")
        return None

    # Extraction d'une sous-image centrée autour du pic
    y_max, x_max = np.unravel_index(np.argmax(frame), frame.shape)
    y1, y2 = y_max - WINDOW_PIXELS, y_max + WINDOW_PIXELS
    x1, x2 = x_max - WINDOW_PIXELS, x_max + WINDOW_PIXELS
    sub_img = frame[y1:y2, x1:x2].astype(float)

    # Nettoyage, normalisation
    sub_img = pfi(sub_img)
    sub_img -= np.min(sub_img)
    sub_img /= np.max(sub_img)

    # Meshgrid + estimation initiale
    Y, X = np.meshgrid(np.arange(sub_img.shape[0]), np.arange(sub_img.shape[1]), indexing='ij')
    initial_guess = (1.0, sub_img.shape[1]/2, sub_img.shape[0]/2, 5.0)

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

    # === PLOT avec contours et barres ===
    extent_mas = np.array([
        -sub_img.shape[1]/2, sub_img.shape[1]/2,
        -sub_img.shape[0]/2, sub_img.shape[0]/2
    ]) * PIXEL_TO_MAS

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sub_img, cmap='inferno', origin='lower', extent=extent_mas)

    contour_levels = np.logspace(np.log10(0.01 * np.max(fit_2d)), np.log10(np.max(fit_2d)), 5)
    x_shifted = (X - x0) * PIXEL_TO_MAS
    y_shifted = (Y - y0) * PIXEL_TO_MAS
    ax.contour(x_shifted, y_shifted, fit_2d, levels=contour_levels, colors='white', linewidths=1.2)

    ax.set_xlabel("Relative RA (mas)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Relative Dec (mas)", fontsize=11, fontweight='bold')
    ax.text(0.02, 0.95, object_name, transform=ax.transAxes, fontsize=12, fontweight='bold', color='white', va='top')
    ax.text(0.02, 0.02, filtre, transform=ax.transAxes, fontsize=12, fontweight='bold', color='white', va='bottom')

    # Colorbar compacte à droite
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax).set_label('I / Imax', fontsize=10, fontweight='bold')

    # Sauvegarde figure
    os.makedirs(save_path, exist_ok=True)
    fig_filename = f"{object_name.replace(' ', '_')}_{filtre.replace(' ', '_')}_frame{frame_index+1}.png"
    plt.savefig(os.path.join(save_path, fig_filename))
    plt.close()

    # Résultats numériques retournés
    return {
        'Etoile': object_name,
        'Filtre': filtre,
        'FWHM_mas': round(FWHM_mas, 2),
        'Sigma_pix': round(sigma, 2),
        'Chi2_reduit': round(chi2_red, 4)
    }

# === TRAITEMENT D’UN FICHIER FITS ===
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
                r1 = process_single_frame_2d(data[0], object_name, filtre1, 0, save_path)
                r2 = process_single_frame_2d(data[1], object_name, filtre2, 1, save_path)
                if r1: results.append(r1)
                if r2: results.append(r2)
            elif data.ndim == 2:
                r = process_single_frame_2d(data, object_name, filtre1, 0, save_path)
                if r: results.append(r)
            else:
                print(f"⚠️ Format inattendu dans {filepath}")
    except Exception as e:
        print(f"❌ Erreur ouverture {filepath} : {e}")
    return results

# === TRAITEMENT GLOBAL D’UN DOSSIER COMPLET ===
def process_directory(fits_root_folder, output_root_folder):
    os.makedirs(output_root_folder, exist_ok=True)
    all_results = []

    for star_folder in os.listdir(fits_root_folder):
        star_path = os.path.join(fits_root_folder, star_folder)
        intensity_star_path = os.path.join(star_path, "Intensity", "star")
        intensity_psf_path = os.path.join(star_path, "Intensity", "psf")

        if not os.path.isdir(intensity_star_path):
            continue

        fits_files_star = [f for f in os.listdir(intensity_star_path) if f.endswith(".fits")]
        fits_files_psf = []
        if os.path.isdir(intensity_psf_path):
            fits_files_psf = [f for f in os.listdir(intensity_psf_path) if f.endswith(".fits")]

        if not fits_files_star:
            continue

        # # === TRAITEMENT DE L'ÉTOILE ===   Une ligne après chaque étoile sans psf
        # star_fits_path = os.path.join(intensity_star_path, fits_files_star[0])
        # star_results = process_fits_file(star_fits_path, fallback_name=star_folder, save_path=out_path_plot)
        # all_results.extend(star_results)

        # # === TRAITEMENT DE LA PSF (si dispo), sinon insérer ligne vide ===
        # if fits_files_psf:
        #     psf_fits_path = os.path.join(intensity_psf_path, fits_files_psf[0])
        #     psf_results = process_fits_file(psf_fits_path, fallback_name=star_folder + "_psf", save_path=out_path_plot)
        #     all_results.extend(psf_results)
        # else:
        #     # Ligne de séparation avec des tirets pour signaler PSF manquante
        #     empty_row = {'Etoile': '-', 'Filtre': '-', 'FWHM_mas': '-', 'Sigma_pix': '-', 'Chi2_reduit': '-'}
        #     all_results.append(empty_row)
        
        # === TRAITEMENT DE L'ÉTOILE === une ligne apres chaque frame de chaque étoile sans psf
        star_fits_path = os.path.join(intensity_star_path, fits_files_star[0])
        star_results = process_fits_file(star_fits_path, fallback_name=star_folder, save_path=out_path_plot)
        
        # === TRAITEMENT DE LA PSF (si dispo) ===
        if fits_files_psf:
            psf_fits_path = os.path.join(intensity_psf_path, fits_files_psf[0])
            psf_results = process_fits_file(psf_fits_path, fallback_name=star_folder + "_psf", save_path=out_path_plot)
            for s, p in zip(star_results, psf_results):
                all_results.append(s)
                all_results.append(p)
            # Si PSF a moins de frames que l’étoile
            if len(psf_results) < len(star_results):
                for s in star_results[len(psf_results):]:
                    all_results.append(s)
                    all_results.append({'Etoile': '-', 'Filtre': '-', 'FWHM_mas': '-', 'Sigma_pix': '-', 'Chi2_reduit': '-'})
        else:
            # Pas de PSF : ajouter une ligne de tirets après chaque frame étoile
            for s in star_results:
                all_results.append(s)
                all_results.append({'Etoile': '-', 'Filtre': '-', 'FWHM_mas': '-', 'Sigma_pix': '-', 'Chi2_reduit': '-'})


    # === SAUVEGARDE CSV FINAL ===
    df = pd.DataFrame(all_results)
    os.makedirs(output_csv, exist_ok=True)
    csv_path = os.path.join(output_csv, 'resultats_fwhm.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Résultats enregistrés dans : {csv_path}")
    print(df)

# === LANCEMENT PRINCIPAL ===
folder_name = "test1/"
input_path = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian/Input/" + folder_name
out_path = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian/Output/" + folder_name
out_path_plot = out_path + "Intensity/"
output_csv = out_path + "Csv/"

process_directory(input_path, out_path)
