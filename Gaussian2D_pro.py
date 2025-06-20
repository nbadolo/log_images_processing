#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:58:00 2025

@author: nbadolo
"""

"""
Ce code ajuste une gaussienne 2D à ce profil en utilisant curve_fit. 
La FWHM est calculée à partir de l'écart-type σ obtenu après l'ajustement de la gaussienne. 
Le profil d'intensité et l'ajustement gaussien sont tracés pour qu'on puisse
visualiser la qualité de l'ajustement.  Plus automatisé que son grand frère Gaussian2D_1D_pro,
il calcule un khi^2 réduit pour évaluer la qualité du Fit. Pour ce faire, il suppose un bruit de lecture 
de 0.01 pour ZIMPOL qu'il utilise ensuite pour calculer l'incertitude sur les données d'observation(y).
Cette instertitude sigma_y est en fin utilisé dans l'estimation du khi²_réd.
Les inputs ainsi les résultats sont dans le dossier : 
    /home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian/Output.
    
                            CODE OKAY !!!!
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable  # Permet de créer une colorbar compacte
from AymardPack import process_fits_image as pfi # Pour l'extraction du bruit et des pixels morts et chauds

# === PARAMÈTRES GLOBAUX ===
PIXEL_TO_MAS = 3.4              # Conversion de pixels en milli-arcsecondes (mas)
READ_NOISE = 0.01               # Bruit de lecture estimé (utilisé pour chi² réduit)
WINDOW_PIXELS = 30              # Taille du zoom autour de l'étoile pour le fit (en pixels)

# === GAUSSIENNE 2D CIRCULAIRE ===
def circular_gaussian_2d(coords, A, x0, y0, sigma):
    """
    Fonction de gaussienne 2D circulaire (avec le même sigma dans toutes les directions).
    :param coords: tuple de meshgrid aplatis (X, Y) pour effectuer le fit
    :param A: Amplitude de la gaussienne
    :param x0, y0: Coordonnées du centre de la gaussienne
    :param sigma: Écart-type de la gaussienne
    :return: Valeurs de la gaussienne à chaque point du meshgrid
    """
    x, y = coords
    return A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)).ravel()

# === FIT SUR UNE IMAGE UNIQUE ===
def process_single_frame_2d(frame, object_name, filtre, frame_index, save_path):
    """
    Cette fonction effectue un fit d'une gaussienne circulaire 2D sur une image centrée autour de son pic.
    Les résultats sont ensuite affichés avec des contours et sauvegardés dans un fichier.
    
    :param frame: Image (frame) à analyser
    :param object_name: Nom de l'étoile à analyser
    :param filtre: Filtre utilisé pour l'image
    :param frame_index: Indice de la frame pour la gestion des fichiers
    :param save_path: Chemin où les résultats seront sauvegardés
    :return: Dictionnaire contenant les résultats du fit (FWHM, chi² réduit, etc.)
    """
    # Recherche du pic de l'étoile pour centrer le zoom
    y_max, x_max = np.unravel_index(np.argmax(frame), frame.shape)
    y1, y2 = y_max - WINDOW_PIXELS, y_max + WINDOW_PIXELS
    x1, x2 = x_max - WINDOW_PIXELS, x_max + WINDOW_PIXELS
    sub_img = frame[y1:y2, x1:x2].astype(float)

    # Normalisation de l'image [0, 1]
    sub_img = pfi(sub_img) # extraction des mauvais pixels 
    sub_img -= np.min(sub_img)
    #print(np.max(sub_img))
    sub_img /= np.max(sub_img)

    # Création du meshgrid pour le fit
    Y, X = np.meshgrid(np.arange(sub_img.shape[0]), np.arange(sub_img.shape[1]), indexing='ij')
    initial_guess = (1.0, sub_img.shape[1] / 2, sub_img.shape[0] / 2, 5.0)

    # Essai de fit avec la fonction gaussienne
    try:
        params, _ = curve_fit(circular_gaussian_2d, (X, Y), sub_img.ravel(), p0=initial_guess)
    except Exception as e:
        print(f"⚠️ Fit échoué pour {object_name}, frame {frame_index} : {e}")
        return None

    # Calcul des paramètres du fit
    A, x0, y0, sigma = params # ce qui donne le 4 dans l'expression du chi2_red
    FWHM_pix = 2 * np.sqrt(2 * np.log(2)) * sigma
    FWHM_mas = FWHM_pix * PIXEL_TO_MAS
    #print("param =" + str(params))
    # Calcul des résidus et du chi² réduit
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

    # # Contours du fit
    # contour_levels = np.linspace(np.min(fit_2d), np.max(fit_2d), 5)
    # x_shifted = (X - x0) * PIXEL_TO_MAS
    # y_shifted = (Y - y0) * PIXEL_TO_MAS
    # ax.contour(x_shifted, y_shifted, fit_2d, levels=contour_levels, colors='white', linewidths=1.2)
    
    # Contours du fit (avec échelle logarithmique pour mieux capturer les faibles signaux)
    min_level = 0.01 * np.max(fit_2d)  # Niveau minimum: 1% du maximum
    max_level = np.max(fit_2d)
    
    # Création des niveaux logarithmiques
    num_levels = 5  # Nombre de niveaux de contours
    contour_levels = np.logspace(np.log10(min_level), np.log10(max_level), num_levels)
    
    x_shifted = (X - x0) * PIXEL_TO_MAS
    y_shifted = (Y - y0) * PIXEL_TO_MAS
    
    ax.contour(x_shifted, y_shifted, fit_2d, levels=contour_levels, colors='white', linewidths=1.2)


    # Annotations texte pour le nom de l'étoile et le filtre
    ax.text(0.02, 0.95, object_name, transform=ax.transAxes,
            fontsize=12, fontweight='bold', color='white', ha='left', va='top')
    ax.text(0.02, 0.02, filtre, transform=ax.transAxes,
            fontsize=12, fontweight='bold', color='white', ha='left', va='bottom')
    
    

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

    # Sauvegarde de la figure
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig_filename = f"{object_name.replace(' ', '_')}_{filtre.replace(' ', '_')}_frame{frame_index+1}.png"
    plt.savefig(os.path.join(save_path, fig_filename))
    plt.show()
    plt.close()

    # Retourne les résultats sous forme de dictionnaire
    return {
        'Etoile': object_name,
        'Filtre': filtre,
        'FWHM_mas': round(FWHM_mas, 2),
        'Sigma_pix': round(sigma, 2),
        'Chi2_reduit': round(chi2_red, 4)
    }

# === TRAITEMENT D'UN FICHIER FITS ===
def process_fits_file(filepath, fallback_name, save_path):
    """
    Traite un fichier FITS contenant des images et effectue le fit de gaussienne pour chaque image.
    
    :param filepath: Chemin vers le fichier FITS à traiter
    :param fallback_name: Nom de l'étoile à utiliser si le fichier FITS ne contient pas de nom d'objet
    :param save_path: Chemin où les résultats seront sauvegardés
    :return: Liste des résultats des fits pour ce fichier
    """
    results = []
    try:
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            header = hdul[0].header
            object_name = header.get('OBJECT', fallback_name)
            filtre1 = header.get('HIERARCH ESO INS3 OPTI5 NAME', 'filtre1')
            filtre2 = header.get('HIERARCH ESO INS3 OPTI6 NAME', 'filtre2')

            # Si l'image est en 3D et contient deux filtres
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


def process_directory(fits_root_folder, output_root_folder):
    """
    Traite tous les fichiers FITS dans un dossier et enregistre les résultats dans un fichier CSV.
    
    :param fits_root_folder: Dossier contenant les fichiers FITS
    :param output_root_folder: Dossier où les résultats seront sauvegardés
    """
    os.makedirs(output_root_folder, exist_ok=True)
    all_results = []

    for star_folder in os.listdir(fits_root_folder):
        star_path = os.path.join(fits_root_folder, star_folder)
        intensity_star_path = os.path.join(star_path, "Intensity", "star")  # <<< changement ici

        if not os.path.isdir(intensity_star_path):
            continue

        fits_files = [f for f in os.listdir(intensity_star_path) if f.endswith(".fits")]
        if not fits_files:
            print(f"⛔ Aucun .fits trouvé dans {intensity_star_path}")
            continue

        fits_path = os.path.join(intensity_star_path, fits_files[0])  # <<< et ici
        #save_path = os.path.join(output_root_folder, star_folder)
        save_path= out_path_plot
        print(f"📁 Traitement de : {star_folder}")
        results = process_fits_file(fits_path, fallback_name=star_folder, save_path=save_path)
        all_results.extend(results)
    # Sauvegarde des résultats dans un fichier CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_csv, 'resultats_fwhm.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Résultats sauvegardés dans : {csv_path}")
    print(df)





# === LANCEMENT ===
folder_name = "test1/"
#folder_name = "large_log_+/"
#folder_name = "resolved_log/"
#folder_name = "all_resolved_log/"
input_path = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian/Input/"+folder_name
out_path = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian/Output/"+folder_name
out_path_plot = out_path + "Intensity/"
output_csv = out_path + "Csv/"


process_directory(input_path, out_path)
