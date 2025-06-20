#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 16:54:19 2025

@author: nbadolo
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
import pandas as pd

"""
Ce code fait la projection de l'image 2D d'une étoile' le long de l'axe 
y (en prenant la ligne centrale) pour obtenir un profil d'intensité en fonction de 
x, donc un profile d'intensité en 1D. Une fois le profil d'intensité en 1D obtenu, 
nous ajustons une gaussienne 1D à ce profil en utilisant curve_fit. 
La FWHM est calculée à partir de l'écart-type σ obtenu après l'ajustement de la gaussienne. 
Le profil d'intensité et l'ajustement gaussien sont tracés pour qu'on puisse
visualiser la qualité de l'ajustement.  Plus automatisé que son grand frère Gaussian2D_1D_pro,
il calcule un khi^2 pour évaluer la qualité du Fit. Pour ce faire, il suppose un bruit de lecture 
de 0.01 pour ZIMPOL qu'il utilise ensuite pour calculel'incertitude sur les données d'observation(y).
Cette instertitude sigma_y est en fin utilisé dans l'estimation du khi^2.
Les inputs ainsi les résultats sont dans le dossier : 
    /home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian/Output.
    
                            CODE OKAY !!!!
"""

# === PARAMÈTRES GLOBAUX ===
PIXEL_TO_MAS = 3.4              # Conversion pixel → milli-arcsecond
WINDOW_PIXELS = 30             # Taille de la fenêtre centrée pour affichage
READ_NOISE = 0.01               # Bruit de lecture estimé pour le chi² rédui. Réaliste pour ZIMPOL

# === FONCTION GAUSSIENNE POUR L’AJUSTEMENT ===
def gauss1d(x, A, mu, sigma):
    """
    Définition d'une gaussienne 1D pour ajustement du profil d'intensité.
    :param x: axe des abscisses (pixels)
    :param A: amplitude de la gaussienne
    :param mu: position centrale de la gaussienne
    :param sigma: écart-type de la gaussienne
    :return: valeurs y correspondant à la gaussienne
    """
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# === TRAITEMENT D'UNE IMAGE INDIVIDUELLE ===
def process_single_frame(frame, object_name, filtre, frame_index, save_path):
    """
    Traite un cadre/image FITS : coupe centrale, fit gaussien, calcul FWHM & chi²,
    sauvegarde du graphique avec résultats affichés.
    
    :param frame: image 2D
    :param object_name: nom de l'étoile (header ou fallback)
    :param filtre: nom du filtre utilisé (extrait du header)
    :param frame_index: index de l’image dans le cube FITS
    :param save_path: chemin pour sauvegarde du graphique
    :return: dictionnaire des résultats (FWHM, sigma, chi²...)
    """
    # Extraire la ligne centrale du profil horizontal
    row = frame[frame.shape[0] // 2, :].astype(float)
    row -= np.min(row)
    row /= np.max(row)

    # Axe x en pixels
    x_pixels = np.arange(len(row))
    initial_guess = [1.0, np.argmax(row), 5.0]  # amplitude, centre, sigma initial

    try:
        params, _ = curve_fit(gauss1d, x_pixels, row, p0=initial_guess)
    except Exception as e:
        print(f"⚠️ Fit échoué pour {object_name}, frame {frame_index} : {e}")
        return None

    # Récupération des paramètres du fit
    A, mu_pix, sigma_pix = params
    FWHM_pix = 2 * np.sqrt(2 * np.log(2)) * sigma_pix
    FWHM_mas = FWHM_pix * PIXEL_TO_MAS

    # Calcul du chi² réduit
    idx_start = int(max(mu_pix - WINDOW_PIXELS, 0))
    idx_end = int(min(mu_pix + WINDOW_PIXELS + 1, len(row)))
    x_crop = x_pixels[idx_start:idx_end]
    x_mas = (x_crop - mu_pix) * PIXEL_TO_MAS
    y = row[idx_start:idx_end]
    y_fit = gauss1d(x_crop, *params)
    sigma_y = np.sqrt(y + READ_NOISE**2)
    residuals = y - y_fit
    dof = len(y) - 3
    chi2_red = np.sum((residuals / sigma_y) ** 2) / dof

    # Génération du graphique
    plt.figure()
    plt.plot(x_mas, y, 'b-', label="Star")
    plt.plot(x_mas, y_fit, 'r--', label="Fit")
    plt.title(f"{object_name} - {filtre}", fontsize=12)
    plt.text(0.05, 0.90, f"FWHM = {FWHM_mas:.2f} mas", transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.text(0.05, 0.80, f"χ²_réd = {chi2_red:.4f}", transform=plt.gca().transAxes,
             fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.xlabel("Radius(mas)")
    plt.ylabel("I / Imax")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()

    # Sauvegarde du graphique
    fig_filename = f"{object_name.replace(' ', '_')}_{filtre.replace(' ', '_')}_frame{frame_index+1}.png"
    plt.savefig(os.path.join(out_path_plot, fig_filename))
    plt.show()
    plt.close()

    # Retourne les résultats pour le tableau final
    return {
        'Etoile': object_name,
        'Filtre': filtre,
        'FWHM_mas': round(FWHM_mas, 2),
        'Sigma_pix': round(sigma_pix, 2),
        'Chi2_reduit': round(chi2_red, 4)
    }

# === TRAITEMENT D’UN FICHIER FITS ===
def process_fits_file(filepath, fallback_name, save_path):
    """
    Traite un fichier FITS contenant 2 frames : une par filtre.
    :param filepath: chemin complet vers le fichier FITS
    :param fallback_name: nom à utiliser si OBJECT est absent dans le header
    :param save_path: dossier où enregistrer les images des fits
    :return: liste de résultats (dicts) pour chaque frame
    """
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
                    result = process_single_frame(data[i], object_name, filters[i], i, save_path)
                    if result:
                        results.append(result)
            else:
                print(f"⚠️ Format inattendu pour {filepath}")

    except Exception as e:
        print(f"❌ Erreur lecture {filepath} : {e}")

    return results

# === TRAITEMENT DU DOSSIER GLOBAL ===
def process_directory(fits_root_folder, output_root_folder):
    """
    Parcourt tous les sous-dossiers de `fits_root_folder`, cherche les fichiers .fits
    dans le dossier `Intensity/` de chaque étoile, applique les traitements et
    enregistre les résultats (images + CSV).
    
    :param fits_root_folder: dossier principal avec les sous-dossiers par étoile
    :param output_root_folder: dossier pour les résultats finaux
    """
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
        #os.makedirs(save_path, exist_ok=True)

        print(f"📁 Traitement de : {star_folder}")
        results = process_fits_file(fits_path, fallback_name=star_folder, save_path=save_path)
        all_results.extend(results)

    # Sauvegarde finale dans un fichier CSV
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_folder, 'resultats_fwhm.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Résultats sauvegardés dans : {csv_path}")
    print(df)

# === APPEL DU SCRIPT ===


folder_name = "test/"
#folder_name = "large_log_+/"
#folder_name = "resolved_log/"
#folder_name = "all_resolved_log/"
input_path = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian/Input/"+folder_name
out_path = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian/Output/"+folder_name
out_path_plot = out_path + "Intensity/"
output_folder = out_path + "Csv/"


process_directory(input_path, out_path)
