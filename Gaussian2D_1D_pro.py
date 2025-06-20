#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 10:49:48 2025

@author: nbadolo
"""


import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
import pandas as pd

# Conversion des pixels en millisecondes d'arc (mas)
PIXEL_TO_MAS = 3.4
# Taille de la fenêtre pour l'affichage autour du pic (en pixels)
WINDOW_PIXELS = 30


"""
Ce code fait la projection de l'image 2D d'une étoile' le long de l'axe 
y (en prenant la ligne centrale) pour obtenir un profil d'intensité en fonction de 
x, donc un profile d'intensité en 1D. Une fois le profil d'intensité en 1D obtenu, 
nous ajustons une gaussienne 1D à ce profil en utilisant curve_fit. 
La FWHM est calculée à partir de l'écart-type σ obtenu après l'ajustement de la gaussienne. 
Le profil d'intensité et l'ajustement gaussien sont tracés pour qu'on puisse
visualiser la qualité de l'ajustement.  Plus automatisé que son grand frère Gaussian2D_1D, il tient compte
d'une arborescence plus compliquée.
Les inputs ainsi les résultats sont dans le dossier : 
    /home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian/Output.
            
                CODE OKAY !!!!
"""


def gauss1d(x, A, mu, sigma):
    """
    Définition de la fonction gaussienne 1D à ajuster sur les profils d'intensité.
    :param x: Les valeurs de l'axe des abscisses (position en pixels)
    :param A: Amplitude de la gaussienne (valeur maximale)
    :param mu: Position du centre de la gaussienne
    :param sigma: Ecart-type de la gaussienne (contrôle la largeur)
    :return: Valeurs de la gaussienne pour chaque position de x
    """
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def process_single_frame(frame, object_name, filtre, frame_index, save_path):
    """
    Traite un seul cadre d'une étoile, effectue l'ajustement gaussien,
    affiche les résultats et enregistre la figure.
    :param frame: Image 2D de l'étoile à analyser
    :param object_name: Nom de l'étoile
    :param filtre: Nom du filtre utilisé pour l'observation
    :param frame_index: Index du cadre (0 ou 1)
    :param save_path: Dossier où sauvegarder la figure générée
    :return: Un dictionnaire avec les résultats de l'ajustement (FWHM et sigma)
    """
    # Sélection de la ligne centrale de l'image (position médiane en Y)
    row = frame[frame.shape[0] // 2, :].astype(float)

    # Normalisation de l'intensité : ramener à l'intervalle [0, 1]
    row -= np.min(row)
    row /= np.max(row)

    # Position des pixels sur l'axe des abscisses
    x_pixels = np.arange(len(row))
    
    # Estimation initiale des paramètres pour l'ajustement gaussien
    initial_guess = [1.0, np.argmax(row), 5.0]

    try:
        # Ajustement de la gaussienne sur la ligne des données
        params, _ = curve_fit(gauss1d, x_pixels, row, p0=initial_guess)
    except Exception as e:
        # En cas d'erreur lors de l'ajustement
        print(f"⚠️ Échec de l'ajustement pour {object_name} (frame {frame_index}) : {e}")
        return None

    # Récupération des paramètres ajustés : A, mu (moyenne), sigma
    A, mu_pix, sigma_pix = params
    # Calcul de la FWHM (largeur à mi-hauteur)
    FWHM_pix = 2 * np.sqrt(2 * np.log(2)) * sigma_pix
    # Conversion de la FWHM en millisecondes d'arc (mas)
    FWHM_mas = FWHM_pix * PIXEL_TO_MAS

    # Tracé des résultats
    idx_start = int(max(mu_pix - WINDOW_PIXELS, 0))  # Limites de la fenêtre d'affichage
    idx_end = int(min(mu_pix + WINDOW_PIXELS + 1, len(row)))
    x_crop = x_pixels[idx_start:idx_end]
    x_mas = (x_crop - mu_pix) * PIXEL_TO_MAS  # Conversion en mas
    y = row[idx_start:idx_end]
    y_fit = gauss1d(x_crop, *params)

    # Création de la figure avec les courbes des données et de l'ajustement
    plt.figure()
    plt.plot(x_mas, y, 'b-', label="Star")
    plt.plot(x_mas, y_fit, 'r--', label="Fit")
    plt.title(f"{object_name} - {filtre}", fontsize=12, fontweight='bold')
    plt.text(0.05, 0.9, f"FWHM = {FWHM_mas:.2f} mas",
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8))
    plt.xlabel("Radius(mas)")
    plt.ylabel("I / Imax")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
  

    # Sauvegarde de la figure sous format PNG dans le dossier spécifié
    fig_filename = f"{object_name.replace(' ', '_')}_{filtre.replace(' ', '_')}.png"
    plt.savefig(os.path.join(out_path_plot, fig_filename))
    plt.show()
    plt.close()  # Fermer la figure pour libérer la mémoire
   

    # Retourner les résultats de l'ajustement sous forme de dictionnaire
    # return {
    #     'Etoile': object_name,
    #     'Filtre': filtre,
    #     'FWHM_mas': round(FWHM_mas, 2),
    #     'Sigma_pix': round(sigma_pix, 2)##folder_name = "test/"
    #     #folder_name = "large_log_+/"
    #     #folder_name = "resolved_log/"
    #     #folder_name = "all_resolved_log/"
    #     input_path = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian/Input/"+folder_name
    #     out_path = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian/Output/"+folder_name
    #     out_path_plot = out_path + "Intensity/"
    #     output_folder = out_path + "Csv/"


        # Exemple d'appel :
        #process_directory(input_path, out_path
    #}

def process_fits_file(filepath, save_path):
    """
    Traite un fichier FITS, extrait les données des cadres et effectue l'ajustement
    pour chaque observation (frame).
    :param filepath: Chemin vers le fichier FITS
    :param save_path: Dossier où enregistrer les figures et le fichier CSV
    :return: Une liste de dictionnaires contenant les résultats des ajustements
    """
    results = []
    try:
        # Ouverture du fichier FITS
        with fits.open(filepath) as hdul:
            data = hdul[0].data
            header = hdul[0].header

            # Récupération du nom de l'étoile et des filtres associés dans les headers
            object_name = header.get('OBJECT', 'inconnu')
            filtre1 = header.get('HIERARCH ESO INS3 OPTI5 NAME', 'inconnu')
            filtre2 = header.get('HIERARCH ESO INS3 OPTI6 NAME', 'inconnu')

            # Si le fichier est un cube 3D avec 2 cadres (frames)
            if data.ndim == 3 and data.shape[0] == 2:
                filters = [filtre1, filtre2]
                for i in range(2):
                    frame = data[i]  # Récupération de l'image du cadre (frame)
                    filtre = filters[i]  # Nom du filtre associé à ce cadre
                    result = process_single_frame(frame, object_name, filtre, i, save_path)
                    if result:
                        results.append(result)
            else:
                print(f"⚠️ Fichier ignoré (pas 2 frames) : {filepath}")

    except Exception as e:
        # En cas d'erreur (par exemple, mauvais format de fichier)
        print(f"❌ Erreur avec {filepath} : {e}")

    return results

def process_directory(fits_folder, out_path):
    """
    Traite tous les fichiers FITS dans le dossier d'entrée, enregistre les résultats dans un CSV
    et les figures dans le dossier de sortie.
    :param fits_folder: Dossier contenant les fichiers FITS
    :param output_folder: Dossier où sauvegarder les résultats (CSV et PNG)
    """
    
    out_path_plot = out_path + "Intensity/"
    output_folder = out_path + "Csv/"

    # Création du dossier de sortie s'il n'existe pas déjà
    os.makedirs(output_folder, exist_ok=True)

    all_results = []  # Liste pour stocker les résultats de chaque observation

    # Parcours de tous les sous-dossiers dans le dossier FITS
    for star_folder in os.listdir(fits_folder):
        star_folder_path = os.path.join(fits_folder, star_folder)
        
        if os.path.isdir(star_folder_path):
            # Vérifier l'existence du sous-dossier "Intensity" dans chaque étoile
            intensity_folder = os.path.join(star_folder_path, 'Intensity')
            if os.path.exists(intensity_folder):
                # Chercher les fichiers FITS dans le sous-dossier "Intensity"
                fits_files = [f for f in os.listdir(intensity_folder) if f.lower().endswith('.fits')]
                for fits_file in fits_files:
                    fits_filepath = os.path.join(intensity_folder, fits_file)
                    print(f"\n🔍 Traitement du fichier : {fits_filepath}")
                    results = process_fits_file(fits_filepath, output_folder)
                    all_results.extend(results)  # Ajouter les résultats à la liste globale

    # Création du DataFrame avec les résultats
    df = pd.DataFrame(all_results)
    
    # Sauvegarde des résultats dans un fichier CSV dans le dossier de sortie
    csv_path = os.path.join(output_folder, 'resultats_fwhm.csv')
    df.to_csv(csv_path, index=False)
    print("\n✅ Résultats :")
    print(df)
    print(f"\n📄 Fichier CSV enregistré : {csv_path}")
    
#folder_name = "test/"
#folder_name = "large_log_+/"
#folder_name = "resolved_log/"
folder_name = "all_resolved_log/"
input_path = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian/Input/"+folder_name
out_path = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian/Output/"+folder_name
out_path_plot = out_path + "Intensity/"
output_folder = out_path + "Csv/"


# Exemple d'appel :
process_directory(input_path, out_path)
