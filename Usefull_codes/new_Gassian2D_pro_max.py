#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 14:14:39 2025

@author: nbadolo
"""

"""
Ce code ajuste une gaussienne 2D à ce profil en utilisant curve_fit. 
La FWHM est calculée à partir de l'écart-type σ obtenu après l'ajustement de la gaussienne. 
Le profil d'intensité et l'ajustement gaussien sont tracés pour qu'on puisse
visualiser la qualité de l'ajustement.  Plus automatisé que son grand frère Gaussian2D_pro,
il gere les psf aussi et met des tirets (dans le tableau de sortie) lorsque l'étoile n'a pasde psf
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from AymardPack import process_fits_image as pfi  # Nettoyage hot/dead pixels

# === PARAMÈTRES GLOBAUX ===
PIXEL_TO_MAS = 3.4       # Conversion pixel vers milli-arcsecondes
READ_NOISE = 0.01        # Bruit de lecture estimé
WINDOW_PIXELS = 100       # Taille de la fenêtre centrée sur le pic

def circular_gaussian_2d(coords, A, x0, y0, sigma):
    """
    Modèle de gaussienne 2D circulaire utilisé pour fitter les PSFs.
    
    Paramètres :
    - coords : tuple de meshgrid (X, Y)
    - A : amplitude
    - x0, y0 : centre de la gaussienne
    - sigma : écart-type (même en X et Y)

    Retourne : tableau aplati de l'image modélisée
    """
    x, y = coords
    return A * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)).ravel()

def process_single_frame_2d(frame, object_name, filtre, frame_index, save_path):
    """
    Traite une image 2D : extrait la PSF, fit une gaussienne 2D, retourne les paramètres
    et enregistre un visuel avec contours et colorbar.

    Paramètres :
    - frame : image 2D à traiter
    - object_name : nom de l'étoile
    - filtre : filtre associé à l'image
    - frame_index : index de la frame dans le fichier FITS
    - save_path : chemin où sauvegarder le visuel

    Retourne : dictionnaire des résultats numériques (FWHM, sigma, chi2)
    """
    if frame.ndim != 2:
        print(f"⚠️ Image inattendue (non 2D) pour {object_name} ({filtre})")
        return None

    y_max, x_max = np.unravel_index(np.argmax(frame), frame.shape)
    y1, y2 = y_max - WINDOW_PIXELS, y_max + WINDOW_PIXELS
    x1, x2 = x_max - WINDOW_PIXELS, x_max + WINDOW_PIXELS
    sub_img = frame[y1:y2, x1:x2].astype(float)

    sub_img = pfi(sub_img)
    sub_img -= np.min(sub_img)
    sub_img /= np.max(sub_img)

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

    extent_mas = np.array([
        -sub_img.shape[1]/2, sub_img.shape[1]/2,
        -sub_img.shape[0]/2, sub_img.shape[0]/2
    ]) * PIXEL_TO_MAS

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sub_img, cmap='inferno', origin='lower', extent=extent_mas)

    contour_levels = np.logspace(np.log10(0.01 * np.max(fit_2d)), np.log10(np.max(fit_2d)), 5)
    x_shifted = (X - x0) * PIXEL_TO_MAS
    y_shifted = (Y - y0) * PIXEL_TO_MAS
    ax.contour(x_shifted, y_shifted, fit_2d, levels=contour_levels, colors='white', linewidths=0.8)

    # ax.set_xlabel("Relative RA (mas)", fontsize=11, fontweight='bold')
    # ax.set_ylabel("Relative Dec (mas)", fontsize=11, fontweight='bold')
    
    # Annotations texte pour le nom de l'étoile et le filtre
    ax.text(0.02, 0.95, object_name, transform=ax.transAxes, fontsize=12, fontweight='bold', color='white', va='top')
    ax.text(0.02, 0.02, filtre, transform=ax.transAxes, fontsize=12, fontweight='bold', color='white', va='bottom')
    
    # Axes et ticks
    ax.set_xlabel("Relative RA(mas)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Relative Dec (mas)", fontsize=11, fontweight='bold')
    ax.tick_params(axis='both', labelsize=9, width=1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax).set_label('I / Imax', fontsize=10, fontweight='bold')

    os.makedirs(save_path, exist_ok=True)
    fig_filename = f"{object_name.replace(' ', '_')}_{filtre.replace(' ', '_')}_frame{frame_index+1}.png"
    plt.savefig(os.path.join(save_path, fig_filename), dpi=100, bbox_inches='tight')
    plt.close()

    return {
        'Etoile': object_name,
        'Filtre': filtre,
        'FWHM_mas': round(FWHM_mas, 2),
        'Sigma_pix': round(sigma, 2),
        'Chi2_reduit': round(chi2_red, 4)
    }

def process_fits_file(filepath, fallback_name, save_path):
    """
    Ouvre un fichier FITS et applique le traitement sur chaque frame présente (1 ou 2).
    
    Paramètres :
    - filepath : chemin du fichier FITS
    - fallback_name : nom à utiliser si le header ne contient pas 'OBJECT'
    - save_path : dossier pour sauvegarder les visuels

    Retourne : liste de dictionnaires de résultats
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

def process_directory(fits_root_folder, output_root_folder):
    """
    Parcourt tous les dossiers d'étoiles, traite les fichiers FITS (étoile + PSF),
    enregistre les visuels et compile les résultats dans un CSV final.

    Paramètres :
    - fits_root_folder : dossier contenant les sous-dossiers d’étoiles
    - output_root_folder : dossier de sortie pour les visuels et le CSV
    """
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
        # ✅ Ici : affichage du traitement en cours
        #print(f"📁 Traitement de : {star_folder}")
        # Couleur verte si la PSF est présente, sinon rouge
        color = "\033[92m" if fits_files_psf else "\033[91m"
        print(f"{color}📁 Traitement de : {star_folder} {'(PSF dispo)' if fits_files_psf else '(PSF absente)'}\033[0m")


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

        if fits_files_psf:
            psf_fits_path = os.path.join(intensity_psf_path, fits_files_psf[0])
            psf_results = process_fits_file(psf_fits_path, fallback_name=star_folder + "_psf", save_path=out_path_plot)

            max_len = max(len(star_results), len(psf_results))
            for i in range(max_len):
                s = star_results[i] if i < len(star_results) else {'Etoile': '-', 'Filtre': '-', 'FWHM_mas': '-', 'Sigma_pix': '-', 'Chi2_reduit': '-'}
                p = psf_results[i] if i < len(psf_results) else {'Etoile': '-', 'Filtre': '-', 'FWHM_mas': '-', 'Sigma_pix': '-', 'Chi2_reduit': '-'}
                all_results.append(s)
                all_results.append(p)
        else:
            for s in star_results:
                all_results.append(s)
                all_results.append({'Etoile': '-', 'Filtre': '-', 'FWHM_mas': '-', 'Sigma_pix': '-', 'Chi2_reduit': '-'})

    df = pd.DataFrame(all_results)
    os.makedirs(output_csv, exist_ok=True)
    csv_path = os.path.join(output_csv, 'resultats_fwhm.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Résultats enregistrés dans : {csv_path}")
    print(df)

# === LANCEMENT PRINCIPAL ===

folder_name ="V854_Cen/"
#folder_name = "test1/"
#folder_name = "large_log_+/"
#folder_name = "resolved_log/"
#folder_name = "all_resolved_log/"
main_path = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian"
input_path = f"{main_path}/Input/{folder_name}"
out_path = f"{main_path}/Output/{folder_name}"
out_path_plot = out_path + "Intensity/"
output_csv = out_path + "Csv/"


process_directory(input_path, out_path)


# #Conversion du CSV en LaTeX

# Chemin vers ton CSV
csv_path = f"{output_csv}resultats_fwhm.csv"
# Chemin de sortie du fichier LaTeX
latex_path = f"{output_csv}resultats_fwhm_table.tex"

# Lecture du CSV
df = pd.read_csv(csv_path)

# Fonction pour échapper les caractères spéciaux LaTeX
def escape_latex(s):
    s = str(s)
    s = s.replace('\\', r'\\')  # Double les backslash pour LaTeX, mais AVANT tout le reste
    s = s.replace('_', r'\_')
    s = s.replace('&', r'\&')
    s = s.replace('%', r'\%')
    s = s.replace('$', r'\$')
    s = s.replace('#', r'\#')
    s = s.replace('{', r'\{')
    s = s.replace('}', r'\}')
    s = s.replace('~', r'\textasciitilde{}')
    s = s.replace('^', r'\^{}')
    return s

# Construction du tableau tabular seul (sans environnement table)
latex_table = "\\begin{tabular}{ll" + "c" * (df.shape[1] - 2) + "}\n"
latex_table += "\\hline\n\\hline\n"

# En-têtes
columns = [escape_latex(col) for col in df.columns.tolist()]
header_row = " & ".join(columns) + " \\\\\n"
latex_table += header_row
latex_table += "\\hline\n"

# Lignes de données
for _, row in df.iterrows():
    line = " & ".join(escape_latex(val) for val in row.tolist()) + " \\\\\n"
    latex_table += line

latex_table += "\\hline\n"
latex_table += "\\end{tabular}\n"

# Sauvegarde dans un fichier .tex
with open(latex_path, 'w') as f:
    f.write(latex_table)

print(f"✅ Tabular LaTeX sauvegardé dans : {latex_path}")