#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 06:31:35 2024

@author: nbadolo
"""
"""
Ce code est la meilleure version de la serie de codes écrits pour l'analyse et le classement de
mon échantillon suivant la resolubilité de l'étoile. De fait, il traite d'abord les images pour les 
debarasser du bruit et des pixel chauds et froids. Il calcule ensuite le profile radial moyen de l'inensité polarisée à partir du pixel 
le plus brillant qu'il prend pour origine, puis calcule la fwhm( resp. la fwm_h ie. la largeur à une certaine hauteur h par rap. au max) 
du profile radial moyen. La mếme opération est repétée pour l'étoile et sa psf. Ensuite, le ratio = Star_fwhm/psf_fwhm est calculé. 
Si l'étoile n'a pas de psf, alors, la valeur moyenne des fwhm (resp. des fwm_h) des psf d'autres étoiles pour le même filtre lui 
est attribuée comme valeur de psf_fwhm (resp. de psf_fwm_h) afin de pouvoir lui calcluler un ratio. 
Ensuite, un classement des étoiles est fait avec pour critères arbitraires : 
    if ratio >= 1.35, alors l'étoile est résolue (resolved)
    elif  1< ratio< 1.35, alors l'étoile est marginalement résolue (marginal)
    else, l'étoile est non résolue (unresolved). 

Les réultats des profiles radiaux sont enregistrés dans le repertoire  : 
    '/home/nbadolo/Bureau/Aymard/Donnees_sph/sphere_files/profils_radiaux/fwhm_mean_radial_profil/
Les resultats numériques sont consignés dans des tables excells qui sont enregistrées dans le repertoire :
    '/home/nbadolo/Bureau/Aymard/Donnees_sph/sphere_files/csv_folder/fwhm_mean_radial_profil/'
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from AymardPack import process_fits_image as pfi # Pour l'extraction du bruit et des pixels morts et chauds
#from AymardPack import calculate2_fwhm as cal_fwhm #Pour le calcule de la fwhm (mi-hauteur)
from AymardPack import calculate_fwm_f as cal_fwm_f # Pour le calcul de la fwm_f (un hauteur h)

# paramètres crutiaux propres à ma classification
h=0.0095 # la hauteur à la quelle on calcule la largeur du profile (pour la fonction cal_fwm_f).   
#h=0.5 # pour la largeur à mihauteur
resol_threshold = 1.35 # le critère de resolubilité

def radial_profile(image):
    """Calcule le profil radial moyen d'une image 2D en fonction du pixel le plus brillant."""
    
    # Trouver le pixel le plus brillant (maximum de l'intensité)
    y, x = np.indices(image.shape)  # Crée les indices de position x et y
    max_intensity_index = np.unravel_index(np.argmax(image), image.shape)  # Trouve les indices du pixel le plus brillant
    center_y, center_x = max_intensity_index  # Coordonnées du pixel le plus brillant
    center = np.array([center_x, center_y])

    # Calcul des distances radiales par rapport au pixel le plus brillant
    radius = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    radius = radius.astype(int)  # Conversion en entiers pour binning

    # Calcul du profil radial moyen
    radial_sum = np.bincount(radius.ravel(), weights=image.ravel())  # Somme pondérée des intensités
    radial_count = np.bincount(radius.ravel())  # Nombre de pixels par rayon
    radial_mean = radial_sum / radial_count  # Intensité moyenne à chaque distance radiale

    #Interpolation des données manquantes, les NAN
    nans = np.isnan(radial_mean)
    not_nans = ~nans
    radial_mean[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(not_nans), radial_mean[not_nans])

    return radial_mean

#Paramètres constants
mas_per_pixel = 3.4  # Conversion en mas
nDim = 1024 # Taille de l'image (dimension de l'image)
r = np.linspace(1, nDim // 2 - 1, nDim // 2 - 1)  # Distance radiale (en pixels)
r_mas = mas_per_pixel * r  # Conversion des distances radiales en millièmes d'arcsecondes

# Dossier principal contenant les étoiles
log = 'large_log_+'
main_folder = '/home/nbadolo/Bureau/Aymard/Donnees_sph/' + log + '/'
fname1 = 'zpl_p23_make_polar_maps-ZPL_SCIENCE_P23_REDUCED'
fname2 = '-zpl_science_p23_REDUCED'
#fname = fname1 + '_I' + fname2 + '_I.fits' # full intensity
fname = fname1 + '_PI' + fname2 + '_PI.fits' # polarized intensity
#fname = fname1 + '_I' + fname2 + '_I.fits' # full intensity


# Limite de la plage d'affichage
max_pixel = 20
line_width = 2.5  # Épaisseur des lignes

# Dossier de sauvegarde pour les profiles
output_folder_png = '/home/nbadolo/Bureau/Aymard/Donnees_sph/sphere_files/profils_radiaux/fwhm_mean_radial_profil/' + log+ '/png/'
output_folder_pdf = '/home/nbadolo/Bureau/Aymard/Donnees_sph/sphere_files/profils_radiaux/fwhm_mean_radial_profil/' + log+ '/pdf/'
os.makedirs(output_folder_png, exist_ok=True)  # Créer le dossier s'il n'existe pas
os.makedirs(output_folder_pdf, exist_ok=True)  # Créer le dossier s'il n'existe pas

# Stockage des résultats
psf_profiles = {}
star_profiles = {}
psf_fwhms = {}
star_fwhms = {}

# Charger les données PSF et des étoiles
for star_dir in os.listdir(main_folder):
    star_path = os.path.join(main_folder, star_dir)
    print(star_dir)
    for subfolder in ['psf', 'star']:
        subfolder_path = os.path.join(star_path, subfolder)

        if os.path.isdir(subfolder_path):
            for subfolder1 in ['both', 'alone']:
                subfolder_path1 = os.path.join(subfolder_path, subfolder1)
                if os.path.isdir(subfolder_path1):
                    for filter_folder in os.listdir(subfolder_path1):
                        filter_path = os.path.join(subfolder_path1, filter_folder)  # chemin du filtre
                        print(star_dir, filter_folder)
                        if os.path.isdir(filter_path):
                            data_file = os.path.join(filter_path, fname)  
                            
                            if os.path.isfile(data_file):
                                with fits.open(data_file) as hdul:
                                    header = hdul[0].header
                                    filter_names = [
                                        header.get('HIERARCH ESO INS3 OPTI5 NAME'), 
                                        header.get('HIERARCH ESO INS3 OPTI6 NAME')
                                   ]  
                                    
                                    cube = hdul[0].data  
                                    print(f"Data shape: {cube.shape}")  # Vérifiez la forme du cube de données
                                    print(f"Data len: {len(cube[0])}")
                                    frame_size = len(cube[0])
                                    for i in range(cube.shape[0]):  
                                        image = cube[i]
                                        image = pfi(image)
                                        normalized_image = image + np.abs(np.min(image)) + 0.001
                                        
                                        if np.max(normalized_image) == 0:
                                            print("Image normalisée vide, vérifiez l'image d'entrée.")
                                        else:
                                            normalized_image /= np.max(normalized_image)
                                        print(f"Min avant normalisation: {np.min(image)}, Max avant normalisation: {np.max(image)}")
                                        print(f"Min après normalisation: {np.min(normalized_image)}, Max après normalisation: {np.max(normalized_image)}")
                                        
                                        # Calculer le profil radial
                                        profile = radial_profile(normalized_image)
                                        

                                        # Seconde normalisation pour avoir un pic d'intensité égal à 1
                                        max_intensity = np.max(profile)  # Trouver le pic d'intensité
                                        if max_intensity > 0:  # Assurez-vous qu'il y a un pic non nul
                                            profile /= max_intensity  # Normalisation pour que le pic soit égal à 1
                                        else:
                                            print(f"Avertissement: Profil radial avec intensité nulle pour {star_dir}, filtre {filter_names[i]}.")

                                        r = np.linspace(1, frame_size // 2 - 1, frame_size // 2 - 1)  # création d'un tableau de distance radiale    
                                        r_mas = mas_per_pixel * r  #  où r est en pixels et r_mas en millièmes d'arcsecondes

                                        # Vérifier que le profil radial est significatif
                                        if np.max(profile) > 0:
                                            # Stocker les profils radiaux
                                            if subfolder == 'psf':
                                                psf_profiles[(star_dir, filter_names[i])] = profile
                                                # Calculer la FWHM de la PSF
                                                try:
                                                    psf_fwhms[(star_dir, filter_names[i])] = cal_fwm_f(r_mas, profile, h)
                                                except Exception as e:
                                                    print(f"Erreur lors du calcul pour {star_dir}, {filter_names[i]} : {e}")
                                                    psf_fwhms[(star_dir, filter_names[i])] = 1  # Valeur par défaut
                                            else:
                                                star_profiles[(star_dir, filter_names[i])] = profile
                                                # Calculer la FWHM de l'étoile
                                                try:
                                                    star_fwhms[(star_dir, filter_names[i])] = cal_fwm_f(r_mas, profile, h)
                                                except Exception as e:
                                                    print(f"Erreur lors du calcul pour {star_dir}, {filter_names[i]} : {e}")
                                                    star_fwhms[(star_dir, filter_names[i])] = 1  # Valeur par défaut
                                        else:
                                            print(f"Profil radial vide ou sans intensité significative pour {star_dir}, filtre {filter_names[i]}.")

print(f"le fwhm de l'étoile {star_fwhms}, sa psf {psf_fwhms}")


# Calculer les ratios FWHM
ratios = []
for (star, filter_name), star_fwhm in star_fwhms.items():
    if (star, filter_name) in psf_fwhms:
        psf_fwhm = psf_fwhms[(star, filter_name)]
        has_psf = True
    else:
        psf_fwhm_values = [fwhm for (s, f), fwhm in psf_fwhms.items() if f == filter_name]
        if len(psf_fwhm_values) > 0:
            psf_fwhm_values = np.array(psf_fwhm_values, dtype=float)
            psf_fwhm_values[np.isnan(psf_fwhm_values)] = 0
            psf_fwhm = np.mean(psf_fwhm_values)
            has_psf = False
        else:
            psf_fwhm = 1
            has_psf = False

    # Vérification si star_fwhm et psf_fwhm sont valides
    if star_fwhm is not None and psf_fwhm is not None and psf_fwhm > 0:
        ratio = star_fwhm / psf_fwhm
    else:
        ratio = 1

    ratios.append({
        'Star': star,
        'Filter': filter_name,
        'FWHM_Star': star_fwhm,
        'FWHM_PSF': psf_fwhm,
        'Ratio': ratio,
        'Has_PSF': has_psf
    })


# Trouver le meilleur filtre par étoile
best_filters = {}
for _, row in pd.DataFrame(ratios).iterrows():
    star_name = row['Star']
    ratio = row['Ratio']

    if star_name not in best_filters:
        best_filters[star_name] = row
    else:
        current_best_ratio = best_filters[star_name]['Ratio']
        if ratio > current_best_ratio:
            best_filters[star_name] = row

# Convertir le dictionnaire en DataFrame
best_filters_df = pd.DataFrame(best_filters).T
total_stars = len(best_filters_df)
print(f"Nombre total d'étoiles classées : {total_stars}")

# Traitement des étoiles en fonction de leur état
resolution_dict = {}
for _, row in best_filters_df.iterrows():
    star_name = row['Star']
    ratio = row['Ratio']

    if star_name not in resolution_dict:
        resolution_dict[star_name] = {
            'resolved': False,
            'marginal': False,
            'unresolved': False
        }

    if ratio >= resol_threshold:
        resolution_dict[star_name]['resolved'] = True
    elif 1 < ratio < resol_threshold :
        resolution_dict[star_name]['marginal'] = True
    else:
        resolution_dict[star_name]['unresolved'] = True

# Stocker les étoiles en fonction de leur état
resolved_stars = pd.DataFrame(columns=best_filters_df.columns)
marginal_stars = pd.DataFrame(columns=best_filters_df.columns)
unresolved_stars = pd.DataFrame(columns=best_filters_df.columns)

for star, states in resolution_dict.items():
    star_rows = best_filters_df[best_filters_df['Star'] == star]
    
    if states['resolved']:
        resolved_stars = pd.concat([resolved_stars, star_rows])
    elif states['marginal']:
        marginal_stars = pd.concat([marginal_stars, star_rows])
    elif states['unresolved']:
        unresolved_stars = pd.concat([unresolved_stars, star_rows])

# Tracer et enregistrer les profils radiaux superposés
for (star_dir, filter_name), star_profile in star_profiles.items():
    fwhm_value = star_fwhms.get((star_dir, filter_name), np.nan)

    # Vérifier si le FWHM a été calculé
    if fwhm_value is None:
        print(f"FWHM non calculé pour {star_dir}, {filter_name}.")
    elif np.isnan(fwhm_value):
        print(f"FWHM est NaN pour {star_dir}, {filter_name}.")
    else:
        print(f"FWHM calculé pour {star_dir}, {filter_name} : {fwhm_value}")

    psf_profile = psf_profiles.get((star_dir, filter_name))

    if psf_profile is not None:
        limit = min(max_pixel, star_profile.shape[0])
        rayon = np.arange(limit) * mas_per_pixel  # Conversion des pixels en mas

        # Récupérer le ratio FWHM et l'état de l'étoile depuis la liste des ratios
        ratio_info = next((r for r in ratios if r['Star'] == star_dir and r['Filter'] == filter_name), None)
        
        if ratio_info is not None:
            ratio = ratio_info['Ratio']
            resolution_state = 'Resolved' if ratio >= resol_threshold else 'Marginal' if 1 < ratio < resol_threshold else 'Unresolved'
            print(f"Ratio pour {star_dir} (filtre {filter_name}): {ratio:.2f}")
            print(f"État de l'étoile {star_dir} (filtre {filter_name}): {resolution_state}")
        else:
            ratio = 1
            resolution_state = 'Unresolved'
            print(f"Aucun ratio trouvé pour {star_dir}, {filter_name}. Utilisation d'un état par défaut.")

        # Créer le tracé des profils radiaux
        plt.figure(figsize=(10, 6))  # Créer une nouvelle figure
        plt.plot(rayon, star_profile[:limit], label=f"{star_dir} in {filter_name} band", linewidth=line_width)
        plt.plot(rayon, psf_profile[:limit], label=f"{star_dir} - PSF", linestyle='--', linewidth=line_width)
        plt.title(f"Radial profiles of {star_dir} and his PSF in  {filter_name} band", fontsize=12, fontweight='bold')  # Titre en gras
        plt.xlabel("Radius (mas)", fontsize=13, fontweight='bold')  # Axe X en gras
        plt.ylabel("Intensity", fontsize=13, fontweight='bold')  # Axe Y en gras
        plt.xlim(0, max_pixel * mas_per_pixel)  # Limiter l'axe des x
        plt.grid()
        plt.legend(fontsize=12)  # Taille de la police pour la légende

        # Ajouter des annotations pour le ratio et l'état de l'étoile
        plt.annotate(f"FWHM_Ratio: {ratio:.2f}", xy=(0.7, 0.67), xycoords='axes fraction', fontsize=13, color='m',
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))
        plt.annotate(f"State: {resolution_state}", xy=(0.7, 0.6), xycoords='axes fraction', fontsize=13, color='r',
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))

        # Enregistrement de la figure
        try:
            filename_png = os.path.join(output_folder_png, f"{star_dir}_{filter_name}.png")
            filename_pdf = os.path.join(output_folder_pdf, f"{star_dir}_{filter_name}.pdf")
            plt.savefig(filename_png)
            plt.savefig(filename_pdf)
            print(f"Profils enregistrés : {filename_png} et {filename_pdf}")
        except Exception as e:
            print(f"Erreur lors de l'enregistrement des fichiers : {e}")
        
        plt.show()  # Afficher le tracé
        plt.close()  # Fermer la figure pour éviter d'encombrer la mémoire

print(f"Tous les profils ont été enregistrés dans les dossiers suivants : {output_folder_png} et {output_folder_pdf}")

# Enregistrement des résultats dans des fichiers CSV
chemin_csv = '/home/nbadolo/Bureau/Aymard/Donnees_sph/sphere_files/csv_folder/fwhm_mean_radial_profil/'+log+'/'
os.makedirs(chemin_csv, exist_ok=True)  # Créer le dossier s'il n'existe pas
resolved_stars.to_csv(chemin_csv + 'resolved_stars.csv', index=False)
marginal_stars.to_csv(chemin_csv + 'marginal_stars.csv', index=False)
unresolved_stars.to_csv(chemin_csv + 'unresolved_stars.csv', index=False)

# Comptage du nombre d'étoiles par catégorie de résolution
resolved_count = len(resolved_stars)
marginal_count = len(marginal_stars)
unresolved_count = len(unresolved_stars)

# Affichage des résultats

print(resolved_stars)
print(f"Nombre d'étoiles résolues : {resolved_count}")
print(marginal_stars)
print(f"Nombre d'étoiles marginalement résolues : {marginal_count}")
print(unresolved_stars)
print(f"Nombre d'étoiles non résolues : {unresolved_count}")

# Vérification que le total des étoiles classées est 53
total_classified = resolved_count + marginal_count + unresolved_count
print(f"Total des étoiles classées : {total_classified}")
