#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse et classement des étoiles par résolubilité via les FWHM (w_h).
Inclut un calcul classique (cal_fwm_f) et une estimation robuste par Dice.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from AymardPack import process_fits_image as pfi  # Extraction du bruit et des pixels morts/chauds
# from AymardPack import calculate2_fwhm as cal_fwhm # Pour la fwhm à mi-hauteur
from AymardPack import calculate_fwm_f as cal_fwm_f  # Largeur à une hauteur h
from scipy.signal import find_peaks
from scipy import optimize as opt

# paramètres cruciaux propres à la classification
h = 0.0095  # hauteur utilisée par cal_fwm_f (peut être remplacée par 0.5 pour vraie FWHM)
# h = 0.5  # largeur à mi-hauteur
resol_threshold = 1.35  # critère de résolubilité


def radial_profile(image):
    """Calcule le profil radial moyen d'une image 2D en partant du pixel le plus brillant."""
    y, x = np.indices(image.shape)
    center_y, center_x = np.unravel_index(np.argmax(image), image.shape)
    center = np.array([center_x, center_y])

    radius = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2).astype(int)
    radial_sum = np.bincount(radius.ravel(), weights=image.ravel())
    radial_count = np.bincount(radius.ravel())
    radial_mean = radial_sum / radial_count

    # interpolation des NaN éventuels
    nans = np.isnan(radial_mean)
    if np.any(nans):
        radial_mean[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), radial_mean[~nans])

    return radial_mean


def calculate_fwhm_with_dice(r, profile):
    """
    Calcule la FWHM en maximisant le coefficient de Dice entre le profil réel
    et un profil gaussien synthétique.
    """
    if len(profile) < 5 or np.max(profile) == 0:
        return np.nan

    valid_idx = profile > 1e-4  # seuil tolérant
    if np.sum(valid_idx) < 3:
        try:
            return cal_fwm_f(r, profile, h)
        except Exception:
            return np.nan

    r_valid = r[valid_idx]
    profile_valid = profile[valid_idx]
    profile_valid = profile_valid / np.max(profile_valid)

    profile_binary_half = (profile_valid >= 0.5).astype(int)

    def cost_function(sigma):
        if sigma <= 0:
            return 1.0
        gaussian = np.exp(-(r_valid ** 2) / (2 * sigma ** 2))
        gaussian = gaussian / np.max(gaussian)
        gaussian_binary_half = (gaussian >= 0.5).astype(int)

        intersection = np.sum((profile_binary_half == 1) & (gaussian_binary_half == 1))
        size_sum = np.sum(profile_binary_half) + np.sum(gaussian_binary_half)
        dice = 0 if size_sum == 0 else 2.0 * intersection / size_sum
        return 1.0 - dice

    idx_half = np.where(profile_valid >= 0.5)[0]
    if len(idx_half) > 1:
        sigma_initial = (r_valid[idx_half[-1]] - r_valid[idx_half[0]]) / (2 * np.sqrt(2 * np.log(2)))
    else:
        sigma_initial = r_valid[np.argmax(profile_valid)] / 2
    sigma_initial = max(sigma_initial, 0.1)

    try:
        result = opt.minimize_scalar(cost_function, bounds=(0.01, 100), method='bounded')
        sigma_optimal = result.x
        fwhm_dice = 2 * np.sqrt(2 * np.log(2)) * sigma_optimal
        return fwhm_dice
    except Exception:
        # Fallback silencieux si l'ajustement échoue
        try:
            return cal_fwm_f(r, profile, h)
        except Exception:
            return np.nan


#Paramètres constants
mas_per_pixel = 3.4  # Conversion en mas
nDim = 1024 # Taille de l'image (dimension de l'image)
r = np.linspace(1, nDim // 2 - 1, nDim // 2 - 1)  # Distance radiale (en pixels)
r_mas = mas_per_pixel * r  # Conversion des distances radiales en millièmes d'arcsecondes

# Dossier principal contenant les étoiles
#log = 'spherical'
log='large_log_+'
main_folder = '/home/nbadolo/Bureau/Aymard/Donnees_sph/' + log + '/'
fname1 = 'zpl_p23_make_polar_maps-ZPL_SCIENCE_P23_REDUCED'
fname2 = '-zpl_science_p23_REDUCED'
#fname = fname1 + '_I' + fname2 + '_I.fits' # full intensity
fname = fname1 + '_PI' + fname2 + '_PI.fits' # polarized intensity
#fname = fname1 + '_DOLP' + fname2 + '_DOLP.fits' # degree of linear polarization
#fname = fname1 + '_I' + fname2 + '_I.fits' # full intensity


# Limite de la plage d'affichage
max_pixel = 20
line_width = 2.5  # Épaisseur des lignes


# Dossier de sauvegarde pour les profiles
output_folder_png = '/home/nbadolo/Bureau/Aymard/Donnees_sph/sphere_files/profils_radiaux/fwhm_mean_radial_profil/' + log+ '/png/'
output_folder_pdf = '/home/nbadolo/Bureau/Aymard/Donnees_sph/sphere_files/profils_radiaux/fwhm_mean_radial_profil/' + log+ '/pdf/'
output_folder_airy = '/home/nbadolo/Bureau/A_profils_Airy/'
os.makedirs(output_folder_png, exist_ok=True)  # Créer le dossier s'il n'existe pas
os.makedirs(output_folder_pdf, exist_ok=True)  # Créer le dossier s'il n'existe pas
os.makedirs(output_folder_airy, exist_ok=True)  # Créer le dossier s'il n'existe pas

# Dossier pour les stats et figures de classification
classification_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_tables/classification'
os.makedirs(classification_dir, exist_ok=True)

# Stockage des résultats
psf_profiles = {}
star_profiles = {}
psf_fwhms = {}
star_fwhms = {}
psf_fwhms_dice = {}  # FWHM calculé avec le coefficient de Dice
star_fwhms_dice = {}  # FWHM calculé avec le coefficient de Dice

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
                                    frame_size = len(cube[0])
                                    for i in range(cube.shape[0]):  
                                        image = cube[i]
                                        image = pfi(image)
                                        normalized_image = image + np.abs(np.min(image)) + 0.001
                                        
                                        if np.max(normalized_image) > 0:
                                            normalized_image /= np.max(normalized_image)
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
                                                # Calculer la FWHM de la PSF (ancienne méthode)
                                                try:
                                                    psf_fwhms[(star_dir, filter_names[i])] = cal_fwm_f(r_mas, profile, h)
                                                except Exception as e:
                                                    print(f"Erreur lors du calcul pour {star_dir}, {filter_names[i]} : {e}")
                                                    psf_fwhms[(star_dir, filter_names[i])] = 1  # Valeur par défaut
                                                # Calculer la FWHM avec Dice (nouvelle méthode, plus robuste)
                                                try:
                                                    psf_fwhms_dice[(star_dir, filter_names[i])] = calculate_fwhm_with_dice(r_mas, profile)
                                                except Exception:
                                                    psf_fwhms_dice[(star_dir, filter_names[i])] = np.nan
                                            else:
                                                star_profiles[(star_dir, filter_names[i])] = profile
                                                # Calculer la FWHM de l'étoile (ancienne méthode)
                                                try:
                                                    star_fwhms[(star_dir, filter_names[i])] = cal_fwm_f(r_mas, profile, h)
                                                except Exception as e:
                                                    print(f"Erreur lors du calcul pour {star_dir}, {filter_names[i]} : {e}")
                                                    star_fwhms[(star_dir, filter_names[i])] = 1  # Valeur par défaut
                                                # Calculer la FWHM avec Dice (nouvelle méthode, plus robuste)
                                                try:
                                                    star_fwhms_dice[(star_dir, filter_names[i])] = calculate_fwhm_with_dice(r_mas, profile)
                                                except Exception:
                                                    star_fwhms_dice[(star_dir, filter_names[i])] = np.nan
                                        else:
                                            print(f"Profil radial vide ou sans intensité significative pour {star_dir}, filtre {filter_names[i]}.")

# Debug verbose initial (désactivé pour limiter le bruit)
# print(f"le fwhm de l'étoile {star_fwhms}, sa psf {psf_fwhms}")
# print(f"le fwhm DICE de l'étoile {star_fwhms_dice}, sa psf {psf_fwhms_dice}")


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

# Convertir ratios en DataFrame pour analyse
ratios_df = pd.DataFrame(ratios)

# ============================================================================
# ANALYSE STATISTIQUE DE W_H (FWHM) PAR FILTRE - RÉPONSE AU REFEREE
# ============================================================================

print("\n" + "="*80)
print("ANALYSE STATISTIQUE DES LARGEURS À MI-HAUTEUR (w_h = FWHM)")
print("="*80)

# On stocke les messages pour les afficher en bloc à la fin
stats_report_lines = []
summary_paths = {}

# Construire une liste complète des FWHM PSF avec Dice
psf_fwhm_dice_list = []
for (star, filter_name), fwhm_dice in psf_fwhms_dice.items():
    if not np.isnan(fwhm_dice):
        psf_fwhm_dice_list.append({
            'Star': star,
            'Filter': filter_name,
            'w_h_Dice': fwhm_dice,
            'Type': 'PSF'
        })

# Construire une liste complète des FWHM Star avec Dice
star_fwhm_dice_list = []
for (star, filter_name), fwhm_dice in star_fwhms_dice.items():
    if not np.isnan(fwhm_dice):
        star_fwhm_dice_list.append({
            'Star': star,
            'Filter': filter_name,
            'w_h_Dice': fwhm_dice,
            'Type': 'Star'
        })

# Combiner tout dans un DataFrame
all_fwhm_dice = pd.DataFrame(psf_fwhm_dice_list + star_fwhm_dice_list)

# Statistiques par filtre pour les PSF
psf_dice_data = pd.DataFrame(psf_fwhm_dice_list)
stats_source = 'Dice'

# Fallback : si aucune FWHM via Dice (nan ou vide), utiliser cal_fwm_f classique
if len(psf_dice_data) == 0:
    fallback_rows = []
    for (star, filter_name), fwhm_val in psf_fwhms.items():
        if fwhm_val is not None and not np.isnan(fwhm_val):
            fallback_rows.append({
                'Star': star,
                'Filter': filter_name,
                'w_h_Dice': fwhm_val,
                'Type': 'PSF'
            })
    psf_dice_data = pd.DataFrame(fallback_rows)
    stats_source = 'cal_fwm_f'

if len(psf_dice_data) > 0:
    filters_list = psf_dice_data['Filter'].unique()
    
    stats_report_lines.append(f"1. STATISTIQUES DESCRIPTIVES DE w_h (FWHM) PAR FILTRE (PSF de référence, source={stats_source})")
    
    fwhm_stats_list = []
    for filt in sorted(filters_list):
        fwhm_values = psf_dice_data[psf_dice_data['Filter'] == filt]['w_h_Dice'].values
        
        if len(fwhm_values) > 0:
            stats_dict = {
                'Filter': filt,
                'N_obs': len(fwhm_values),
                'Mean (mas)': np.mean(fwhm_values),
                'Median (mas)': np.median(fwhm_values),
                'Std (mas)': np.std(fwhm_values),
                'Min (mas)': np.min(fwhm_values),
                'Q1 (mas)': np.percentile(fwhm_values, 25),
                'Q3 (mas)': np.percentile(fwhm_values, 75),
                'Max (mas)': np.max(fwhm_values),
                'CV (%)': 100 * np.std(fwhm_values) / np.mean(fwhm_values)  # Coefficient de variation
            }
            fwhm_stats_list.append(stats_dict)
            
            stats_report_lines.append(f"{filt}:")
            stats_report_lines.append(f"  • Observations PSF : {len(fwhm_values)}")
            stats_report_lines.append(f"  • Moyenne ± écart-type : {np.mean(fwhm_values):.4f} ± {np.std(fwhm_values):.4f} mas")
            stats_report_lines.append(f"  • Médiane : {np.median(fwhm_values):.4f} mas")
            stats_report_lines.append(f"  • Plage [min, max] : [{np.min(fwhm_values):.4f}, {np.max(fwhm_values):.4f}] mas")
            stats_report_lines.append(f"  • Quartiles Q1-Q3 : [{np.percentile(fwhm_values, 25):.4f}, {np.percentile(fwhm_values, 75):.4f}] mas")
            stats_report_lines.append(f"  • Coefficient de variation (CV) : {100 * np.std(fwhm_values) / np.mean(fwhm_values):.2f}%")
    
    fwhm_stats_df = pd.DataFrame(fwhm_stats_list)
    
    # Sauvegarder les statistiques
    stats_output_path = os.path.join(classification_dir, 'fwhm_statistics_by_filter.csv')
    fwhm_stats_df.to_csv(stats_output_path, index=False)
    summary_paths['stats_csv'] = stats_output_path
    
    # Générer la version LaTeX
    stats_latex_path = os.path.join(classification_dir, 'fwhm_statistics_by_filter.tex')
    latex_table = ""
    
    # Ajouter les en-têtes sur deux lignes (nom sur la première, unité sur la deuxième)
    header_names = []
    header_units = []
    for col in fwhm_stats_df.columns:
        col_str = str(col)
        if '(' in col_str and ')' in col_str:
            # Séparer le nom et l'unité
            name_part = col_str.split('(')[0].strip().replace('_', r'\_')
            unit_part = '(' + col_str.split('(')[1].strip()
            header_names.append(name_part)
            header_units.append(unit_part)
        else:
            header_names.append(col_str.replace('_', r'\_'))
            header_units.append('-')
    
    latex_table += "\\toprule\n"
    latex_table += " & ".join(header_names) + " \\\\\n"
    latex_table += " & ".join(header_units) + " \\\\\n"
    latex_table += "\\midrule\n"
    
    # Ajouter les lignes de données
    for _, row in fwhm_stats_df.iterrows():
        formatted_row = [f"{val:.2f}" if isinstance(val, (int, float)) else str(val).replace('_', r'\_') for val in row.tolist()]
        line = " & ".join(formatted_row) + " \\\\\n"
        latex_table += line
    
    latex_table += "\\bottomrule\n"
    
    with open(stats_latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
        f.flush()
    summary_paths['stats_tex'] = stats_latex_path
    print(f"📄 Tableau LaTeX stats sauvegardé dans : {stats_latex_path}")
    
    # Évaluation de la représentativité (Réponse question 3)
    stats_report_lines.append("3. ÉVALUATION DE LA REPRÉSENTATIVITÉ ET STABILITÉ DE w_h PAR FILTRE")
    
    for filt in sorted(filters_list):
        fwhm_values = psf_dice_data[psf_dice_data['Filter'] == filt]['w_h_Dice'].values
        if len(fwhm_values) > 0:
            cv = 100 * np.std(fwhm_values) / np.mean(fwhm_values)
            mean_val = np.mean(fwhm_values)
            std_val = np.std(fwhm_values)
            
            stats_report_lines.append(f"{filt}:")
            stats_report_lines.append(f"  • Dispersion typique : ±{std_val:.4f} mas ({cv:.1f}% de la moyenne)")
            
            if cv < 5:
                stats_report_lines.append("  ✓ STABLE : Très faible variabilité (CV < 5%)")
                stats_report_lines.append("    → Une seule PSF de référence est suffisamment représentative")
            elif cv < 10:
                stats_report_lines.append("  ⚠ MODÉRÉE : Faible variabilité (5% ≤ CV < 10%)")
                stats_report_lines.append("    → Une PSF de référence est généralement acceptable")
            else:
                stats_report_lines.append("  ⚠ IMPORTANTE : Variabilité notable (CV ≥ 10%)")
                stats_report_lines.append("    → Utiliser une moyenne est recommandée pour réduire l'incertitude")
            
            # Analyse de l'impact sur la classification
            stats_report_lines.append("  • Impact potentiel sur la classification :")
            # Seuil nominal
            seuil = resol_threshold  # 1.35
            # Borne inférieure et supérieure
            w_h_min = np.min(fwhm_values)
            w_h_max = np.max(fwhm_values)
            ratio_avec_min = seuil * (w_h_max / w_h_min)  # Ratio avec PSF_min
            ratio_avec_max = seuil * (w_h_min / w_h_max)  # Ratio avec PSF_max
            
            stats_report_lines.append(f"    - Utiliser PSF_min ({w_h_min:.4f} mas) → ratio multiplié par {w_h_max/w_h_min:.3f}")
            stats_report_lines.append(f"    - Utiliser PSF_max ({w_h_max:.4f} mas) → ratio divisé par {w_h_max/w_h_min:.3f}")
            stats_report_lines.append(f"    - Écart potentiel au seuil {seuil} : ±{100*std_val/mean_val/seuil:.1f}%")
    
    # ========================================================================
    # 2. VISUALISATIONS : HISTOGRAMMES ET BOX PLOTS
    # ========================================================================
    stats_report_lines.append("2. VISUALISATIONS DE w_h PAR FILTRE")
    
    # Créer des figures
    n_filters = len(filters_list)
    
    # Figure 1 : Histogrammes par filtre
    fig, axes = plt.subplots(1, n_filters, figsize=(5*n_filters, 4))
    if n_filters == 1:
        axes = [axes]
    
    for idx, filt in enumerate(sorted(filters_list)):
        fwhm_values = psf_dice_data[psf_dice_data['Filter'] == filt]['w_h_Dice'].values
        
        axes[idx].hist(fwhm_values, bins=max(3, len(fwhm_values)//2), 
                       color='steelblue', alpha=0.7, edgecolor='black')
        axes[idx].axvline(np.mean(fwhm_values), color='red', linestyle='--', 
                         linewidth=2, label=f'Moyenne: {np.mean(fwhm_values):.4f}')
        axes[idx].axvline(np.median(fwhm_values), color='green', linestyle='--', 
                         linewidth=2, label=f'Médiane: {np.median(fwhm_values):.4f}')
        axes[idx].set_xlabel('w_h [mas]', fontsize=10)
        axes[idx].set_ylabel('Fréquence', fontsize=10)
        axes[idx].set_title(f'Filtre {filt}\n(N={len(fwhm_values)})', fontsize=11, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    hist_output_path = os.path.join(classification_dir, 'fwhm_histograms_by_filter.png')
    plt.savefig(hist_output_path, dpi=150, bbox_inches='tight')
    summary_paths['hist_png'] = hist_output_path
    plt.close()
    
    # Figure 2 : Box plots par filtre
    fig, ax = plt.subplots(figsize=(6, 5))

    box_data = [psf_dice_data[psf_dice_data['Filter'] == filt]['w_h_Dice'].values
                for filt in sorted(filters_list)]

    palette = ['#c7dcee', '#a4c2e6', '#7da9d8', '#5f92cb', '#3f7bbf']
    bp = ax.boxplot(box_data, labels=sorted(filters_list), patch_artist=True,
                    notch=True, showmeans=True, meanline=False,
                    meanprops=dict(marker='o', markerfacecolor='white',
                                   markeredgecolor='#2f4a6d', markersize=6,
                                   markeredgewidth=1.4))

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(palette[i % len(palette)])
        patch.set_edgecolor('#2f4a6d')
        patch.set_linewidth(1.4)
    for whisker in bp['whiskers']:
        whisker.set(color='#2f4a6d', linewidth=1.3)
    for cap in bp['caps']:
        cap.set(color='#2f4a6d', linewidth=1.3)
    for median in bp['medians']:
        median.set(color='#d62728', linewidth=2.2)

    ax.set_xlabel('Filter', fontsize=14)
    ax.set_ylabel('w_h (mas)', fontsize=14)
    ax.tick_params(axis='x', labelsize=14, rotation=45)
    ax.tick_params(axis='y', labelsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    median_handle = plt.Line2D([], [], color='#d62728', linewidth=2.2, label='Median')
    ax.legend(handles=[median_handle], loc='upper right', frameon=False, fontsize=14)
    
    plt.tight_layout()
    boxplot_output_path = os.path.join(classification_dir, 'fwhm_boxplot_by_filter.png')
    boxplot_output_path_pdf = os.path.join(classification_dir, 'fwhm_boxplot_by_filter.pdf')
    plt.savefig(boxplot_output_path, dpi=150, bbox_inches='tight')
    plt.savefig(boxplot_output_path_pdf, dpi=300, bbox_inches='tight')
    summary_paths['boxplot_png'] = boxplot_output_path
    summary_paths['boxplot_pdf'] = boxplot_output_path_pdf
    plt.close()
    
    # Figure 3 : Comparaison inter-filtres (scatter plot)
    if n_filters > 1:
        fig, ax = plt.subplots(figsize=(6, 5))
        
        colors_map = {filt: f'C{idx}' for idx, filt in enumerate(sorted(filters_list))}
        
        for filt in sorted(filters_list):
            fwhm_values = psf_dice_data[psf_dice_data['Filter'] == filt]['w_h_Dice'].values
            x_pos = np.random.normal(list(sorted(filters_list)).index(filt), 0.04, len(fwhm_values))
            ax.scatter(x_pos, fwhm_values, alpha=0.6, s=100, label=filt, 
                      color=colors_map[filt], edgecolors='black', linewidth=0.5)
        
        ax.set_xticks(range(n_filters))
        ax.set_xticklabels(sorted(filters_list))
        ax.set_ylabel('w_h (mas)', fontsize=14)
        ax.set_xlabel('Filter', fontsize=14)
        ax.set_title('Intra-filter variability of w_h', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        plt.tight_layout()
        scatter_output_path = os.path.join(classification_dir, 'fwhm_variability_scatter.png')
        plt.savefig(scatter_output_path, dpi=150, bbox_inches='tight')
        summary_paths['scatter_png'] = scatter_output_path
        plt.close()

    else:
        print("\n⚠ Aucune FWHM PSF exploitable (même après fallback). Pas de stats ni de figures générées.")

# Identifier les étoiles sans PSF propre (affichage en fin de script)
stars_without_psf = ratios_df[ratios_df['Has_PSF'] == False][['Star', 'Filter', 'FWHM_PSF']].drop_duplicates()

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

    # Générer le graphique même sans PSF
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
    plt.figure(figsize=(6, 5))  # Créer une nouvelle figure
    plt.plot(rayon, star_profile[:limit], label=f"{star_dir}", linewidth=line_width)
    if psf_profile is not None:
        plt.plot(rayon, psf_profile[:limit], label=f"PSF", linestyle='--', linewidth=line_width)
    #plt.title(f"Radial profiles of {star_dir} and his PSF in  {filter_name} band", fontsize=14, fontweight='bold')  # Titre en gras
    plt.xlabel("R$_\star$(mas)", fontsize=14)  # Axe X en gras
    plt.ylabel("DoLP/$\mathrm{DoLP_{max}}$", fontsize=14)  # Axe Y en gras
    plt.xlim(0, max_pixel * mas_per_pixel)  # Limiter l'axe des x
    #plt.grid()
    plt.legend(fontsize=14)  # Taille de la police pour la légende
    # Détection des pics du profil étoile (prominence à ajuster si besoin)
    peaks, _ = find_peaks(star_profile[:limit], prominence=0.001)
    peak_radii = rayon[peaks]

    # Traits verticaux sur chaque pic détecté
    for rp in peak_radii:
        plt.axvline(rp, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)

    # # Ajouter des annotations pour le ratio et l'état de l'étoile
    # plt.annotate(f"FWHM_Ratio: {ratio:.2f}", xy=(0.7, 0.67), xycoords='axes fraction', fontsize=13, color='m',
    #              bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))
    # plt.annotate(f"State: {resolution_state}", xy=(0.7, 0.6), xycoords='axes fraction', fontsize=13, color='r',
    #              bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.3"))

    plt.annotate(f"{filter_name}", xy=(0.08, 0.9), xycoords='axes fraction', fontsize=13, color='k',
                 )

    # Enregistrement de la figure
    try:
        filename_png = os.path.join(output_folder_png, f"{star_dir}_{filter_name}.png")
        filename_pdf = os.path.join(output_folder_pdf, f"{star_dir}_{filter_name}.pdf")
        plt.savefig(filename_png)
        plt.savefig(filename_pdf)
        plt.savefig(os.path.join(output_folder_airy, f"{star_dir}_{filter_name}.png"))
        plt.savefig(os.path.join(output_folder_airy, f"{star_dir}_{filter_name}.pdf"))
        print(f"Profils enregistrés : {filename_png} et {filename_pdf}")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement des fichiers : {e}")
    
    #plt.show()  # Afficher le tracé
    plt.close()  # Fermer la figure pour éviter d'encombrer la mémoire

print(f"Tous les profils ont été enregistrés dans les dossiers suivants : {output_folder_png} et {output_folder_pdf}")


# Enregistrement des résultats dans des fichiers CSV
chemin_csv = '/home/nbadolo/Bureau/Aymard/Donnees_sph/sphere_files/csv_folder/fwhm_mean_radial_profil/'+log+'/'
os.makedirs(chemin_csv, exist_ok=True)  # Créer le dossier s'il n'existe pas
resolved_stars.to_csv(chemin_csv + 'resolved_stars.csv', index=False)
marginal_stars.to_csv(chemin_csv + 'marginal_stars.csv', index=False)
unresolved_stars.to_csv(chemin_csv + 'unresolved_stars.csv', index=False)

# Créer une table finale consolidée avec tous les objets et leur statut
final_classification = best_filters_df.copy()
final_classification['Statut_Res'] = 'Non résolu'  # Valeur par défaut

# Assigner les statuts de résolution
for _, row in final_classification.iterrows():
    ratio = row['Ratio']
    if ratio >= resol_threshold:
        final_classification.loc[final_classification['Star'] == row['Star'], 'Statut_Res'] = 'Clairement résolu'
    elif 1 < ratio < resol_threshold:
        final_classification.loc[final_classification['Star'] == row['Star'], 'Statut_Res'] = 'Marginalement résolu'

# Enregistrer la table finale consolidée
final_classification.to_csv(chemin_csv + 'all_stars_classification.csv', index=False)
print(f"Table finale consolidée sauvegardée : {chemin_csv}all_stars_classification.csv")

# Enregistrer aussi dans le dossier A_profils_Airy
chemin_airy = '/home/nbadolo/Bureau/A_profils_Airy/'
os.makedirs(chemin_airy, exist_ok=True)  # Créer le dossier s'il n'existe pas
resolved_stars.to_csv(chemin_airy + 'resolved_stars.csv', index=False)
marginal_stars.to_csv(chemin_airy + 'marginal_stars.csv', index=False)
unresolved_stars.to_csv(chemin_airy + 'unresolved_stars.csv', index=False)
final_classification.to_csv(chemin_airy + 'all_stars_classification.csv', index=False)
print(f"Fichiers aussi sauvegardés dans : {chemin_airy}")

# Générer le fichier LaTeX avec seulement les données du tableau
def generate_classification_latex_table(df, latex_path):
    """
    Génère un fichier LaTeX contenant seulement les données du tableau de classification.
    """
    
    # Fonction pour échapper les caractères spéciaux LaTeX SEULEMENT pour les données
    def escape_latex_data(s):
        s = str(s)
        s = s.replace('\\', r'\\')  # Double les backslash pour LaTeX, mais AVANT tout le reste
        s = s.replace('_', r'\_')
        s = s.replace('&', r'\&')
        s = s.replace('%', r'\%')
        s = s.replace('#', r'\#')
        s = s.replace('~', r'\textasciitilde{}')
        s = s.replace('^', r'\^{}')
        return s

    # Fonction pour formater les nombres avec 2 décimales
    def format_number(val):
        try:
            # Tente de convertir en float
            num = float(val)
            return f"{num:.2f}"
        except (ValueError, TypeError):
            # Si ce n'est pas un nombre, retourne tel quel (comme les noms d'étoiles)
            return escape_latex_data(val)

    # Fonction pour convertir les statuts en acronymes
    def convert_status_to_acronym(status):
        status_map = {
            'Clairement résolu': 'CR',
            'Marginalement résolu': 'MR', 
            'Non résolu': 'NR'
        }
        return status_map.get(status, status)

    # Fonction pour convertir les valeurs PSF en français
    def convert_psf_to_french(psf_value):
        if psf_value is True or psf_value == 'True' or psf_value == True:
            return 'oui'
        elif psf_value is False or psf_value == 'False' or psf_value == False:
            return 'non'
        else:
            return str(psf_value)

    # Copie du DataFrame pour les modifications
    df_latex = df.copy()
    
    # Convertir les statuts en acronymes AVANT le renommage des colonnes
    if 'Statut_Res' in df_latex.columns:
        df_latex['Statut_Res'] = df_latex['Statut_Res'].apply(convert_status_to_acronym)
    
    # Convertir les valeurs PSF en français AVANT le renommage
    if 'Has_PSF' in df_latex.columns:
        df_latex['Has_PSF'] = df_latex['Has_PSF'].apply(convert_psf_to_french)
    
    # Renommer les colonnes pour LaTeX APRÈS les conversions
    df_latex.columns = df_latex.columns.str.replace('Ratio', r'$\eta$', regex=False)
    df_latex.columns = df_latex.columns.str.replace('Statut_Res', 'Statut de\\\\l\'enveloppe', regex=False)

    # Lignes de données seulement (pour inclusion dans longtable)
    latex_table = ""
    
    # Lignes de données
    for _, row in df_latex.iterrows():
        # Applique le formatage des nombres et l'échappement LaTeX approprié
        formatted_row = [format_number(val) for val in row.tolist()]
        line = " & ".join(formatted_row) + " \\\\\n"
        latex_table += line

    # Sauvegarde dans un fichier .tex avec force flush
    try:
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_table)
            f.flush()  # Force l'écriture immédiate
        print(f"📄 Tableau LaTeX classification sauvegardé dans : {latex_path}")
        
        # Vérification immédiate
        if os.path.exists(latex_path):
            size = os.path.getsize(latex_path)
            print(f"   ✅ Fichier vérifié : {size} bytes")
        else:
            print(f"   ❌ Fichier non trouvé après écriture !")
            
    except Exception as e:
        print(f"❌ ERREUR dans generate_classification_latex_table : {e}")
        raise e

# Générer le fichier LaTeX avec les données du tableau
latex_data_path = os.path.join(chemin_csv, 'Tables', 'all_stars_classification.tex')
os.makedirs(os.path.dirname(latex_data_path), exist_ok=True)
generate_classification_latex_table(final_classification, latex_data_path)

# Comptage du nombre d'étoiles par catégorie de résolution
resolved_count = len(resolved_stars)
marginal_count = len(marginal_stars)
unresolved_count = len(unresolved_stars)

# Affichage des résultats
print(f"Nombre d'étoiles résolues : {resolved_count}")
print(resolved_stars)
print(f"Nombre d'étoiles marginalement résolues : {marginal_count}")
print(marginal_stars)
print(f"Nombre d'étoiles non résolues : {unresolved_count}")

print(unresolved_stars)

# Total
total_classified = resolved_count + marginal_count + unresolved_count
print(f"Total des étoiles classées : {total_classified}")

# Bloc récapitulatif final (stats + chemins)
print("\n" + "="*80)
print("RÉCAPITULATIF STATISTIQUES w_h / FICHIERS")
print("="*80)
if stats_report_lines:
    for line in stats_report_lines:
        print(line)
else:
    print("Aucune statistique FWHM calculée (PSF manquantes ou données vides).")

if summary_paths:
    print("\nChemins de sauvegarde :")
    for key, path in summary_paths.items():
        print(f"  - {key}: {path}")

# 📊 Camembert (seulement s’il y a au moins une étoile)
if total_classified > 0:
    labels = ['Clairement\nrésolues', 'Marginalement\nrésolues', 'Non\nrésolues']
    sizes = [resolved_count, marginal_count, unresolved_count]
    colors = ['#377eb8', '#4daf4a', '#ff69b4']

    plt.figure(figsize=(6, 5))
    wedges, texts, autotexts = plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 13},
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'}
    )
    plt.setp(texts, fontsize=13)
    plt.setp(autotexts, fontsize=13, color='white')
    plt.axis('equal')

    chart_folder = os.path.join(chemin_csv, "Charts")
    os.makedirs(chart_folder, exist_ok=True)
    pie_path = os.path.join(chart_folder, 'envelope_classification_pie.png')
    plt.savefig(pie_path, dpi=300, bbox_inches='tight')
    print(f"Camembert sauvegardé dans : {pie_path}")
else:
    print("Aucune étoile classifiée, camembert non généré.")

# Vérification que le total des étoiles classées est 53
total_classified = resolved_count + marginal_count + unresolved_count
print(f"Total des étoiles classées : {total_classified}")

# Rappel du chemin de la table LaTeX générée
print(f"\n📄 Table LaTeX de classification générée : {latex_data_path}")

# ============================================================================
# AFFICHAGE FINAL : ÉTOILES SANS PSF DE RÉFÉRENCE PROPRE
# ============================================================================
print(f"\n{'='*80}")
print(f"ÉTOILES SANS PSF DE RÉFÉRENCE PROPRE : {len(stars_without_psf['Star'].unique())}/{len(ratios_df['Star'].unique())}")
print(f"{'='*80}")
if len(stars_without_psf) > 0:
    print(stars_without_psf.to_string(index=False))
    # Sauvegarder dans un fichier CSV
    output_path = '/home/nbadolo/Bureau/Aymard/stars_without_psf.csv'
    stars_without_psf.to_csv(output_path, index=False)
    print(f"\n✅ Liste sauvegardée dans: {output_path}")
else:
    print("✅ Toutes les étoiles ont une PSF de référence propre.")
print(f"{'='*80}\n")
