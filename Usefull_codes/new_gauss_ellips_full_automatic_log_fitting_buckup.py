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
- Fait un fit automatique de l’ellipse (centre, axes, angle) en maximisant le coefficient de Dice.
- Gère les cas où l’ellipse sortirait du cadre (robuste pour petits nSubDim).
- Affiche et sauvegarde pour chaque image :
    - L’image log(PI) avec le contour de l’ellipse ajustée et le centre.
    - Les axes, labels, colorbar et annotations homogènes pour publication.
- Compile les résultats morphologiques dans un fichier CSV pour chaque étoile et mode d’observation.

Ce script est conçu pour produire des figures et des mesures directement exploitables pour la publication scientifique.
"""

# Importation des bibliothèques nécessaires
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from math import pi, cos, sin
from astropy.nddata import Cutout2D
from astropy.io import fits
import scipy.optimize as opt
from skimage import measure, draw
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
from astropy.wcs import WCS





# --- Ajout : extraction des distances depuis une table CSV ---
def read_csv_distances(csv_path):
    """
    Lit un fichier CSV contenant les distances des étoiles.
    Suppose que la première colonne est le nom de l'objet et la colonne 'Distance' existe.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    # Renomme la colonne du nom de l'objet pour correspondre à la jointure
    if 'Object' in df.columns:
        df = df.rename(columns={'Object': 'Étoile'})
    return df


# Adapter le chemin du fichier CSV
csv_dist_path = '/home/nbadolo/Bureau/Aymard/Tables/ML/Hipparcos/Hip/Sample_hip.csv'
dist_table = read_csv_distances(csv_dist_path)
# Harmonisation automatique des noms d'étoiles pour la jointure
def clean_star_name(name):
    # Remplace les underscores par des espaces, retire les points, met en minuscules et retire les espaces multiples
    name = str(name)
    name = name.replace('_', ' ')
    name = name.replace('.', '')
    name = name.lower()
    name = re.sub(r'\s+', ' ', name).strip()
    # Applique le mapping manuel si le nom correspond
    return star_name_map.get(name, name)

# Mapping manuel pour les cas ambigus de noms d'étoiles
star_name_map = {
    'l02 pup': 'l2 pup',
    'bet gru': 'beta gru',
    'pi01 gru': 'pi01 gru',
    'l2 pup': 'l2 pup',
    'gy aql': 'gy aql',
    '17 lep': '17 lep',
    # Ajoute ici d'autres corrections spécifiques si besoin
}

# Appliquer le nettoyage sur la colonne de la table des distances
dist_table['star_clean'] = dist_table['Étoile'].apply(clean_star_name)
from matplotlib.patches import Arc

# Fonction pour calculer le profil radial moyen d'une image 2D
# en fonction du pixel le plus brillant.
# Cette fonction est utilisée pour analyser la distribution de l'intensité


star_filters = {
    'Y_Pav':    ('both', ['V_N_R']),
    'R_Hor':    ('both', ['V_N_R']),
    'R_Scl':    ('both', ['V_N_R']),
    'Y_Scl':    ('both', ['V_N_R']),
    'R_Crt':    ('alone', ['N_R']),
    'SW_Col':   ('both', ['V_N_R']),
    'W_Hya':    ('both',  ['Cnt820_Cnt748']),
    'SW_Vir':   ('alone', ['N_R']),
    'V_Hya':    ('both', ['V_N_R']),
    'Alpha_Her':('both', ['V_Cnt748']),
    'R_Hya':     ('alone', ['CntHa']),
    'Chi_Cyg':  ('both', ['V_Cnt748']),
    'Z_Eri':     ('both', ['V_N_R']),
    'R_Peg':     ('both', ['V_N_R']),
    'BW_Oct':    ('both', ['V_N_R']),
    'AC_Cet':    ('both', ['V_N_R']),
    'DZ_Aqr':    ('both', ['V_N_R']),
    'Z_Peg':     ('both', ['V_N_R']),
    'W_Peg':     ('both', ['V_N_R']),
    'RT_Vir':   ('alone', ['Cnt820']),
    'RX_Lep':   ('alone', ['CntHa']),
    'Beta_Gru': ('both', ['V_N_R']), 
    'T_Mic':    ('both', ['V_N_R']),
    'R_Dor':    ('both',  ['Cnt820_Cnt748']),
    'AK_Hya':   ('alone', ['N_R']),
    'R_Leo':    ('both', ['V_Cnt748']),
    'BK_Vir':   ('alone', ['Cnt820']),
    'T_Cet':    ('both',  ['CntHa_B_Ha']),
    'U_Del':    ('alone', ['CntHa']),
    'U_Her':    ('alone', ['VBB']),
    'W_Aql':    ('alone', ['VBB']),
    'V_PsA':    ('alone', ['N_R']),
    'R_Aql':    ('alone', ['N_R']),
    'S_Pav':    ('alone', ['N_R']),
    'GY_Aql':   ('alone', ['VBB']),
    'SV_Aqr':   ('alone', ['VBB']),
    'Ups_Cet':  ('both', ['V_N_R']),
    'V1943_Sgr':('both', ['V_N_R']),
    'Psi_Phe':  ('both', ['V_N_R']),
    'S_Lep':  ('alone', ['I_PRIM']),
    '17_Lep':   ('alone', ['I_PRIM']),
    'L02_Pup':  ('both', ['V_N_R']),
    'CW_Cnc': ('alone', ['I_PRIM']),
    'Mira':  ('both',  ['CntHa_B_Ha']),
    'Pi.01_Gru':('both',  ['V_N_R']),
   
}

star_specific_filter = {
    'Y_Pav':    'N_R',
    'R_Hor':    'N_R',
    'R_Scl':    'N_R',
    'Y_Scl':    'V',
    'R_Crt':    'N_R',
    'SW_Col':   'N_R',
    'W_Hya':    'Cnt748',
    'SW_Vir':   'N_R',
    'V_Hya':    'N_R',
    'Alpha_Her': 'V',
    'R_Hya':    'CntHa',
    'Chi_Cyg':  'Cnt748',
    'Z_Eri':    'V',
    'R_Peg':    'V',
    'BW_Oct':   'N_R',
    'AC_Cet':   'V',
    'DZ_Aqr':   'V',
    'Z_Peg':    'N_R',
    'W_Peg':     'N_R',
    'RT_Vir':   'Cnt820',
    'RX_Lep':   'CntHa',
    'Beta_Gru': 'V',
    'T_Mic':    'N_R',
    'R_Dor':    'Cnt820',
    'AK_Hya':   'N_R',
    'R_Leo':    'V',
    'BK_Vir':   'Cnt820',
    'T_Cet':    'B_Ha',
    'U_Del':    'CntHa',
    'U_Her':    'VBB',
    'W_Aql':    'VBB',
    'V_PsA':    'N_R',
    'R_Aql':    'N_R',
    'S_Pav':    'N_R',
    'GY_Aql':   'VBB',
    'SV_Aqr':   'VBB',
    'Ups_Cet':  'N_R',
    'V1943_Sgr': 'N_R',
    'Psi_Phe':  'V',
    'S_Lep':    'I_PRIM',
    '17_Lep':   'I_PRIM',
    'L02_Pup':  'V',
    'CW_Cnc':   'I_PRIM',
    'Mira':     'B_Ha',
    'Pi.01_Gru': 'N_R',
}

clearly_resolved = [
    "AK_Hya", "R_Hya", "U_Her", "S_Pav", "Mira", "W_Aql", "R_Crt", "R_Leo",
    "R_Dor", "BK_Vir", "V_PsA", "SW_Col", "GY_Aql", "SW_Vir", "RT_Vir",
    "W_Hya", "L02_Pup", "W_Peg"
]


results = []
def generate_morphology_latex_table(df, latex_path):
    """
    Génère un fichier LaTeX contenant le tableau des résultats morphologiques avec entête personnalisée et arrondi à 3 chiffres après la virgule pour les nombres si besoin.
    :param df: DataFrame contenant les résultats morphologiques
    :param latex_path: Chemin de sauvegarde du fichier LaTeX  
    """
    import re
    def escape_latex_data(s):
        s = str(s)
        s = s.replace('\\', r'\\')
        s = s.replace('_', r'\_')
        s = s.replace('&', r'\&')
        s = s.replace('%', r'\%')
        s = s.replace('#', r'\#')
        s = s.replace('~', r'\textasciitilde{}')
        s = s.replace('^', r'\^{}')
        return s
    def format_latex_number(val):
        sval = str(val)
        if re.match(r'^-?\d+\.\d+$', sval):
            decimals = sval.split('.')[-1]
            if len(decimals) > 3:
                return f"{float(val):.3f}"
            else:
                return sval
        else:
            return sval
    # Entête personnalisée
    custom_header = (
        "Star & Filter & "
        "\\begin{tabular}{c}$a$ \\ (mas) \end{tabular} & "
        "\\begin{tabular}{c} $\\sigma_a$\\ (mas)\\end{tabular} & "
        "\\begin{tabular}{c} b\\ (mas) \end{tabular} & "
        "\\begin{tabular}{c} $\\sigma_b$ \\ (mas)\\end{tabular} & "
        "\\begin{tabular}{c} $D$ \\ (pc) \end{tabular} & "
        "\\begin{tabular}{c} $a$ \\ (UA) \end{tabular} & "
        "$b/a$ & $e$ & "
        "\\begin{tabular}{c} $X_0$ \\ (mas) \end{tabular} & "
        "\\begin{tabular}{c} $Y_0$\\ (mas)\\end{tabular} & "
        "\\begin{tabular}{c} $\\theta$\\ (deg) \end{tabular} & $\\chi_{red}^2$ \\" 
    )
    ncols = len(df.columns)
    latex_table = ""
    latex_table += f"\\begin{{tabular}}{{{'c'*ncols}}}\n"
    latex_table += "\\toprule\n"
    latex_table += custom_header + "\n"
    latex_table += "\\midrule\n"
    for _, row in df.iterrows():
        formatted_row = [escape_latex_data(format_latex_number(val)) for val in row]
        line = " & ".join(formatted_row) + " \\\\"
        latex_table += line + "\n"
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    with open(latex_path, 'w') as f:
        f.write(latex_table)

# Fonction principale pour l'analyse morphologique des images polarisées
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
def radial_profile(image):
    """
    Calcule le profil radial moyen d'une image 2D autour de son centre.
    """
    y, x = np.indices(image.shape)
    center = np.array([(x.max() - x.min())/2.0, (y.max() - y.min())/2.0])
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), weights=image.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)
    return radialprofile


fold_name ='First'
#large_log_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log_+/'
#large_log_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/newly_resolved/'
#large_log_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/clearly_resolved/'
#large_log_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/ICHTUS/'
#large_log_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/BK_Vir/'
#large_log_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/Already_observed/'
large_log_dir = f'/home/nbadolo/Bureau/Aymard/Donnees_sph/{fold_name}/'


def log_image(folder_name, star_name, obsmod, star_specific_filter=None):
    import time
    start_time = time.time()
    # Récupération de la distance pour l'étoile (une seule fois)
    distance_star = None
    if 'dist_table' in globals():
        star_clean = star_name.lower().replace('_', ' ').replace('.', '').strip()
        match = dist_table[dist_table['Étoile'].str.lower().str.replace('_',' ').str.replace('.','').str.strip() == star_clean]
        if not match.empty:
            distance_star = match.iloc[0]['Distance']
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
    
    # Dictionnaire pour la taille du champ de vue DoLP selon l'étoile
    custom_dolp_cutout_size = {
    'W_Aql': 320,
    'Mira': 260,
    'RT_Vir': 280,
    'W_Peg': 280,
    }

    nSubDim_DOLP = custom_dolp_cutout_size.get(star_name, 200)
    size_DOLP = (nSubDim_DOLP, nSubDim_DOLP)
    #lst_threshold = [0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]
    lst_threshold = np.linspace(0.005, 0.1, 50)  # 50 seuils de 0.5% à 10% 
    #lst_threshold = np.linspace(0, 0.1, 100)  # 100 seuils de 0% à 10%
    pix2mas = 3.4
    position = (nDim // 2, nDim // 2)
    
    label_size_small_panel = 14
    label_size_great_panel = 18
    label_size = label_size_small_panel
    
    # Calcul des limites en mas
    # pour l'affichage
    x_min = -pix2mas * nSubDim // 2
    x_max = pix2mas * (nSubDim // 2 - 1)
    y_min = -pix2mas * nSubDim // 2
    y_max = pix2mas * (nSubDim // 2 - 1)

    # Nettoyage complet des répertoires de sortie avant traitement
    outdir_panels = f'/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Morphologies_contours/Panels'
    outdir_uniq = f'/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Morphologies_contours/Unique'
    outdir = f'/home/nbadolo/Bureau/Aymard/Donnees_sph/{folder_name}/{star_name}/plots/fits/log_scale/fully_automatic/'
    
    # Création des répertoires s'ils n'existent pas
    os.makedirs(outdir_panels, exist_ok=True)
    os.makedirs(outdir_uniq, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    #os.makedirs(dolp_specific_dir, exist_ok=True)
    
    # print(f"Nettoyage complet des répertoires pour {star_name}...")
    
    # # Supprime TOUS les fichiers dans outdir2 (répertoire global)
    # all_files_global = glob.glob(os.path.join(outdir2, '*'))
    # for file_path in all_files_global:
    #     if os.path.isfile(file_path):  # Ne supprime que les fichiers, pas les dossiers
    #         try:
    #             os.remove(file_path)
    #             print(f"  Fichier global supprimé : {os.path.basename(file_path)}")
    #         except Exception as e:
    #             print(f"  Erreur lors de la suppression de {file_path} : {e}")
    
    # # Supprime TOUS les fichiers dans outdir (répertoire local)
    # all_files_local = glob.glob(os.path.join(outdir, '*'))
    # for file_path in all_files_local:
    #     if os.path.isfile(file_path):  # Ne supprime que les fichiers, pas les dossiers
    #         try:
    #             os.remove(file_path)
    #             print(f"  Fichier local supprimé : {os.path.basename(file_path)}")
    #         except Exception as e:
    #             print(f"  Erreur lors de la suppression de {file_path} : {e}")
    
    # files_global_count = len([f for f in all_files_global if os.path.isfile(f)]) if all_files_global else 0
    # files_local_count = len([f for f in all_files_local if os.path.isfile(f)]) if all_files_local else 0
    
    # if files_global_count > 0 or files_local_count > 0:
    #     print(f"Nettoyage terminé : {files_global_count} fichiers globaux et {files_local_count} fichiers locaux supprimés.")
    # else:
    #     print("Aucun fichier à supprimer.")

    

    for fltr in lst_fltr_star2:
        print(f"Début traitement étoile : {star_name}", flush=True)
        fdir_star_fltr = os.path.join(fdir_star, fltr)
        fname1 = 'zpl_p23_make_polar_maps-ZPL_SCIENCE_P23_REDUCED'
        fname2 = '-zpl_science_p23_REDUCED'
        file_PI_star = os.path.join(fdir_star_fltr, fname1 + '_PI' + fname2 + '_PI.fits')
        file_DOLP_star = os.path.join(fdir_star_fltr, fname1 + '_DOLP' + fname2 + '_DOLP.fits')
        if not os.path.exists(file_PI_star):
            print(f"Fichier manquant : {file_PI_star}")
        if not os.path.exists(file_DOLP_star):
            print(f"Fichier DoLP manquant : {file_DOLP_star}")
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

        # Extraction du plan z pour DoLP
        hdu_dolp = fits.open(file_DOLP_star)
        data_dolp = hdu_dolp[0].data

        # Affichage des informations
        for z in range(n_fsize):
            intensity = data[z, :, :]
            cutout = Cutout2D(intensity, position=position, size= size )
            sub_v = cutout.data
            
            # Gestion des cubes DoLP 2D ou 3D
            if data_dolp.ndim == 3:
                dolp_z = data_dolp[z, :, :]
            else:
                dolp_z = data_dolp
            # Découpe DoLP avec une taille plus grande pour le champ de vue
            cutout_dolp = Cutout2D(dolp_z, position=position, size=size_DOLP)
            sub_v_dolp = cutout_dolp.data
            # hdu_dolp.close()  # Retiré car hdu_dolp est une HDUList, pas un PrimaryHDU

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

            # Correction : garantir que a_f >= b_f
            a_major = max(a_f, b_f)
            b_minor = min(a_f, b_f)
            diameter_mas = 2 * a_major * pix2mas
            diameter_minor_mas = 2 * b_minor * pix2mas
            diameter_err_mas = 2 * pix2mas  # erreur simple : ±1 pixel sur chaque demi-axe
            diameter_minor_err_mas = 2 * pix2mas
            ellipticity = 1 - (b_minor / a_major)
            results.append({
                'Étoile': star_name,
                'Filtre': fltr_arr[z],
                'D_maj_mas': diameter_mas,
                'sigma_D_maj_mas': diameter_err_mas,
                'D_min_mas': diameter_minor_mas,
                'sigma_D_min_mas': diameter_minor_err_mas,
                'b_a': b_minor / a_major,  # rapport b/a pour q dans le tableau
                'e': ellipticity,
                'X0_mas': x_centroid_mas,
                'Y0_mas': y_centroid_mas,
                'theta_deg': np.degrees(theta_f),
                'Cout': best_cost,
                'Contraste': best_threshold
            })
            print(f"Traitement : filtre={fltr_arr[z]}, seuil={best_threshold:.4f}, fit_cost={best_cost:.4f}")

            # Plot PI avec contour et centre (champ de vue standard)
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
            cbar.ax.tick_params(labelsize=14, width=1.2)
            # Affiche l’exposant/scientific notation à côté de la colorbar
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()
            offset_text = cbar.ax.yaxis.get_offset_text()
            offset_str = offset_text.get_text()
            offset_text.set_visible(False)
            cbar.ax.text(1.25, 0.5, offset_str, transform=cbar.ax.transAxes, fontsize=14, color='black', va='center', ha='left')

            # Adapter les coordonnées des contours et du centre
            x_contour_mas = (Ell_rot[0, :] - nSubDim // 2) * pix2mas
            y_contour_mas = (Ell_rot[1, :] - nSubDim // 2) * pix2mas
            x_centroid_mas = (x_f - nSubDim // 2) * pix2mas
            y_centroid_mas = (y_f - nSubDim // 2) * pix2mas
            ax.plot(x_contour_mas, y_contour_mas, color='cyan', linewidth=2, linestyle='--')
            ax.scatter([x_centroid_mas], [y_centroid_mas], color='red', marker='x')
            ax.set_xlabel("Relative RA (mas)", fontsize=14)
            ax.set_ylabel("Relative Dec (mas)", fontsize=14)
            ax.tick_params(axis='both', labelsize=14, width=1.2)
            ax.locator_params(axis='x', nbins=5)
            ax.locator_params(axis='y', nbins=5)
            ax.text(0.02, 0.95, f'{star_name2}', transform=ax.transAxes, fontsize=14, color='white', va='top')
            ax.text(0.02, 0.02, f'{fltr_arr[z]}', transform=ax.transAxes, fontsize=14, color='white', va='bottom')

            plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10)
            fig_name = f'{star_name}_{obsmod}_{fltr_arr[z]}_{z}_unique_max_contour_for_Pol_Intensity'
            plt.savefig(os.path.join(outdir, fig_name + '.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(outdir, fig_name + '.pdf'), dpi=300, bbox_inches='tight') 
            plt.savefig(os.path.join(outdir_uniq, fig_name + '.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(outdir_uniq, fig_name + '.pdf'), dpi=300, bbox_inches='tight')
            print(f"Figure contour sauvegardée : {os.path.join(outdir_uniq, fig_name + '.png')}", flush=True)
            #plt.savefig(os.path.join(outdir, fig_name + '.eps'), format='eps', dpi=300, bbox_inches='tight')
            #plt.show() 
            plt.close()

            # === Creation du panel DoLP+contours  & PI+fit ===

            fig_panel, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            # DoLP+isocontours à gauche
            # Recalcule les axes pour le champ DoLP élargi
            x_min_DOLP = -pix2mas * nSubDim_DOLP // 2
            x_max_DOLP = pix2mas * (nSubDim_DOLP // 2 - 1)
            y_min_DOLP = -pix2mas * nSubDim_DOLP // 2
            y_max_DOLP = pix2mas * (nSubDim_DOLP // 2 - 1)
            im1 = ax1.imshow(sub_v_dolp, cmap='plasma', origin='lower', vmin=np.nanmin(sub_v_dolp), vmax=np.nanmax(sub_v_dolp), extent=[x_min_DOLP, x_max_DOLP, y_min_DOLP, y_max_DOLP])
            # Niveaux de contours automatiques selon la plage DoLP
            dolp_min = np.nanmin(sub_v_dolp)
            dolp_max = np.nanmax(sub_v_dolp)
            contour_levels = np.linspace(dolp_min, dolp_max, 5)
            cs = ax1.contour(sub_v_dolp, levels=contour_levels, colors='white', linewidths=1.5, origin='lower', extent=[x_min_DOLP, x_max_DOLP, y_min_DOLP, y_max_DOLP])
            ax1.clabel(cs, inline=True, fontsize=10, fmt='%.2f')
            # Ajout de deux cercles autour de l'étoile centrale
            # Bloc complet cercles fixes DoLP, affichage et sauvegarde
            from matplotlib.patches import Circle
            center_x = (x_max_DOLP + x_min_DOLP) / 2
            center_y = (y_max_DOLP + y_min_DOLP) / 2
            R_sun_cm = 6.957e10
            if 'affected_stars_dolp' not in globals():
                global affected_stars_dolp
                affected_stars_dolp = []
            star_name_clean = clean_star_name(star_name)
            match_row = dist_table[dist_table['Étoile'].apply(clean_star_name) == star_name_clean]
            use_fixed_circles = False
            radius_star_rsun = None
            distance_star_pc = None
            if not match_row.empty:
                radius_star_rsun = match_row.iloc[0]['Radius']
                distance_star_pc = match_row.iloc[0]['Distance']
                if (distance_star_pc < 150) or (radius_star_rsun > 500):
                    use_fixed_circles = True
                    affected_stars_dolp.append({'Étoile': star_name, 'Distance_pc': distance_star_pc, 'Radius_Rsun': radius_star_rsun})
            # Création des cercles en rayons stellaires
            if distance_star_pc is not None and radius_star_rsun is not None and radius_star_rsun > 0 and distance_star_pc > 0:
                R_sun_cm = 6.957e10
                radius_star_cm = radius_star_rsun * R_sun_cm
                distance_star_cm = distance_star_pc * 3.0857e18
                radius_star_rad = radius_star_cm / distance_star_cm
                radius_stellar_mas = radius_star_rad * 206265 * 1000
                if use_fixed_circles:
                    radii_rstar = [3, 5, 10]  # 3 cercles en R*
                    linestyles = ['--', '-.', ':']
                else:
                    radii_rstar = [3, 5, 10, 20]  # 4 cercles en R*
                    linestyles = ['--', '-.', ':', '-.']
                radii_mas = [r * radius_stellar_mas for r in radii_rstar]
                circles = [Circle((center_x, center_y), r_mas, edgecolor='orange', facecolor='none', lw=2, linestyle=ls)
                           for r_mas, ls in zip(radii_mas, linestyles)]
                for circ in circles:
                    ax1.add_patch(circ)
                legend_unit = 'R$_\star$'
            else:
                print(f"⚠️ Distance ou rayon stellaire non trouvés pour l'étoile {star_name}, cercles non tracés.")
            # Croix rouge au centre
            ax1.scatter([center_x], [center_y], color='red', marker='x', s=80, zorder=10)
            # Ajout d'une barre d'échelle physique en R* en bas à droite
            if distance_star_pc is not None and radius_star_rsun is not None and radius_star_rsun > 0 and distance_star_pc > 0:
                # Barre d'échelle en R*
                scale_val = 10  # 10 R*
                R_sun_cm = 6.957e10
                radius_star_cm = radius_star_rsun * R_sun_cm
                distance_star_cm = distance_star_pc * 3.0857e18
                radius_star_rad = radius_star_cm / distance_star_cm
                radius_stellar_mas = radius_star_rad * 206265 * 1000
                scale_mas = scale_val * radius_stellar_mas
                legend_unit = 'R$_\star$'
                # Conversion mas -> largeur en axes (champ DoLP)
                bar_length_axes = scale_mas / (x_max_DOLP - x_min_DOLP)
                x_bar_axes = 0.90 - bar_length_axes  # fin à 0.98
                y_bar_axes = 0.04  # bas
                ax1.plot([x_bar_axes, x_bar_axes + bar_length_axes], [y_bar_axes, y_bar_axes], color='white', lw=3, transform=ax1.transAxes, solid_capstyle='butt')
                ax1.text(x_bar_axes + bar_length_axes/2, y_bar_axes + 0.01, f'{scale_val:.0f}{legend_unit}', color='white', fontsize=14, ha='center', va='bottom',  transform=ax1.transAxes)
            else:
                print(f"⚠️ Distance ou rayon stellaire non trouvés pour l'étoile {star_name}, barre d'échelle non tracée.")
            

            ax1.set_xlabel('Relative RA (mas)', fontsize=label_size)
            ax1.set_ylabel('Relative Dec (mas)', fontsize=label_size)

            # #Personnalisation des labels selon l'étoile pour les grands panels
            # if star_name2 in ['U Her','R Crt', 'RX Lep', 'SV Aqr', 'R Peg', 'Chi Cyg', 'R Hya', 'T Mic']:
            #     ax1.set_xlabel('Relative RA (mas)', fontsize=label_size)
            # else:
            #     ax1.set_xlabel('', fontsize=label_size)

            # if star_name2 in ['U Her', 'RX Lep', 'R Peg', 'R Hya']:
            #     ax1.set_ylabel('Relative Dec (mas)', fontsize=label_size)
            # else:
            #     ax1.set_ylabel('', fontsize=label_size)

            ax1.tick_params(axis='both', labelsize=label_size, width=1.2)
            # for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            #     label.set_fontweight('bold')
            ax1.locator_params(axis='x', nbins=5)
            ax1.locator_params(axis='y', nbins=5)

            # ax1.set_xticks([])
            # ax1.set_yticks([])
            # ax1.tick_params(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)

            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes('right', size='5%', pad=0.03)
            cb1 = fig_panel.colorbar(im1, cax=cax1, orientation='vertical')
            cb1.ax.tick_params(labelsize=label_size)
            cmapProp = {'drawedges': True}            
            cb1.formatter.set_powerlimits((0, 0))
            cb1.ax.yaxis.get_offset_text().set(size=label_size)

            ax1.text(0.02, 0.95, f'{star_name2}', transform=ax1.transAxes, fontsize=label_size, color='white', va='top')
            ax1.text(0.02, 0.02, f'{fltr_arr[z]}', transform=ax1.transAxes, fontsize=label_size, color='white', va='bottom')

            # PI + ellipse à droite
            im2 = ax2.imshow(
                np.log10(sub_v + np.abs(np.min(sub_v)) + 10),
                cmap='inferno',
                origin='lower',
                extent=[x_min+1, x_max, y_min+1, y_max]
            )
            # Ajout orientation N-W en bas à droite sur la figure PI (même position que la légende UA du DoLP)
            arrow_len = 0.03 * (x_max - x_min)
            x_arrow = x_max - 0.04 * (x_max - x_min)
            y_arrow = y_min + 0.04 * (y_max - y_min)
            # Flèche Nord (verticale vers le haut)
            ax2.arrow(x_arrow, y_arrow, 0, arrow_len, head_width=0.02*arrow_len, head_length=0.04*arrow_len, fc='white', ec='white', lw=2)
            # Flèche Ouest (horizontale vers la gauche)
            ax2.arrow(x_arrow, y_arrow, -arrow_len, 0, head_width=0.02*arrow_len, head_length=0.04*arrow_len, fc='white', ec='white', lw=2)
            offset_label = 8
            ax2.text(x_arrow, y_arrow + arrow_len + offset_label, 'N', color='white', fontsize=label_size, ha='center', va='bottom')
            ax2.text(x_arrow - arrow_len - offset_label, y_arrow, 'W', color='white', fontsize=label_size, ha='right', va='center')
            #ax2.set_title('Morphologie PI + ellipse', fontsize=label_size)
            ax2.set_xlabel('Relative RA (mas)', fontsize=label_size)
            # # Personnalisation des labels selon l'étoile pour les grands panels
            # if star_name2 in ['U Her','R Crt', 'RX Lep', 'SV Aqr', 'R Peg', 'Chi Cyg', 'R Hya', 'T Mic']:
            #     ax2.set_xlabel('Relative RA (mas)', fontsize=label_size)
            # else:
            #     ax2.set_xlabel('', fontsize=label_size)
            #ax2.set_ylabel('Relative Dec (mas)', fontsize=label_size)
            ax2.tick_params(axis='both', labelsize=label_size, width=1.2)
            # for label in ax2.get_xticklabels() + ax2.get_yticklabels():
            #     label.set_fontweight('bold')
            ax2.locator_params(axis='x', nbins=5)
            ax2.locator_params(axis='y', nbins=5)
            ax2.axes.yaxis.set_ticklabels([])  # Pas de labels y sur la 2e image

            # ax2.set_xticks([])
            # ax2.set_yticks([])
            # ax2.tick_params(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)

            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes('right', size='5%', pad=0.03)
            cb2 = fig_panel.colorbar(im2, cax=cax2, orientation='vertical')
            cb2.ax.tick_params(labelsize=label_size)
            cb2.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
            #ax2.plot(x_contour_mas, y_contour_mas, color='cyan', linewidth=2, linestyle='--')
            ax2.scatter([x_centroid_mas], [y_centroid_mas], color='red', marker='x')
            ax2.text(0.02, 0.95, f'{star_name2}', transform=ax2.transAxes, fontsize=label_size, color='white', va='top')
            ax2.text(0.02, 0.02, f'{fltr_arr[z]}', transform=ax2.transAxes, fontsize=label_size, color='white', va='bottom')

            plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10, wspace=0.15)
            fig_panel_name = f'{star_name}_{obsmod}_{fltr_arr[z]}_{z}_PI_DoLP_panel'
            plt.savefig(os.path.join(outdir, fig_panel_name + '.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(outdir, fig_panel_name + '.pdf'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(outdir_panels +'/png', fig_panel_name + '.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(outdir_panels +'/pdf', fig_panel_name + '.pdf'), dpi=300, bbox_inches='tight')
            if fltr_arr[z] == star_specific_filter.get(star_name, None):
                specific_dir = os.path.join(outdir_panels, 'specific')
                if not os.path.exists(specific_dir):
                    os.makedirs(specific_dir)
                # On ne sauvegarde que si aucune image pour ce filtre n'existe déjà
                already_exists = any(str(fltr_arr[z]) in fname and star_name in fname for fname in os.listdir(specific_dir))
                if not already_exists:
                    fig_panel_name_unique = f"{star_name}_{mode}_{fltr_arr[z]}_PI_DoLP_panel"
                    fig_path = os.path.join(specific_dir, fig_panel_name_unique + '.png')
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    print(f"Figure panel PI+DoLP sauvegardée : {fig_path}", flush=True)
                else:
                    print(f"Panel déjà présent pour {star_name} [{fltr_arr[z]}], non sauvegardé.", flush=True)
                plt.close(fig_panel)
            else:
                plt.close(fig_panel)
                print(f"Panel ignoré pour {star_name} [{fltr_arr[z]}] : filtre non spécifique.", flush=True)

            # Enregistrement de la figure de gauche (DoLP seule, avec contours et cercles) dans un dossier dédié
            outdir_dolp_only = '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Morphologies_contours/DoLP_only'
            os.makedirs(outdir_dolp_only, exist_ok=True)
            fig_dolp, ax_dolp = plt.subplots(figsize=(6, 5))
            im_dolp = ax_dolp.imshow(sub_v_dolp, cmap='plasma', origin='lower', vmin=np.nanmin(sub_v_dolp), vmax=np.nanmax(sub_v_dolp), extent=[x_min_DOLP, x_max_DOLP, y_min_DOLP, y_max_DOLP])
            dolp_min = np.nanmin(sub_v_dolp)
            dolp_max = np.nanmax(sub_v_dolp)
            contour_levels = np.linspace(dolp_min, dolp_max, 5)
            cs_dolp = ax_dolp.contour(sub_v_dolp, levels=contour_levels, colors='white', linewidths=1.5, origin='lower', extent=[x_min_DOLP, x_max_DOLP, y_min_DOLP, y_max_DOLP])
            ax_dolp.clabel(cs_dolp, inline=True, fontsize=10, fmt='%.2f')
            # Ajout des cercles comme dans le panel
            for r_mas, ls in zip(radii_mas, linestyles):
                circ_dolp = Circle((center_x, center_y), r_mas, edgecolor='orange', facecolor='none', lw=2, linestyle=ls)
                ax_dolp.add_patch(circ_dolp)
            # Barre d'échelle en R* (identique au panel)
            if distance_star_pc is not None and radius_star_rsun is not None and radius_star_rsun > 0 and distance_star_pc > 0:
                scale_val = 10  # 10 R*
                R_sun_cm = 6.957e10
                radius_star_cm = radius_star_rsun * R_sun_cm
                distance_star_cm = distance_star_pc * 3.0857e18
                radius_star_rad = radius_star_cm / distance_star_cm
                radius_stellar_mas = radius_star_rad * 206265 * 1000
                scale_mas = scale_val * radius_stellar_mas
                legend_unit = 'R$_\star$'
                bar_length_axes = scale_mas / (x_max_DOLP - x_min_DOLP)
                x_bar_axes = 0.90 - bar_length_axes
                y_bar_axes = 0.04
                ax_dolp.plot([x_bar_axes, x_bar_axes + bar_length_axes], [y_bar_axes, y_bar_axes], color='white', lw=3, transform=ax_dolp.transAxes, solid_capstyle='butt')
                ax_dolp.text(x_bar_axes + bar_length_axes/2, y_bar_axes + 0.01, f'{scale_val:.0f}{legend_unit}', color='white', fontsize=14, ha='center', va='bottom',  transform=ax_dolp.transAxes)
            # Flèches N-W en haut à droite (identique au panel)
            arrow_len = 0.04 * (x_max_DOLP - x_min_DOLP)
            x_arrow = x_max_DOLP - 0.04 * (x_max_DOLP - x_min_DOLP)# vers la gauche
            y_arrow = y_max_DOLP - 0.12 * (y_max_DOLP - y_min_DOLP)# vers le bas
            ax_dolp.arrow(x_arrow, y_arrow, 0, arrow_len, head_width=0.02*arrow_len, head_length=0.04*arrow_len, fc='white', ec='white', lw=2)
            ax_dolp.arrow(x_arrow, y_arrow, -arrow_len, 0, head_width=0.02*arrow_len, head_length=0.04*arrow_len, fc='white', ec='white', lw=2)
            offset_label = 8
            ax_dolp.text(x_arrow, y_arrow + arrow_len + offset_label, 'N', color='white', fontsize=label_size, ha='center', va='bottom')
            ax_dolp.text(x_arrow - arrow_len - offset_label, y_arrow, 'W', color='white', fontsize=label_size, ha='right', va='center')
            ax_dolp.set_xlabel('Relative RA (mas)', fontsize=label_size)
            ax_dolp.set_ylabel('Relative Dec (mas)', fontsize=label_size)
            ax_dolp.tick_params(axis='both', labelsize=label_size, width=1.2)
            ax_dolp.locator_params(axis='x', nbins=5)
            ax_dolp.locator_params(axis='y', nbins=5)
            ax_dolp.text(0.02, 0.95, f'{star_name2}', transform=ax_dolp.transAxes, fontsize=label_size, color='white', va='top')
            ax_dolp.text(0.02, 0.02, f'{fltr_arr[z]}', transform=ax_dolp.transAxes, fontsize=label_size, color='white', va='bottom')
            divider_dolp = make_axes_locatable(ax_dolp)
            cax_dolp = divider_dolp.append_axes('right', size='5%', pad=0.03)
            cb_dolp = fig_dolp.colorbar(im_dolp, cax=cax_dolp, orientation='vertical')
            cb_dolp.ax.tick_params(labelsize=label_size)
            cb_dolp.formatter.set_powerlimits((0, 0))
            cb_dolp.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
            cb_dolp.ax.yaxis.get_offset_text().set(size=label_size)
            plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10)
            fig_dolp_name = f'{star_name}_{obsmod}_{fltr_arr[z]}_{z}_DoLP_only'
            
            plt.savefig(os.path.join(outdir, fig_dolp_name + '.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(outdir, fig_dolp_name + '.pdf'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(outdir_dolp_only, fig_dolp_name + '.png'), dpi=300, bbox_inches='tight')
            # Sauvegarde conditionnelle dans le dossier spécifique DoLP (une seule image par filtre/étoile, et seulement si clairement résolue)
            
            dolp_specific_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Morphologies_contours/DoLP_specific'
            os.makedirs(dolp_specific_dir, exist_ok=True)
            # already_exists = any(str(fltr_arr[z]) in fname and star_name in fname for fname in os.listdir(dolp_specific_dir))
            # if not already_exists and star_name in clearly_resolved:
            #     fig_path = os.path.join(dolp_specific_dir, fig_dolp_name + '.png')
            #     plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            #     print(f"Figure DoLP-only sauvegardée : {fig_path}", flush=True)
            # else:
            #     print(f"DoLP déjà présente pour {star_name} [{fltr_arr[z]}], non sauvegardée.", flush=True)
            # plt.savefig(os.path.join(outdir_dolp_only, fig_dolp_name + '.pdf'), dpi=300, bbox_inches='tight')
            
            if fltr_arr[z] == star_specific_filter.get(star_name, None):
                # On ne sauvegarde que si aucune image pour ce filtre n'existe déjà
                already_exists = any(str(fltr_arr[z]) in fname and star_name in fname for fname in os.listdir(dolp_specific_dir))
                if not already_exists and star_name in clearly_resolved:
                    #fig_panel_name_unique = f"{star_name}_{mode}_{fltr_arr[z]}_PI_DoLP_panel"
                    fig_path = os.path.join(dolp_specific_dir, fig_dolp_name + '.pdf')
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    print(f"Figure panel PI+DoLP sauvegardée : {fig_path}", flush=True)
                else:
                    print(f"Panel déjà présent pour {star_name} [{fltr_arr[z]}], non sauvegardé.", flush=True)
                plt.close(fig_panel)
            else:
                plt.close(fig_panel)
                print(f"Panel ignoré pour {star_name} [{fltr_arr[z]}] : filtre non spécifique.", flush=True)

            plt.close(fig_dolp)
        else:
            plt.close(fig_panel)
            print(f"Panel ignoré pour {star_name} [{fltr_arr[z]}] : filtre non spécifique.", flush=True)
            #plt.close(fig_panel)
            # print(f"Figure panel PI+DoLP sauvegardée : {os.path.join(outdir_panels, fig_panel_name + '.png')}", flush=True)

            # Calcul du profil radial moyen normalisé
            profile = radial_profile(sub_v)
            profile_norm = profile / np.max(profile)
            r_pix = np.arange(len(profile_norm))
            r_mas = r_pix * pix2mas

            # Plot du profil radial moyen
            plt.figure(figsize=(6, 5))
            plt.plot(r_mas, profile_norm, color='#1f77b4', lw=2, label='Radial profile')
            plt.axhline(best_threshold, color='#d62728', ls='--', lw=2, label=f'$h$ (contrast) = {best_threshold:.3f}')
            plt.xlabel('θ (mas)', fontsize=label_size)
            plt.ylabel('PIL/PIL_max', fontsize=label_size)
            #plt.title(f'{star_name2} - {fltr_arr[z]}', fontsize=label_size)
            plt.tick_params(axis='both', labelsize=label_size, width=1.2)
            plt.legend(fontsize=label_size, loc='upper right')
            #plt.grid(alpha=0.3)
            plt.xlim(0, diameter_mas/2)  # Limite l'axe des x au rayon majeur de l'enveloppe
            plt.tight_layout()
            fig_name = f'{star_name}_profile_radial_PI_{fltr_arr[z]}_{z}'
            plt.savefig(os.path.join(outdir, fig_name + '.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(outdir, fig_name + '.pdf'), dpi=300, bbox_inches='tight')
            #plt.savefig(os.path.join(outdir, fig_name + '.eps'), format='eps', dpi=300, bbox_inches='tight')
            #plt.show()
            plt.close()


    # Sauvegarde du DataFrame individuel
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(outdir, f'morpho_results_{star_name}_{obsmod}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Résultats morphologiques sauvegardés dans : {csv_path}")
        
        # 📝 Génération de la version LaTeX individuelle
        latex_path = os.path.join(outdir, f'morpho_results_{star_name}_{obsmod}.tex')
        generate_morphology_latex_table(df, latex_path)
    
    # Retourne les résultats pour compilation globale
    elapsed_time = time.time() - start_time
    print(f"⏱️ Durée d'exécution log_image : {elapsed_time:.2f} secondes")
    return results

# Exemple d'appel de la fonction
# log_image('First','V854_Cen', 'alone')
# log_image('First','V854_Cen', 'both')



# Liste globale pour compiler tous les résultats
all_results = []

# Vérification de l'existence du répertoire principal
if not os.path.exists(large_log_dir):
    print(f"❌ ERREUR : Le répertoire {large_log_dir} n'existe pas !")
    print("   Vérifiez le chemin dans 'large_log_dir'")
else:
    print(f"✅ Répertoire principal trouvé : {large_log_dir}")
    # Liste des dossiers trouvés
    found_dirs = [d for d in os.listdir(large_log_dir) if os.path.isdir(os.path.join(large_log_dir, d))]
    print(f"📁 Dossiers trouvés : {found_dirs}")
    print(f"🔍 Étoiles dans star_filters : {list(star_filters.keys())}")

# L'appel à la fonction principale pour chaque étoile dans le répertoire
for star_name in os.listdir(large_log_dir):
    star_path = os.path.join(large_log_dir, star_name)
    if not os.path.isdir(star_path):
        continue
    if star_name in star_filters:
        mode, filters = star_filters[star_name]
        print(f"🌟 Traitement de {star_name} | Mode : {mode} | Filtres : {filters}")
        star_results = log_image('large_log_+', star_name, mode, star_specific_filter=star_specific_filter)
        if star_results:
            print(f"   ✅ {len(star_results)} résultats obtenus pour {star_name}")
            all_results.extend(star_results)  # Ajoute les résultats de cette étoile à la liste globale
        else:
            print(f"   ⚠️ Aucun résultat pour {star_name}")
    else:
        print(f"❌ {star_name} n'est pas dans le dictionnaire star_filters, ignoré.")

print(f"\n📊 RÉSUMÉ FINAL :")
print(f"📈 Total de résultats collectés : {len(all_results)}")

# Sauvegarde de la table globale
if all_results:
    global_df = pd.DataFrame(all_results)
    
    # Création du dossier All_tables s'il n'existe pas
    all_tables_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_tables/morpho'
    os.makedirs(all_tables_dir, exist_ok=True)
    print(f"📁 Répertoire All_tables créé/vérifié : {all_tables_dir}")

    # TEST: Vérification des permissions d'écriture
    test_file = os.path.join(all_tables_dir, 'test_write.txt')
    try:
        with open(test_file, 'w') as f:
            f.write("Test d'écriture")
        print(f"✅ Permissions d'écriture OK dans {all_tables_dir}")
        os.remove(test_file)  # Supprime le fichier de test
    except Exception as e:
        print(f"❌ ERREUR de permissions d'écriture : {e}")

    global_latex_path = os.path.join(all_tables_dir, 'morpho_results_all_stars_global.tex')
    global_csv_path = os.path.splitext(global_latex_path)[0] + '.csv'
    
    # Force l'écriture du CSV avec gestion d'erreur explicite
    try:
        global_df.to_csv(global_csv_path, index=False)
        print(f"✅ CSV écrit sans erreur")
    except Exception as e:
        print(f"❌ ERREUR lors de l'écriture CSV : {e}")
    
    # Vérification immédiate de la création du fichier CSV
    if os.path.exists(global_csv_path):
        file_size = os.path.getsize(global_csv_path)
        print(f"✅ Fichier CSV créé avec succès : {file_size} bytes")
        # Lecture du début du fichier pour vérifier le contenu
        try:
            with open(global_csv_path, 'r') as f:
                first_lines = f.read(200)  # Lit les 200 premiers caractères
            print(f"📄 Contenu CSV (début) : {first_lines}")
        except Exception as e:
            print(f"⚠️ Impossible de lire le CSV : {e}")
    else:
        print(f"❌ ERREUR : Fichier CSV non trouvé après écriture !")
    
    # 📝 Génération de la version LaTeX globale
    global_latex_path = os.path.join(all_tables_dir, 'morpho_results_all_stars_global.tex')
    
    # Force l'écriture du LaTeX avec gestion d'erreur explicite
    try:
        generate_morphology_latex_table(global_df, global_latex_path)
        print(f"✅ LaTeX écrit sans erreur")
    except Exception as e:
        print(f"❌ ERREUR lors de l'écriture LaTeX : {e}")
    
    # Vérification immédiate de la création du fichier LaTeX
    if os.path.exists(global_latex_path):
        file_size = os.path.getsize(global_latex_path)
        print(f"✅ Fichier LaTeX créé avec succès : {file_size} bytes")
        # Lecture du début du fichier pour vérifier le contenu
        try:
            with open(global_latex_path, 'r') as f:
                first_lines = f.read(200)  # Lit les 200 premiers caractères
            print(f"📄 Contenu LaTeX (début) : {first_lines}")
        except Exception as e:
            print(f"⚠️ Impossible de lire le LaTeX : {e}")
    else:
        print(f"❌ ERREUR : Fichier LaTeX non trouvé après écriture !")
    
    # Force le flush du système de fichiers
    import sys
    sys.stdout.flush()
    
    print(f"\n🎉 TABLE GLOBALE CRÉÉE !")
    print(f"📊 Nombre total de mesures : {len(all_results)}")
    print(f"🌟 Nombre d'étoiles traitées : {global_df['Étoile'].nunique()}")
    print(f"🔬 Filtres utilisés : {global_df['Filtre'].nunique()}")
    print(f"📄 Fichier CSV global sauvegardé : {global_csv_path}")
    print(f"📄 Fichier LaTeX global sauvegardé : {global_latex_path}")
    
    # Listing final du répertoire
    try:
        files_in_dir = os.listdir(all_tables_dir)
        print(f"📁 Fichiers dans {all_tables_dir} : {files_in_dir}")
    except Exception as e:
        print(f"❌ Impossible de lister le répertoire : {e}")
        
else:
    print(f"\n⚠️ Aucun résultat à compiler dans la table globale.")
    print(f"   Raison : all_results contient {len(all_results)} éléments")
    print(f"   Vérifiez que les étoiles sont bien traitées et retournent des résultats")

# Après avoir rempli la liste 'results' avec tous les filtres
results_df = pd.DataFrame(results)

# Sélectionner, pour chaque étoile, la ligne où le grand diamètre est maximal
idx_max_diam = results_df.groupby('Étoile')['D_maj_mas'].idxmax()
filtered_results_df = results_df.loc[idx_max_diam].reset_index(drop=True)

# Sauvegarder la nouvelle table (par exemple en CSV)
filtered_results_df.to_csv('filtered_results_max_diam.csv', index=False)
# --- Ajout : extraction des distances depuis une table CSV ---
def read_csv_distances(csv_path):
    """
    Lit un fichier CSV contenant les distances des étoiles.
    Suppose que la première colonne est le nom de l'objet et la colonne 'Distance' existe.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    # Renomme la colonne du nom de l'objet pour correspondre à la jointure
    if 'Object' in df.columns:
        df = df.rename(columns={'Object': 'Étoile'})
    return df

# Adapter le chemin du fichier CSV
csv_dist_path = '/home/nbadolo/Bureau/Aymard/Tables/ML/Hipparcos/Hip/Sample_hip.csv'
dist_table = read_csv_distances(csv_dist_path)
# Harmonisation automatique des noms d'étoiles pour la jointure
def clean_star_name(name):
    # Remplace les underscores par des espaces, retire les points, met en minuscules et retire les espaces multiples
    name = str(name)
    name = name.replace('_', ' ')
    name = name.replace('.', '')
    name = name.lower()
    name = re.sub(r'\s+', ' ', name).strip()
    # Applique le mapping manuel si le nom correspond
    return star_name_map.get(name, name)

# Mapping manuel pour les cas ambigus de noms d'étoiles
star_name_map = {
    'l02 pup': 'l2 pup',
    'bet gru': 'beta gru',
    'pi01 gru': 'pi01 gru',
    'l2 pup': 'l2 pup',
    'gy aql': 'gy aql',
    '17 lep': '17 lep',
    # Ajoute ici d'autres corrections spécifiques si besoin
}

# Appliquer le nettoyage sur les deux colonnes
filtered_results_df['star_clean'] = filtered_results_df['Étoile'].apply(clean_star_name)
dist_table['star_clean'] = dist_table['Étoile'].apply(clean_star_name)

# Diagnostic : affichage des noms nettoyés
print("Noms nettoyés morpho:", filtered_results_df['star_clean'].unique())
print("Noms nettoyés distances:", dist_table['star_clean'].unique())

# Jointure sur la colonne nettoyée
filtered_results_df = filtered_results_df.merge(dist_table, on='star_clean', how='left', suffixes=('', '_dist'))

# Calcul de la taille physique du grand axe en UA (utilise la colonne 'Distance' issue de dist_table)
filtered_results_df['D_maj_UA'] = filtered_results_df['D_maj_mas'] * filtered_results_df['Distance'] * 0.001
# --- Fin ajout ---
# Génération et sauvegarde d'un histogramme des ellipticités basé sur la table filtrée

plt.figure(figsize=(7, 5))
hist_vals, bins, patches = plt.hist(filtered_results_df['e'], bins=20, color='#4A90E2', edgecolor='#bbbbbb', linewidth=1.1, alpha=0.85, rwidth=0.92)
# Bordures fines et grises claires
for patch in patches:
    patch.set_linewidth(1.1)
    patch.set_edgecolor('#bbbbbb')
# Ajout d'une ligne moyenne
# Ajout d'une ligne médiane
median_e = filtered_results_df['e'].median()
plt.axvline(median_e, color='#D0021B', linestyle ='--', linewidth=2, label=f'Median = {median_e:.2f}')
# Ajout d'annotations sur chaque barre
for val, patch in zip(hist_vals, patches):
    if val > 0:
        plt.text(patch.get_x() + patch.get_width()/2, val + 0.05, f'{int(val)}', ha='center', va='bottom', fontsize=12, color='#333333')
plt.xlabel('$e$', fontsize=14)
plt.ylabel('N. stars', fontsize=14)
#plt.title('Histogram of ellipticities\n(max diameter per star)', fontsize=17, fontweight='bold', color='#4A90E2', pad=18)
#plt.grid(alpha=0.3, linestyle='--')
plt.legend(fontsize=14, loc='upper right')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(all_tables_dir + 'histogramme_ellipticite_max_diam.png', dpi=300, bbox_inches='tight')
plt.savefig(all_tables_dir + 'histogramme_ellipticite_max_diam.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Histogramme des ellipticités stylé sauvegardé sous 'histogramme_ellipticite_max_diam.png' et '.pdf'")

# Histogramme des tailles physiques (D_maj_UA)
plt.figure(figsize=(7, 5))
hist_vals_phys, bins_phys, patches_phys = plt.hist(filtered_results_df['D_maj_UA'].dropna(), bins=20, color='#50B878', edgecolor='#bbbbbb', linewidth=1.1, alpha=0.85, rwidth=0.92)
for patch in patches_phys:
    patch.set_linewidth(1.1)
    patch.set_edgecolor('#bbbbbb')
median_phys = filtered_results_df['D_maj_UA'].median()
plt.axvline(median_phys, color='#D0021B', linestyle='--', linewidth=2, label=f'Median = {median_phys:.1f} AU')
for val, patch in zip(hist_vals_phys, patches_phys):
    if val > 0:
        plt.text(patch.get_x() + patch.get_width()/2, val + 0.05, f'{int(val)}', ha='center', va='bottom', fontsize=12, color='#333333')
plt.xlabel('$D_{maj}$ (AU)', fontsize=14)
plt.ylabel('N. stars', fontsize=14)
#plt.title('Histogram of physical sizes\n(max diameter per star)', fontsize=16, fontweight='bold', color='#50B878', pad=16)
plt.legend(fontsize=14, loc='upper right')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(all_tables_dir + 'histogramme_taille_physique_max_diam.png', dpi=300, bbox_inches='tight')
plt.savefig(all_tables_dir + 'histogramme_taille_physique_max_diam.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Histogramme des tailles physiques sauvegardé sous 'histogramme_taille_physique_max_diam.png' et '.pdf'")

# Sauvegarde de la table filtrée en CSV
if 'affected_stars_dolp' in globals() and len(affected_stars_dolp) > 0:
    # Liste unique des étoiles concernées
    etoiles_uniques = sorted(set([star['Étoile'] for star in affected_stars_dolp]))
    print('\nListe unique des Étoiles avec cercles fixes DoLP (distance < 150 pc ou rayon > 500 R☉):')
    for etoile in etoiles_uniques:
        print(f"  - {etoile}")
    # Sauvegarde dans le dossier spécifique All_tables/morpho
    all_tables_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_tables/morpho'
    os.makedirs(all_tables_dir, exist_ok=True)
    with open(os.path.join(all_tables_dir, 'affected_stars_dolp_unique.txt'), 'w') as f:
        for etoile in etoiles_uniques:
            f.write(f"{etoile}\n")
# Colonnes à conserver
cols_to_keep = [
    'Étoile', 'Filtre', 'D_maj_mas', 'sigma_D_maj_mas', 'D_min_mas', 'sigma_D_min_mas',
    'Distance', 'D_maj_UA', 'b_a', 'e', 'X0_mas', 'Y0_mas', 'theta_deg', 'Cout'
]
filtered_results_df_export = filtered_results_df[cols_to_keep]
filtered_results_df_export.to_csv(os.path.join(all_tables_dir, 'filtered_results_max_diam.csv'), index=False)
filtered_latex_path = os.path.join(all_tables_dir, 'filtered_results_max_diam.tex')
# Conversion du CSV en LaTeX

df_csv = pd.read_csv(os.path.join(all_tables_dir, 'filtered_results_max_diam.csv'))
generate_morphology_latex_table(df_csv, filtered_latex_path)
print(f"✅ Table CSV et LaTeX filtrée sauvegardée sous '{os.path.join(all_tables_dir, 'filtered_results_max_diam.csv')}' et '{filtered_latex_path}'")


# === ADAPTATION EXPORT LATEX ===
# Adapter la fonction generate_morphology_latex_table pour inclure les colonnes D_maj_UA, sigma_D_maj_UA, D_maj_UA_dolp, sigma_D_maj_UA_dolp

def generate_morphology_latex_table(df, latex_path):
    """
    Génère un fichier LaTeX contenant le tableau des résultats morphologiques avec entête personnalisée et arrondi à 3 chiffres après la virgule pour les nombres si besoin.
    :param df: DataFrame contenant les résultats morphologiques
    :param latex_path: Chemin de sauvegarde du fichier LaTeX  
    """
    import re
    def escape_latex_data(s):
        s = str(s)
        s = s.replace('\\', r'\\')
        s = s.replace('_', r'\_')
        s = s.replace('&', r'\&')
        s = s.replace('%', r'\%')
        s = s.replace('#', r'\#')
        s = s.replace('~', r'\textasciitilde{}')
        s = s.replace('^', r'\^{}')
        return s
    def format_latex_number(val):
        sval = str(val)
        if re.match(r'^-?\d+\.\d+$', sval):
            decimals = sval.split('.')[-1]
            if len(decimals) > 3:
                return f"{float(val):.3f}"
            else:
                return sval
        else:
            return sval
    # Entête personnalisée
    custom_header = (
        "Star & Filter & "
        "\\begin{tabular}{c}$a$ \\ (mas) \end{tabular} & "
        "\\begin{tabular}{c} $\\sigma_a$\\ (mas)\\end{tabular} & "
        "\\begin{tabular}{c} b\\ (mas) \end{tabular} & "
        "\\begin{tabular}{c} $\\sigma_b$ \\ (mas)\\end{tabular} & "
        "\\begin{tabular}{c} $D$ \\ (pc) \end{tabular} & "
        "\\begin{tabular}{c} $a$ \\ (UA) \end{tabular} & "
        "$b/a$ & $e$ & "
        "\\begin{tabular}{c} $X_0$ \\ (mas) \end{tabular} & "
        "\\begin{tabular}{c} $Y_0$\\ (mas)\\end{tabular} & "
        "\\begin{tabular}{c} $\\theta$\\ (deg) \end{tabular} & $\\chi_{red}^2$ \\" 
    )
    ncols = len(df.columns)
    latex_table = ""
    latex_table += f"\\begin{{tabular}}{{{'c'*ncols}}}\n"
    latex_table += "\\toprule\n"
    latex_table += custom_header + "\n"
    latex_table += "\\midrule\n"
    for _, row in df.iterrows():
        formatted_row = [escape_latex_data(format_latex_number(val)) for val in row]
        line = " & ".join(formatted_row) + " \\\\"
        latex_table += line + "\n"
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    with open(latex_path, 'w') as f:
        f.write(latex_table)

# === ASSEMBLAGE DES GRANDS PANELS PAGINÉS ===
def assemble_grand_panels_paginated(panel_dir, output_path_base, ncols=2, nrows=7):
    """
    Crée plusieurs grands panels paginés, chaque panel contenant au maximum ncols*nrows images.
    - panel_dir: dossier où sont stockés les panels individuels (DoLP+contours & PI)
    - output_path_base: base du chemin de sortie (sans extension ni numéro de page)
    - ncols: nombre de colonnes
    - nrows: nombre de lignes
    """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import os
    image_files = [f for f in os.listdir(panel_dir) if f.endswith('PI_DoLP_panel.png')]
    n_panels_per_page = ncols * nrows
    total = len(image_files)
    page = 0
    for start in range(0, total, n_panels_per_page):
        page += 1
        end = min(start + n_panels_per_page, total)
        fig, axes = plt.subplots(nrows, ncols, figsize=(12*ncols, 5*nrows))
        if nrows == 1:
            axes = np.array([axes])
        axes = axes.reshape(nrows, ncols)
        for idx, fname in enumerate(image_files[start:end]):
            row = idx // ncols
            col = idx % ncols
            img_path = os.path.join(panel_dir, fname)
            img = mpimg.imread(img_path)
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            #axes[row, col].set_title(fname.replace('_PI_DoLP_panel.png',''), fontsize=18)
        # Désactive les axes vides
        for idx in range(end-start, n_panels_per_page):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].axis('off')
        panel_name = 'PI_panel'
        # Ajuste les espacements pour un remplissage optimal
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.10, hspace=0.01)
        plt.savefig(f"{output_path_base}png/{panel_name}_page_{page}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_path_base}pdf/{panel_name}_page_{page}.pdf", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Grand panel page {page} sauvegardé sous {output_path_base}png/{panel_name}_page_{page}.png et {output_path_base}pdf/{panel_name}_page_{page}.pdf")
        # Appel de la fonction pour assembler les panels paginés
panels_directory = '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Morphologies_contours/Panels/specific/'
output_base = '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Morphologies_contours/Panels/grand_panel/'
os.makedirs(output_base + 'png', exist_ok=True)
os.makedirs(output_base + 'pdf', exist_ok=True)
#assemble_grand_panels_paginated(panels_directory, output_base, ncols=2, nrows=7)