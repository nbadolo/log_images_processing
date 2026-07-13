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
from skimage.measure import EllipseModel
from matplotlib.path import Path
from matplotlib.colors import to_hex
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib.patheffects as PathEffects
from AymardPack import process_fits_image as pfi  # Pour le traitement des pixels chauds/froids
import shutil

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
    #'W_Hya':    ('alone', ['V']),
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
    #'RT_Vir':   ('alone', ['V']),
    'RX_Lep':   ('alone', ['CntHa']),
    'Beta_Gru': ('both', ['V_N_R']), 
    'T_Mic':    ('both', ['V_N_R']),
    'R_Dor':    ('both',  ['Cnt820_Cnt748']),
    'AK_Hya':   ('alone', ['N_R']),
    'R_Leo':    ('both', ['V_Cnt748']),
    #'BK_Vir':   ('alone', ['Cnt820']),
    'BK_Vir':   ('alone', ['V']),
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
    'Pi.01_Gru':('both',  ['Cnt820_Cnt748']),
    'V854_Cen': ('alone', ['V']),
   
}

star_specific_filter = {
    'Y_Pav':    'N_R',
    'R_Hor':    'N_R',
    'R_Scl':    'N_R',
    'Y_Scl':    'V',
    'R_Crt':    'N_R',
    'SW_Col':   'N_R',
    'W_Hya':    'Cnt748',
    #'W_Hya':    'V',
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
    #'RT_Vir':   'Cnt820',
    'RT_Vir':   'V',
    'RX_Lep':   'CntHa',
    'Beta_Gru': 'V',
    'T_Mic':    'N_R',
    'R_Dor':    'Cnt748',
    'AK_Hya':   'N_R',
    'R_Leo':    'V',
    #'BK_Vir':   'Cnt820',
    'BK_Vir':   'V',
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
    'L02_Pup':  'N_R',
    'CW_Cnc':   'I_PRIM',
    'Mira':     'Cnt820',
    'Pi.01_Gru': 'Cnt820',
    'V854_Cen': 'V',
}

clearly_resolved = [
    "AK_Hya", "R_Hya", "U_Her", "S_Pav", "Mira", "W_Aql", "R_Crt", "Pi.01_Gru",
    "R_Dor", "BK_Vir", "V_PsA", "SW_Col", "GY_Aql", "SW_Vir", "RT_Vir",
    "W_Hya", "L02_Pup", "W_Peg"
]

marginally_resolved = [
    "V_Hya", "BW_Oct", "R_Leo", "V1943_Sgr", "Y_Pav", "Chi_Cyg",
    "R_Scl", "U_Del", "T_Mic", "Z_Peg", "Y_Scl"
]

all_resolved = clearly_resolved + marginally_resolved

## les structures particulières
bipolar = [
    "AK_Hya", "R_Hya", "U_Her", "S_Pav", "W_Aql", "R_Crt", 
    "V_PsA", "SW_Col",  "SW_Vir", "W_Hya", "L02_Pup", "W_Peg", "R_Dor"
]

spiral_arc = [
    "Mira",  "Pi.01_Gru", "GY_Aql"
]

# spherical = [
#     "R_Dor", "BK_Vir", "RT_Vir",
    
# ]



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

def add_text_with_outline(ax, x, y, text, **kwargs):
    """
    Ajoute du texte sur une image avec un contour noir pour meilleure lisibilité.
    """
    txt = ax.text(x, y, text, **kwargs)
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
    return txt

fold_name ='large_log_+'  # Choix du dossier d'analyse : 'clearly_resolved', 'marginally_resolved', 'all_resolved', 'bipolar', 'spiral_arc', 'spherical'
#large_log_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log_+/'
#large_log_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/newly_resolved/'
#large_log_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/clearly_resolved/'
#large_log_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/ICHTUS/'
#large_log_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/BK_Vir/'
#large_log_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/Already_observed/'
large_log_dir = f'/home/nbadolo/Bureau/Aymard/Donnees_sph/{fold_name}/'
structure= None   #choix de la structure à analyser. Si  aucune strurcture particulière, mettre "structure=None"




def log_image(folder_name, star_name, obsmod, star_specific_filter=None):
    global os
    global fits
    global Cutout2D
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
    'W_Aql': 400,
    'Mira': 260,
    'RT_Vir': 240, 
    'W_Peg': 280,
    'AK_Hya': 150,
    'BK_Vir': 160,
    'U_Her': 160,
    'W_Peg': 200,
    'W_Hya': 200,
    'R_Crt': 240,
    'SW_Col': 200,
    'S_Pav': 200,
    'R_Hya': 200,
    'GY_Aql': 260,
    'SW_Vir': 240,
    'Pi.01_Gru': 140,
    'L02_Pup': 240,
    'R_Dor': 200,
    'V854_Cen': 80,
    'SS_Lep': 160,
    'SV_Aqr': 160,
    'R_Peg': 160,
    'Chi_Cyg': 100,
    'U_Del': 160,
    'R_Leo': 160,
    'T_Mic': 160,
    'Ups_Cet': 120,
    'RX_Lep': 160,
    'DZ_Aqr': 160,
    'AC_Cet': 160,
    'Y_Scl': 160,
    'V1943': 160,
    'Alpha_Her': 100,
    'RT_Vir': 150,
    }

    # custom_dolp_contours = {
    #             'AK_Hya': [0.018,  0.06],
    #             'BK_Vir': [0.018, 0.1],
    #             'RT_Vir': [0.02, 0.06, 0.1],
    #             'SW_Col': [0.013, 0.03, 0.05, 0.1],
    #             'W_Peg': [0.025, 0.1],
    #             'U_Her': [0.03, 0.05, 0.2, 0.3],
            
    #         }
    custom_dolp_contours = {
                'AK_Hya': [0.0135,  0.02],
                'BK_Vir': [0.012],
                'RT_Vir': [0.012,  0.04],
                'SW_Col': [0.015, 0.02, 0.03, 0.05],
                'W_Peg': [0.017,0.03,0.05],
                'U_Her': [0.008, 0.019, 0.0386],
                'R_Crt': [0.017,  0.04],
                'V_PsA': [0.01, 0.02, 0.04],
                'S_Pav': [0.0095, 0.02, 0.03, 0.04],
                'R_Hya': [0.016, 0.03],
                'GY_Aql': [ 0.04, 0.11],
                'SW_Vir': [0.013, 0.03, 0.05, 0.07],
                'Pi.01_Gru': [ 0.045, 0.06, 0.1],
                'L02_Pup': [ 0.04, 0.1, 0.2, 0.3],
                'R_Dor': [0.02, 0.03, 0.05, 0.07],
                'Mira': [0.014, 0.04],
                'W_Hya': [ 0.035, 0.049,  0.1, 0.17],
                'V854_Cen': [0.014, 0.035, 0.04],
            
    
            }
    # Dictionnaire pour personnaliser la taille de la barre d'échelle (en R*) pour chaque étoile
    custom_scale_bar = {
            'R_Dor': 5,
            'W_Hya': 2,
            'W_Aql': 12,
            'R_Hya': 2,
            'AK_Hya': 10,
            'R_Crt': 10,
            'SW_Col': 40,
            'V854_Cen': 100,
            'Pi.01_Gru': 6,
            'L02_Pup': 10,
            'SW_Vir': 5,
            'S_Pav': 5,
            'U_Her': 2,
            'W_Peg': 5,
            'GY_Aql': 2,
            'Mira': 6,
            'R_Dor': 2,
            'RT_Vir': 2,
            'BK_Vir': 5,
            'Alpha_Her': 4,
            'Chi_Cyg': 4,
            '17_Lep': 30,
            'R_Leo': 5,
            'Ac_Cet': 30,
            'RT_Vir': 6,
    
    }
    
    # ============================================================================
    # APPROCHE AUTOMATIQUE POUR VMAX (ACTIVABLE/DÉSACTIVABLE)
    # ============================================================================
    USE_AUTO_VMAX = False  # Mettre False pour utiliser les dictionnaires manuels

    # Toggle: plot a single white circle at exactly 3 R* on the DoLP map
    # Applies to `SW_Col` and stars in `clearly_resolved`. Set to False to disable.
    PLOT_3RSTAR_ON_DOLP = False
    
    # --- Dictionnaires manuels (DÉSACTIVÉS si USE_AUTO_VMAX=True) ---
    # Dictionnaire pour personnaliser les limites vmin/vmax de la colorbar DoLP
    # Utile quand des pixels hors zone d'intérêt faussent l'échelle
    custom_dolp_vrange = {
        'R_Hya': {'vmin': None, 'vmax': 0.05},     # Limite le max à 0.04
        'AK_Hya': {'vmin': None, 'vmax': 0.035},   # Limite le max à 0.04
        'BK_Vir': {'vmin': None, 'vmax': 0.03},   # Limite le max à 0.025
        'RT_Vir': {'vmin': None, 'vmax': 0.055},    # Limite le max à 0.04
        'W_Hya': {'vmin': None, 'vmax': 0.16},     # Limite le max à 0.04
        #'Pi.01_Gru': {'vmin': None, 'vmax': 0.04}, # Limite le max à 0.04
        'GY_Aql': {'vmin': None, 'vmax': 0.13},    # Limite le max à 0.12
        'Mira': {'vmin': None, 'vmax': 0.05},      # Limite le max à 0.05
        'S_Pav': {'vmin': None, 'vmax': 0.05},     # Limite le max à 0.05
        'Y_Pav': {'vmin': None, 'vmax': 0.06},    # Limite le max à 0.04
         'U_Del': {'vmin': None, 'vmax': 0.15},     # Limite le max à 0.04
        'Chi_Cyg': {'vmin': None, 'vmax': 0.15},   # Limite le max à 0.04
        'SS_Lep': {'vmin': None, 'vmax': 0.02},    # Limite le max à 0.04
        'SV_Aqr': {'vmin': None, 'vmax': 0.15},    # Limite le max à 0.04
        # 'Z_Eri': {'vmin': None, 'vmax': 0.04},     # Limite le max à 0.04
         'RX_Lep': {'vmin': None, 'vmax': 0.03},    # Limite le max à 0.04
         'Psi_Phe': {'vmin': None, 'vmax': 0.03},   # Limite le max à 0.04
        # 'Ups_Cet': {'vmin': None, 'vmax': 0.04},   # Limite le max à 0.04
        'DZ_Aqr': {'vmin': None, 'vmax': 0.06},    # Limite le max à 0.04
        # 'V1943_Sgr': {'vmin': None, 'vmax': 0.04}, # Limite le max à 0.04
         'BW_Oct': {'vmin': None, 'vmax': 0.04},    # Limite le max à 0.04
        # 'R_Crt': {'vmin': None, 'vmax': 0.04},     # Limite le max à 0.04
        # 'U_Her': {'vmin': None, 'vmax': 0.04},     # Limite le max à 0.04
         'R_Hor': {'vmin': None, 'vmax': 0.25},     # Limite le max à 0.04
         
         'Bet_Gru': {'vmin': None, 'vmax': 0.04},   # Limite le max à 0.04
        # 'CW_Cnc': {'vmin': None, 'vmax': 0.04},    # Limite le max à 0.04
         'R_Leo': {'vmin': None, 'vmax': 0.10},    # Limite le max à 0.04
         'Alpha_Her': {'vmin': None, 'vmax': 0.10},    # Limite le max à 0.04
         'Z_Peg': {'vmin': None, 'vmax': 0.15},    # Limite le max à 0.04
         'R_Peg': {'vmin': None, 'vmax': 0.20},    # Limite le max à 0.04
    }
    
    # Dictionnaire pour personnaliser la correction gamma des images DoLP par étoile
    # gamma < 1 assombrit, gamma > 1 éclaircit (valeur par défaut = 1.0, pas de correction)
    gam=1.4
    custom_dolp_gamma = {
        'R_Hya': gam,       # Assombrir modérément
        'AK_Hya': gam,      # Assombrir modérément
        'BK_Vir': gam,      # Assombrir modérément
        'GY_Aql': gam,      # Assombrir modérément
        'Mira': gam,        # Assombrir modérément
        'S_Pav': gam,       # Assombrir modérément
        'U_Del': gam,       # Assombrir modérément
        'Chi_Cyg': gam,     # Assombrir modérément
        'SS_Lep': gam,      # Assombrir modérément
        'SV_Aqr': gam,      # Assombrir modérément
        'Z_Eri': gam,       # Assombrir modérément
        'RX_Lep': gam,      # Assombrir modérément
        'Psi_Phe': gam,     # Assombrir modérément
        'Ups_Cet': gam,     # Assombrir modérément
        'DZ_Aqr': gam,      # Assombrir modérément
        'V1943_Sgr': gam,   # Assombrir modérément
        'BW_Oct': gam,      # Assombrir modérément
        'R_Crt': gam,       # Assombrir modérément
        'U_Her': gam,       # Assombrir modérément
        'R_Hor': gam,       # Assombrir modérément
        'Pi.01_Gru': gam,   # Assombrir modérément
        'Bet_Gru': gam,     # Assombrir modérément
        'CW_Cnc': gam,      # Assombrir modérément
        'R_Leo': gam,       # Assombrir modérément
    }
    nSubDim_DOLP = custom_dolp_cutout_size.get(star_name, 200)
    size_DOLP = (nSubDim_DOLP, nSubDim_DOLP)
    #lst_threshold = [0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]
    #lst_threshold = np.linspace(0.005, 0.1, 50)  # 50 seuils de 0.5% à 10% 
    lst_threshold = [0.03, 0.04]  # 100 seuils de 0% à 10%
    pix2mas = 3.4
    position = (nDim // 2, nDim // 2)
    
    label_size_small_panel = 18
    label_size_great_panel = 26
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
    outdir_dolp_only = f'/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Morphologies_contours/DoLP_only'
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
        file_I_star = os.path.join(fdir_star_fltr, fname1 + '_I' + fname2 + '_I.fits')
        if not os.path.exists(file_PI_star):
            print(f"Fichier manquant : {file_PI_star}")
        if not os.path.exists(file_DOLP_star):
            print(f"Fichier DoLP manquant : {file_DOLP_star}")
            continue

        hdu = fits.open(file_PI_star)
        data = hdu[0].data
        header = hdu[0].header
         # Détection automatique d'inversion de l'axe RA (CDELT1 < 0)
        invert_x = False
        try:
            wcs_tmp = WCS(header)
            cdelt = np.asarray(wcs_tmp.wcs.cdelt)
            if cdelt.size > 0 and cdelt[0] < 0:
                invert_x = True
        except Exception:
            if header.get('CDELT1', 1.0) < 0:
                invert_x = True
        if invert_x:
            print(f"DEBUG: RA axis inverted for {star_name} (CDELT1<0). Will flip X axis for display.", flush=True)

        # Détection robuste de l'inversion de l'axe RA : on échantillonne la transformation pixel->world
        invert_x = False
        try:
            wcs_tmp = WCS(header)
            # centre d'image (défini plus haut)
            cx, cy = position
            # requête pixel->world en origin=0 (indexation numpy)
            ra0, dec0 = wcs_tmp.wcs_pix2world([[cx, cy]], 0)[0]
            ra1, dec1 = wcs_tmp.wcs_pix2world([[cx + 1, cy]], 0)[0]
            # gérer le wrap RA en degrés : ramener la différence dans [-180, +180]
            delta = ((ra1 - ra0 + 180.0) % 360.0) - 180.0
            invert_x = (delta < 0)
            if np.isnan(delta):
                # fallback si pixel->world invalide
                raise ValueError("WCS pixel->world returned NaN")
        except Exception:
            # fallback : essayer la matrice CD si présente, sinon CDELT1
            try:
                cd = getattr(wcs_tmp.wcs, 'cd', None)
                if cd is not None:
                    # signe du terme cd[0,0] donne le sens local de l'axe x->RA
                    invert_x = (cd[0, 0] < 0)
                else:
                    invert_x = (header.get('CDELT1', 1.0) < 0)
            except Exception:
                invert_x = (header.get('CDELT1', 1.0) < 0)
        if invert_x:
            print(f"DEBUG: RA axis appears inverted for {star_name} (pixel->world delta RA = {delta if 'delta' in locals() else 'N/A'}). Flipping X axis for display.", flush=True)

        # # Prépare extents X (small image) et X_DOLP (large DoLP) en tenant compte de l'inversion
        # def x_extent_for(xmin, xmax, invert):
        #     return [xmax, xmin] if invert else [xmin, xmax]

        # x_extent = x_extent_for(cutout.xmin, cutout.xmax, invert_x)
        # x_extent_dolp = x_extent_for(cutout_dolp.xmin, cutout_dolp.xmax, invert_x)
        def x_extent_for(xmin, xmax, invert):
            return [xmax, xmin, xmin, xmax] if False else ([xmax, xmin] if invert else [xmin, xmax])
        
        # Récupère les deux filtres du header
        star_name2 = header.get('OBJECT')
        fltr1 = header.get('HIERARCH ESO INS3 OPTI5 NAME', 'Filtre1 inconnu')
        fltr2 = header.get('HIERARCH ESO INS3 OPTI6 NAME', 'Filtre2 inconnu')
        fltr_arr = [fltr1, fltr2]
        n_fsize = data.shape[0]  # nombre de plans dans le cube (souvent 2)

        # Extraction du plan z pour DoLP
        hdu_dolp = fits.open(file_DOLP_star)
        data_dolp = hdu_dolp[0].data
        
        # Appliquer le traitement de pixels chauds/froids uniquement pour BK_Vir
        if star_name == 'BK_Vir':
            if data_dolp.ndim == 3:
                # Traiter chaque plan si c'est un cube 3D
                for z_idx in range(data_dolp.shape[0]):
                    # Traitement des pixels morts/chauds
                    data_dolp[z_idx, :, :] = pfi(data_dolp[z_idx, :, :])
                    # Atténuation supplémentaire des pixels chauds avec clipping au 97e percentile
                    p_max = np.percentile(data_dolp[z_idx, :, :], 97)
                    data_dolp[z_idx, :, :] = np.clip(data_dolp[z_idx, :, :], None, p_max)
            else:
                # Traiter directement si c'est une image 2D
                data_dolp = pfi(data_dolp)
                # Atténuation supplémentaire des pixels chauds avec clipping au 97e percentile
                p_max = np.percentile(data_dolp, 97)
                data_dolp = np.clip(data_dolp, None, p_max)

        #Extraction du plan z pour I
        hdu_I = fits.open(file_I_star)
        data_I = hdu_I[0].data
        # Affichage des informations
        for z in range(n_fsize):
            fig_panel = None
            fig_dolp = None
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
            
            # extent format expected: [xmin, xmax, ymin, ymax] pour imshow
            half = sub_v.shape[0] / 2.0
            x_min = -pix2mas * half
            x_max = pix2mas * (half - 1.0)
            y_min = -pix2mas * half
            y_max = pix2mas * (half - 1.0)
            half_DOLP = size_DOLP[0] / 2.0
            x_min_DOLP = -pix2mas * half_DOLP
            x_max_DOLP = pix2mas * (half_DOLP - 1.0)
            y_min_DOLP = -pix2mas * half_DOLP
            y_max_DOLP = pix2mas * (half_DOLP - 1.0)
            # helper extent (imshow expects [xmin,xmax,ymin,ymax])
            # extent_small = x_extent_for(x_min + 1.0, x_max, invert_x) + [y_min + 1.0, y_max]
            # extent_panel = extent_small.copy()
            # extent_dolp = x_extent_for(x_min_DOLP, x_max_DOLP, invert_x) + [y_min_DOLP, y_max_DOLP]
            
            extent_small = x_extent_for(x_min + 1.0, x_max, invert_x) + [y_min + 1.0, y_max]
            extent_panel = extent_small.copy()
            extent_dolp = x_extent_for(x_min_DOLP, x_max_DOLP, invert_x) + [y_min_DOLP, y_max_DOLP]
            # Garder extents dans l'ordre normal (xmin,xmax,ymin,ymax)
            extent_small = [x_min + 1.0, x_max, y_min + 1.0, y_max]
            extent_panel = extent_small.copy()
            extent_dolp = [x_min_DOLP, x_max_DOLP, y_min_DOLP, y_max_DOLP]
            # inversion visuelle gérée uniquement en appliquant x_sign aux coordonnées tracées
            x_sign = -1.0 if invert_x else 1.0
            # Gestion des cubes I 2D ou 3D
            if data_I.ndim == 3:
                I_z = data_I[z, :, :]
            else:
                I_z = data_I
                
            # Découpe I avec une taille plus grande pour le champ de vue
            cutout_I = Cutout2D(I_z, position=position, size=size_DOLP)
            sub_v_I = cutout_I.data 

            

            # --- Nouveau : crée aussi une découpe I de la même taille que PI (sub_v)
            # pour permettre une division élément par élément sans erreur de broadcast.
            try:
                cutout_I_small = Cutout2D(I_z, position=position, size=size)
                sub_v_I_small = cutout_I_small.data
            except Exception:
                # Si problème (bords), essayer d'ajuster en centrant et reclampant
                sub_v_I_small = sub_v_I

            # Forcer en float pour éviter divisions entières et NaN/Infs non gérés
            sub_v = sub_v.astype(float)
            sub_v_I_small = sub_v_I_small.astype(float)
            sub_v_dolp = sub_v_dolp.astype(float)

            # Calculer la carte DoLP à afficher :
            # - si étoile == 'V854_Cen' : remplacer DoLP classique par PI/I (sub_v / sub_v_I_small)
            # - sinon : garder la DoLP issue du fichier (_DOLP)
            eps = 1e-12
            if star_name == 'V854_Cen':
                with np.errstate(divide='ignore', invalid='ignore'):
                    sub_v_dolp_calc = sub_v / (sub_v_I_small + eps)
                    sub_v_dolp_calc[~np.isfinite(sub_v_dolp_calc)] = 0.0
                sub_v_dolp_display = sub_v_dolp_calc
            else:
                # affichage identique à la DoLP mesurée
                sub_v_dolp_display = sub_v_dolp

            # Division sécurisée : PI / I (DoLP approximé) sur la même grille que sub_v
            #sub_v_dolp_calc = np.divide(sub_v, sub_v_I_small, out=np.zeros_like(sub_v, dtype=float), where=(sub_v_I_small != 0))
            #sub_v_dolp_calc = np.divide(sub_v, sub_v_I, out=np.zeros_like(sub_v), where=sub_v_I!=0)

            
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
            print(f"DEBUG before angle correction: theta_fit(deg) = {np.degrees(theta_f):.3f}, invert_x={invert_x}", flush=True)
            # si l'axe X a été inversé (RA croît vers la gauche), appliquer transformation theta -> pi - theta
            if invert_x:
                theta_fit = np.pi - theta_f
                print(f"DEBUG applied mirror on X: theta_fit_corrected(deg) = {np.degrees(theta_fit):.3f}", flush=True)
            else:
                theta_fit = theta_f
            # ensuite appliquer la conversion locale déjà en place (convention du code)
            theta_f = np.pi / 2 - theta_fit
            theta_f = np.mod(theta_f, 2.0 * np.pi)
            print(f"DEBUG after conversion (theta_f used for rotation) deg = {np.degrees(theta_f):.3f}", flush=True)

            t = np.linspace(0, 2 * pi, nSubDim)
            Ell = np.array([a_f * np.cos(t), b_f * np.sin(t)])
            theta_f = np.pi / 2 - theta_f
            M_rot = np.array([[cos(theta_f), -sin(theta_f)], [sin(theta_f), cos(theta_f)]])
            Ell_rot = np.dot(M_rot, Ell)
            # theta_f a déjà été converti et normalisé ci‑dessus : l'utiliser tel quel pour la rotation
            theta_rot = theta_f
            M_rot = np.array([[cos(theta_rot), -sin(theta_rot)], [sin(theta_rot), cos(theta_rot)]])
            Ell_rot = np.dot(M_rot, Ell)
            # Traduire l'ellipse pour la placer au centroïde trouvé par le fit (coord en pixels)
            Ell_rot[0, :] += x_f
            Ell_rot[1, :] += y_f

            nSubDim = sub_v.shape[0]
            x_mas = (np.arange(nSubDim) - nSubDim // 2) * pix2mas
            y_mas = (np.arange(nSubDim) - nSubDim // 2) * pix2mas
            # theta_f doit être l'angle de rotation en radians (déjà converti/normalisé ci‑dessus)
            # theta_rot = theta_f
            # M_rot = np.array([[cos(theta_rot), -sin(theta_rot)], [sin(theta_rot), cos(theta_rot)]])
            # Ell_rot = np.dot(M_rot, Ell)
            # # Traduire l'ellipse pour la placer au centroïde trouvé par le fit (coord en pixels)
            # Ell_rot[0, :] += x_f
            # Ell_rot[1, :] += y_f
            # nSubDim = sub_v.shape[0]
            # x_mas = (np.arange(nSubDim) - nSubDim // 2) * pix2mas
            # y_mas = (np.arange(nSubDim) - nSubDim // 2) * pix2mas
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
            # Ne garder que le filtre spécifique défini pour cette étoile
            selected_filter = star_specific_filter.get(star_name)
            if selected_filter and fltr_arr[z] != selected_filter:
                continue
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
                'theta_deg': float((np.degrees(theta_fit)) % 180.0),
                'Cout': best_cost,
                'Contraste': best_threshold
            })
            print(f"Traitement : filtre={fltr_arr[z]}, seuil={best_threshold:.4f}, fit_cost={best_cost:.4f}")

            # Recalcule les axes pour le champ DoLP élargi
            x_min_DOLP = -pix2mas * nSubDim_DOLP // 2
            x_max_DOLP = pix2mas * (nSubDim_DOLP // 2 - 1)
            y_min_DOLP = -pix2mas * nSubDim_DOLP // 2
            y_max_DOLP = pix2mas * (nSubDim_DOLP // 2 - 1)
            
            

            # === Creation du panel DoLP+contours  & PI+fit ===

            fig_panel, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            # DoLP+isocontours à gauche
            
            # ============================================================================
            # CALCUL AUTOMATIQUE DE VMAX (APPROCHE HYBRIDE)
            # ============================================================================
            if USE_AUTO_VMAX:
                from scipy.ndimage import gaussian_filter
                
                # 1. Définir zone annulaire externe pour estimer le bruit
                center = nSubDim_DOLP // 2
                y_grid, x_grid = np.ogrid[:nSubDim_DOLP, :nSubDim_DOLP]
                r_grid = np.sqrt((x_grid - center)**2 + (y_grid - center)**2)
                r_max = nSubDim_DOLP // 2
                
                # Zone de bruit : anneau externe (70-90% du rayon)
                mask_bruit = (r_grid > 0.7 * r_max) & (r_grid < 0.9 * r_max)
                pixels_bruit = sub_v_dolp_display[mask_bruit]
                
                # 2. Estimer le bruit (médiane + MAD)
                median_bruit = np.nanmedian(pixels_bruit)
                mad_bruit = np.nanmedian(np.abs(pixels_bruit - median_bruit))
                sigma_bruit = 1.4826 * mad_bruit  # conversion MAD → std
                seuil_signal = median_bruit +  1* sigma_bruit  # seuil 2-sigma (standard astronomie)
                
                # 3. Lissage gaussien léger (optionnel, sigma=0.7)
                sub_v_dolp_smooth = gaussian_filter(sub_v_dolp_display, sigma=0.6)

                # 4. Masquage des pixels sous le seuil (bruit) — remis en place, la variante "sans masque" est commentée ci-dessous
                sub_v_dolp_clean = sub_v_dolp_smooth.copy()
                sub_v_dolp_clean[sub_v_dolp_clean < seuil_signal] = np.nan

                # Variante sans masquage (commentée pour l'instant) :
                # sub_v_dolp_clean = sub_v_dolp_smooth

                # 5. Calculer vmax sur les pixels de signal détecté
                pixels_signal = sub_v_dolp_clean[~np.isnan(sub_v_dolp_clean)]
                if len(pixels_signal) > 0:
                    vmax_dolp = np.nanpercentile(pixels_signal, 99)
                else:
                    vmax_dolp = np.nanpercentile(sub_v_dolp_display, 99)
                
                vmin_dolp = 0  # Toujours partir de 0 pour DoLP
                
                # Utiliser l'image lissée et masquée pour l'affichage
                sub_v_dolp_display = sub_v_dolp_clean
                
                # Gamma adaptatif : plus fort si vmax est grand (besoin d'assombrir) — COMMENTÉ pour désactiver le contraste
                # gamma = 1 + 0.5 * min(vmax_dolp / 0.05, 1.0)  # varie entre 1.0 et 1.5
                gamma = 1.0  # Pas de correction gamma (contraste désactivé)
                
                from matplotlib.colors import PowerNorm
                norm_dolp = PowerNorm(gamma=gamma, vmin=vmin_dolp, vmax=vmax_dolp) if gamma != 1.0 else None
                
                print(f"  → Bruit: {median_bruit:.4f} ± {sigma_bruit:.4f} | Seuil: {seuil_signal:.4f} | vmax auto: {vmax_dolp:.4f} | gamma: {gamma:.2f}")
                
            else:
                # Mode manuel (dictionnaires)
                vrange = custom_dolp_vrange.get(star_name, {})
                vmin_dolp = vrange.get('vmin') if vrange.get('vmin') is not None else np.nanmin(sub_v_dolp_display)
                vmax_dolp = vrange.get('vmax') if vrange.get('vmax') is not None else np.nanmax(sub_v_dolp_display)
                
                # Correction gamma pour assombrir les images DoLP visuellement (sans modifier les données)
                from matplotlib.colors import PowerNorm
                gamma = custom_dolp_gamma.get(star_name, 1.0)  # 1.0 = pas de correction
                norm_dolp = PowerNorm(gamma=gamma, vmin=vmin_dolp, vmax=vmax_dolp) if gamma != 1.0 else None
            
            # Fond gris pour mieux voir les NaN (cohérent avec la figure DoLP seule)
            if USE_AUTO_VMAX:
                ax1.set_facecolor('#2a2a2a')

            if norm_dolp is not None:
                im1 = ax1.imshow(sub_v_dolp_display, cmap='plasma', origin='lower', norm=norm_dolp, extent=extent_dolp, interpolation='bilinear')
            else:
                im1 = ax1.imshow(sub_v_dolp_display, cmap='plasma', origin='lower', vmin=vmin_dolp, vmax=vmax_dolp, extent=extent_dolp, interpolation='bilinear')
            # Niveaux de contours automatiques selon la plage DoLP
            dolp_min = np.nanmin(sub_v_dolp_display)
            dolp_max = np.nanmax(sub_v_dolp_display)
            # Personnalisation des niveaux de contours DoLP pour certaines étoiles
            
            contour_levels = custom_dolp_contours.get(star_name, np.linspace(dolp_min, dolp_max, 5))
            cs = ax1.contour(sub_v_dolp_display, levels=contour_levels, colors='white', linewidths=1.5, origin='lower', extent=[x_min_DOLP, x_max_DOLP, y_min_DOLP, y_max_DOLP])
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
            # Définitions par défaut pour éviter UnboundLocalError si les données de distance/rayon manquent
            radii_rstar = []
            radii_mas = []
            linestyles = []
            if not match_row.empty:
                radius_star_rsun = match_row.iloc[0]['Radius']
                distance_star_pc = match_row.iloc[0]['Distance']
                if (distance_star_pc < 150) or (radius_star_rsun > 500):
                    use_fixed_circles = True
                    affected_stars_dolp.append({'Étoile': star_name, 'Distance_pc': distance_star_pc, 'Radius_Rsun': radius_star_rsun})
            # Création des cercles en rayons stellaires
        # Dictionnaire pour personnaliser la taille de la barre d'échelle (en R*)
            
            # 'AK_Hya': 5,
            # Autres étoiles et leur taille de barre personnalisée

            if distance_star_pc is not None and radius_star_rsun is not None and radius_star_rsun > 0 and distance_star_pc > 0:
                R_sun_cm = 6.957e10
                radius_star_cm = radius_star_rsun * R_sun_cm
                distance_star_cm = distance_star_pc * 3.0857e18
                radius_star_rad = radius_star_cm / distance_star_cm
                radius_stellar_mas = radius_star_rad * 206265 * 1000
                if use_fixed_circles:
                    if star_name == 'SW_Col':
                        radii_rstar = [20,  120]
                        linestyles = ['--',  '-.']
                    elif star_name == 'U_Her':
                        radii_rstar = [0, 9]
                        linestyles = ['--', '-.']
                    elif star_name == 'RT_Vir':
                        radii_rstar = [0,  5]
                        linestyles = ['--',  ':']
                    elif star_name == 'W_Peg':
                        radii_rstar = [0, 16]
                        linestyles = ['--', '-.']
                    elif star_name == 'W_Aql':
                        radii_rstar = [5, 53]
                        linestyles = ['--', '-.']
                    elif star_name == 'AK_Hya':
                        radii_rstar = [3, 21]
                        linestyles = ['--', '-.']
                    elif star_name == 'W_Hya':
                        radii_rstar = [1.05, 5]
                        linestyles = ['--', '-.']
                    elif star_name == 'Mira':
                        radii_rstar = [10, 15]
                        linestyles = ['--', '-.']
                    elif star_name == 'R_Dor':
                        radii_rstar = [1, 4.5]
                        linestyles = ['--', '-.']
                    elif star_name == 'R_Crt':
                        radii_rstar = [0, 40]
                        linestyles = ['--', '-.']
                    elif star_name == 'V_PsA':
                        radii_rstar = [2, 34]
                        linestyles = ['--', '-.']
                    elif star_name == 'S_Pav':
                        radii_rstar = [0, 12]
                        linestyles = ['--', '-.']
                    elif star_name == 'R_Hya':
                        radii_rstar = [0, 10]
                        linestyles = ['--', '-.']
                    elif star_name == 'BK_Vir':
                        radii_rstar = [0, 3]
                        linestyles = ['--', '-.']
                    elif star_name == 'GY_Aql':
                        radii_rstar = [1.2, 17]
                        linestyles = ['--', '-.']
                    elif star_name == 'SW_Vir':
                        radii_rstar = [1.2, 16.5]
                        linestyles = ['--', '-.']
                    elif star_name == 'Pi.01_Gru':
                        radii_rstar = [3.5, 10.5]
                        linestyles = ['--', '-.']
                    elif star_name == 'L02_Pup':
                        radii_rstar = [3, 25]
                        linestyles = ['--', '-.']
                    elif star_name == 'V854_Cen':
                        radii_rstar = [10, 60]
                        linestyles = ['--', '-.']           
                    else:
                        radii_rstar = [3, 10]
                        linestyles = ['--', '-.']
                    
                    
                else:
                    if star_name == 'SW_Col':
                        radii_rstar = [20, 120]
                        linestyles = ['--', '-.']
                    elif star_name == 'U_Her':
                        radii_rstar = [0, 8]
                        linestyles = ['--', '-.']
                    elif star_name == 'RT_Vir':
                        radii_rstar = [0, 5]
                        linestyles = ['--', ':', '-.']
                    elif star_name == 'W_Peg':
                        radii_rstar = [0, 12]
                        linestyles = ['--', '-.']
                    elif star_name == 'W_Aql':
                        radii_rstar = [5,  45]
                        linestyles = ['--',  '-.']
                    elif star_name == 'AK_Hya':
                        radii_rstar = [3, 21]
                        linestyles = ['--', '-.']
                    elif star_name == 'W_Hya':
                        radii_rstar = [1.05, 8]
                        linestyles = ['--', '-.']
                    elif star_name == 'Mira':
                        radii_rstar = [0, 10]
                        linestyles = ['--', '-.']
                    elif star_name == 'R_Dor':
                        radii_rstar = [1, 4.5]
                        linestyles = ['--', '-.']
                    elif star_name == 'R_Crt':
                        radii_rstar = [0, 40]
                        linestyles = ['--', '-.']
                    elif star_name == 'Pi.01_Gru':
                        radii_rstar = [0, 7]
                        linestyles = ['--', '-.']
                    elif star_name == 'V_PsA':
                        radii_rstar = [3, 20]
                        linestyles = ['--', '-.']
                    elif star_name == 'BK_Vir':
                        radii_rstar = [0, 3]
                        linestyles = ['--', '-.']
                    elif star_name == 'V854_Cen':
                        radii_rstar = [10, 140]
                        linestyles = ['--', '-.']
                    else:
                        radii_rstar = [3, 5, 10, 20]
                        linestyles = ['--', '-.', ':', '-.']
                radii_mas = [r * radius_stellar_mas for r in radii_rstar]
                circles = [Circle((center_x, center_y), r_mas, edgecolor='cyan', facecolor='none', lw=2, linestyle=ls)
                           for r_mas, ls in zip(radii_mas, linestyles)]
                # Choice: Plot either 3R* circle OR personalized circles
                try:
                    if PLOT_3RSTAR_ON_DOLP and radius_stellar_mas is not None and radius_stellar_mas > 0:
                        # Apply to SW_Col and stars listed in clearly_resolved
                        if (star_name == 'SW_Col') or (star_name in clearly_resolved):
                            circle_3R_mas = 3.0 * radius_stellar_mas
                            # Styled circle for presentation: dash-dot white line with black outline
                            circ3 = Circle((center_x, center_y), circle_3R_mas, edgecolor='cyan', facecolor='none', lw=4, linestyle='-.', zorder=5)
                            try:
                                circ3.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
                            except Exception:
                                pass
                            ax1.add_patch(circ3)
                    else:
                        # Plot personalized circles instead when 3R* is disabled
                        for circ in circles:
                            ax1.add_patch(circ)
                except Exception:
                    # Defensive: if radius not available or other error, skip drawing
                    pass
                legend_unit = 'R$_\star$'
                # === Calcul, affichage et sauvegarde du rayon du grand cercle ===
                grand_rstar = max(radii_rstar) if radii_rstar else None
                petit_rstar = min(radii_rstar) if radii_rstar else None
                petit_mas = min(radii_mas) if radii_mas else None
                grand_mas = max(radii_mas) if radii_mas else None
                print(f"Rayon du grand cercle pour {star_name} : {grand_rstar:.3f} R* | {grand_mas:.1f} mas", flush=True)
                print(f"Rayon du petit cercle pour {star_name} : {petit_rstar:.3f} R* | {petit_mas:.1f} mas", flush=True)
                # Sauvegarde dans un fichier CSV dédié
                radii_csv = os.path.join(outdir, 'radii_grand_cercle.csv')
                import csv
                # Ajout ou création du fichier
                write_header = not os.path.exists(radii_csv)
                with open(radii_csv, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(['Étoile', 'Rayon_petit_cercle_Rstar', 'Rayon_petit_cercle_mas', 'Rayon_grand_cercle_Rstar', 'Rayon_grand_cercle_mas'])
                    writer.writerow([star_name, petit_rstar, petit_mas, grand_rstar, grand_mas])
            else:
                print(f"⚠️ Distance ou rayon stellaire non trouvés pour l'étoile {star_name}, cercles non tracés.")
            # Croix rouge au centre
            #ax1.scatter([center_x], [center_y], color='red', marker='x', s=80, zorder=10)
            # Ajout d'une barre d'échelle physique en R* en bas à droite
            if distance_star_pc is not None and radius_star_rsun is not None and radius_star_rsun > 0 and distance_star_pc > 0:
                # Barre d'échelle en R* (personnalisable)
                scale_val = custom_scale_bar.get(star_name, 10)  # 10 R* par défaut
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
                ax1.text(x_bar_axes + bar_length_axes/2, y_bar_axes + 0.01, f'{scale_val:.0f}{legend_unit}', color='white', fontsize=label_size, ha='center', va='bottom',  transform=ax1.transAxes, path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
            else:
                # Si la distance ou le rayon stellaire est manquant, trace une barre d'échelle de secours en mas
                print(f"⚠️ Distance ou rayon stellaire non trouvés pour l'étoile {star_name}, barre d'échelle en R* non tracée. Affichage d'une barre de secours en mas.")
                # Barre de secours: 1/10 du champ DoLP en mas
                try:
                    fallback_scale_mas = (nSubDim_DOLP / 10.0) * pix2mas
                    bar_length_axes = fallback_scale_mas / (x_max_DOLP - x_min_DOLP)
                    x_bar_axes = 0.90 - bar_length_axes
                    y_bar_axes = 0.04
                    ax1.plot([x_bar_axes, x_bar_axes + bar_length_axes], [y_bar_axes, y_bar_axes], color='white', lw=3, transform=ax1.transAxes, solid_capstyle='butt')
                    ax1.text(x_bar_axes + bar_length_axes/2, y_bar_axes + 0.01, f'{int(fallback_scale_mas)} mas', color='white', fontsize=label_size, ha='center', va='bottom', transform=ax1.transAxes, path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
                except Exception:
                    # En cas d'erreur, se contente d'afficher le message d'avertissement
                    pass
            

            # Petit panel DoLP+PI : avec labels et graduations
            if label_size == label_size_great_panel:
                # Grand panel : graduations avec labels
                ax1.set_xlabel('Relative RA (mas)', fontsize=label_size)
                ax1.set_ylabel('Relative Dec (mas)', fontsize=label_size)
                ax1.tick_params(axis='both', labelsize=label_size, width=1.2)
                ax1.locator_params(axis='x', nbins=5)
                ax1.locator_params(axis='y', nbins=5)
            else:
                # Petit panel : avec graduations et labels d'axes
                ax1.tick_params(axis='both', labelsize=label_size, width=1.2)
                ax1.set_xlabel('Relative RA (mas)', fontsize=label_size)
                ax1.set_ylabel('Relative Dec (mas)', fontsize=label_size)

            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes('right', size='5%', pad=0.03)
            cb1 = fig_panel.colorbar(im1, cax=cax1, orientation='vertical')
            cb1.ax.tick_params(labelsize=label_size)

            # Détecte l'exposant commun
            vmin, vmax = cb1.vmin, cb1.vmax
            if vmax != 0:
                common_exp = int(np.floor(np.log10(np.abs(vmax))))
            else:
                common_exp = 0

            # Formatter: affiche mantisse uniquement (1 décimale)
            def mantissa_1decimal(x, pos):
                if x == 0:
                    return ''  # Masquer la graduation 0.0
                mantissa = x / (10 ** common_exp)
                return f'{mantissa:.1f}'

            cb1.ax.yaxis.set_major_formatter(FuncFormatter(mantissa_1decimal))

            # Affiche l'exposant en haut à droite (forcé)
            if label_size_great_panel != label_size_small_panel:
                exp_x = -2.4 + (label_size - label_size_small_panel) * (-4.0 + 2.4) / (label_size_great_panel - label_size_small_panel)
            else:
                exp_x = -2.4
            cb1.ax.yaxis.get_offset_text().set_visible(False)
            cb1.ax.text(exp_x, 1.001, f'×1e{common_exp}', 
                            transform=cb1.ax.transAxes, 
                            fontsize=label_size, 
                            verticalalignment='bottom',
                            horizontalalignment='left',
                            clip_on=False)
            cb1.ax.yaxis.get_offset_text().set(size=label_size)
            #plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10)
            # cb1.ax.tick_params(labelsize=label_size)
            # cmapProp = {'drawedges': True}            
            # cb1.formatter.set_powerlimits((0, 0))
            # cb1.ax.yaxis.get_offset_text().set(size=label_size)

            # Boîte en haut-gauche pour le nom de l'étoile
            ax1.text(0.02, 0.95, f'{star_name2}', transform=ax1.transAxes, fontsize=label_size, color='white', va='top', path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
            
            # Boîte en bas-gauche pour le filtre
            ax1.text(0.02, 0.02, f'{fltr_arr[z]}', transform=ax1.transAxes, fontsize=label_size, color='white', va='bottom', path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])

            # PI + ellipse à droite
            # im2 = ax2.imshow(
            #     np.log10(sub_v + np.abs(np.min(sub_v)) + 10),
            #     cmap='inferno',
            #     origin='lower',
            #     extent=[x_min+1, x_max, y_min+1, y_max]
            # )
            im2 = ax2.imshow(
                np.log10(sub_v + np.abs(np.min(sub_v)) + 10),
                cmap='plasma',
                origin='lower',
                extent=extent_panel
            )
            # Ajout orientation N-W en haut à droite sur la figure PI (comme pour la figure DoLP séparée)
            arrow_len = 0.03 * (x_max - x_min)
            x_arrow = x_max - 0.04 * (x_max - x_min)
            # place near the top of the panel (mirror of DoLP placement)
            y_arrow = y_max - 0.20 * (y_max - y_min)
            # Flèche Nord (verticale vers le haut)
            ax2.arrow(x_arrow, y_arrow, 0, arrow_len, head_width=0.02*arrow_len, head_length=0.04*arrow_len, fc='white', ec='white', lw=2)
            # Flèche Ouest (horizontale vers la gauche)
            ax2.arrow(x_arrow, y_arrow, -arrow_len, 0, head_width=0.02*arrow_len, head_length=0.04*arrow_len, fc='white', ec='white', lw=2)
            offset_label = 15
            ax2.text(x_arrow, y_arrow + arrow_len + offset_label, 'N', color='white', fontsize=label_size, ha='center', va='bottom', path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
            ax2.text(x_arrow - arrow_len - offset_label, y_arrow, 'E', color='white', fontsize=label_size, ha='right', va='center', path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
            # Barre d'échelle en bas à droite du panneau PI
            try:
                if distance_star_pc is not None and radius_star_rsun is not None and radius_star_rsun > 0 and distance_star_pc > 0:
                    # Affiche en R* si possible
                    scale_val_panel = custom_scale_bar.get(star_name, 10)
                    pi_scale_val_panel = 0.5*scale_val_panel  #  pour le panneau PI
                    R_sun_cm = 6.957e10
                    radius_star_cm = radius_star_rsun * R_sun_cm
                    distance_star_cm = distance_star_pc * 3.0857e18
                    radius_star_rad = radius_star_cm / distance_star_cm
                    radius_stellar_mas = radius_star_rad * 206265 * 1000
                    scale_mas_panel = pi_scale_val_panel * radius_stellar_mas
                    legend_unit = 'R$_\\star$'
                    bar_length_axes = scale_mas_panel / (x_max - x_min)
                    x_bar_axes = 0.90 - bar_length_axes
                    y_bar_axes = 0.04
                    ax2.plot([x_bar_axes, x_bar_axes + bar_length_axes], [y_bar_axes, y_bar_axes], color='white', lw=3, transform=ax2.transAxes, solid_capstyle='butt')
                    ax2.text(x_bar_axes + bar_length_axes/2, y_bar_axes + 0.01, f'{pi_scale_val_panel:.0f}{legend_unit}', color='white', fontsize=label_size, ha='center', va='bottom', transform=ax2.transAxes, path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
                else:
                    # Barre de secours en mas
                    fallback_scale_mas = (nSubDim / 10.0) * pix2mas
                    bar_length_axes = fallback_scale_mas / (x_max - x_min)
                    x_bar_axes = 0.90 - bar_length_axes
                    y_bar_axes = 0.04
                    ax2.plot([x_bar_axes, x_bar_axes + bar_length_axes], [y_bar_axes, y_bar_axes], color='white', lw=3, transform=ax2.transAxes, solid_capstyle='butt')
                    ax2.text(x_bar_axes + bar_length_axes/2, y_bar_axes + 0.01, f'{int(fallback_scale_mas)} mas', color='white', fontsize=label_size, ha='center', va='bottom', transform=ax2.transAxes, path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
            except Exception:
                pass
            # Petit panel DoLP+PI : PAS de labels ni graduations
            # if label_size == label_size_great_panel:
            #     # Grand panel : graduations sans labels (inversé temporairement)
            #     # ax2.set_xlabel('Relative RA (mas)', fontsize=label_size)
            #     # ax2.set_ylabel('Relative Dec (mas)', fontsize=label_size)
            #     ax2.tick_params(axis='both', labelsize=label_size, width=1.2)
            #     ax2.locator_params(axis='x', nbins=5)
            #     ax2.locator_params(axis='y', nbins=5)
            # else:
            #     # Petit panel : PAS de graduations ni labels d'axes (inversé temporairement)
            #     ax2.tick_params(axis='both', labelsize=label_size, width=1.2)
            #     ax2.locator_params(axis='x', nbins=5)
            #     ax2.locator_params(axis='y', nbins=5)
            #     ax2.axes.yaxis.set_ticklabels([])
            #     ax2.set_xticks([])
            #     ax2.set_yticks([])
            #     ax2.tick_params(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.tick_params(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)

            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes('right', size='5%', pad=0.03)
            cb2 = fig_panel.colorbar(im2, cax=cax2, orientation='vertical')
            cb2.ax.tick_params(labelsize=label_size)
            cb2.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
            #ax2.plot(x_contour_mas, y_contour_mas, color='cyan', linewidth=2, linestyle='--')
            ax2.scatter([x_centroid_mas], [y_centroid_mas], color='red', marker='x')
            ax2.text(0.02, 0.95, f'{star_name2}', transform=ax2.transAxes, fontsize=label_size, color='white', va='top', path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
            ax2.text(0.02, 0.02, f'{fltr_arr[z]}', transform=ax2.transAxes, fontsize=label_size, color='white', va='bottom', path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])

            plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10, wspace=0.15)
            fig_panel_name = f'{star_name}_{obsmod}_{fltr_arr[z]}_{z}_PI_DoLP_panel'
            plt.savefig(os.path.join(outdir, fig_panel_name + '.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(outdir, fig_panel_name + '.pdf'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(outdir_panels +'/png', fig_panel_name + '.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(outdir_panels +'/pdf', fig_panel_name + '.pdf'), dpi=300, bbox_inches='tight')
            if star_name== 'V854_Cen':
                plt.savefig(os.path.join(outdir_panels +'/eps', fig_panel_name + '.eps'), format='eps', dpi=300, bbox_inches='tight')
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
                if fig_panel is not None:
                    plt.close(fig_panel)
                # === Export CDS : copier et renommer les fichiers DoLP et PI réduits pour l'étoile + filtre spécifique ===
                try:
                    cds_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/CDS_export'
                    os.makedirs(cds_dir, exist_ok=True)
                    def _safe_fname(s):
                        return re.sub(r'[^0-9a-zA-Z]+', '_', str(s).lower()).strip('_')
                    safe_star = _safe_fname(star_name)
                    src_dolp = file_DOLP_star
                    src_pi = file_PI_star
                    dst_dolp = os.path.join(cds_dir, f"{safe_star}_dolp.fits")
                    dst_pil = os.path.join(cds_dir, f"{safe_star}_pil.fits")
                    if os.path.exists(src_dolp):
                        shutil.copy2(src_dolp, dst_dolp)
                        print(f"Copié DoLP → {dst_dolp}", flush=True)
                    else:
                        print(f"⚠️ Source DoLP manquante: {src_dolp}", flush=True)
                    if os.path.exists(src_pi):
                        shutil.copy2(src_pi, dst_pil)
                        print(f"Copié PI → {dst_pil}", flush=True)
                    else:
                        print(f"⚠️ Source PI manquante: {src_pi}", flush=True)
                except Exception as e:
                    print(f"⚠️ Erreur copie CDS pour {star_name}: {e}", flush=True)
            else:
                if fig_panel is not None:
                    plt.close(fig_panel)
                print(f"Panel ignoré pour {star_name} [{fltr_arr[z]}] : filtre non spécifique.", flush=True)

            # Enregistrement de la figure de gauche (DoLP seule, avec contours et cercles) dans un dossier dédié
            outdir_dolp_only = '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Morphologies_contours/DoLP_only'
            os.makedirs(outdir_dolp_only, exist_ok=True)
            fig_dolp, ax_dolp = plt.subplots(figsize=(6, 5))
            
            # Appliquer fond gris pour meilleure visibilité des zones NaN (si mode auto)
            if USE_AUTO_VMAX:
                ax_dolp.set_facecolor('#2a2a2a')
            
            if norm_dolp is not None:
                im_dolp = ax_dolp.imshow(sub_v_dolp_display, cmap='plasma', origin='lower', norm=norm_dolp, 
                                        extent=extent_dolp, interpolation='bilinear')
            else:
                im_dolp = ax_dolp.imshow(sub_v_dolp_display, cmap='plasma', origin='lower', vmin=vmin_dolp, 
                                        vmax=vmax_dolp, extent=extent_dolp, interpolation='bilinear')
            # Pour les contours, utiliser les vraies valeurs min/max de l'image (pas les limites de la colorbar)
            dolp_min_real = np.nanmin(sub_v_dolp_display)
            dolp_max_real = np.nanmax(sub_v_dolp_display)
            contour_levels = custom_dolp_contours.get(star_name, np.linspace(dolp_min_real, dolp_max_real, 5))
            cs_dolp = ax_dolp.contour(sub_v_dolp_display, levels=contour_levels, colors='white', linewidths=1., origin='lower', extent=[x_min_DOLP, x_max_DOLP, y_min_DOLP, y_max_DOLP])
            ax_dolp.clabel(cs_dolp, inline=True, fontsize=10, fmt='%.2f')
            
            # Ajout des cercles comme dans le panel
            try:
                if PLOT_3RSTAR_ON_DOLP and 'radius_stellar_mas' in locals() and radius_stellar_mas is not None and radius_stellar_mas > 0:
                    # Draw only a single white circle at exactly 3 R* (thicker line)
                    circle_3R_mas = 3.0 * radius_stellar_mas
                    circ_dolp = Circle((center_x, center_y), circle_3R_mas, edgecolor='cyan', facecolor='none', lw=4, linestyle='-.', zorder=5)
                    try:
                        circ_dolp.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
                    except Exception:
                        pass
                    ax_dolp.add_patch(circ_dolp)
                else:
                    for r_mas, ls in zip(radii_mas, linestyles):
                        circ_dolp = Circle((center_x, center_y), r_mas, edgecolor='cyan', facecolor='none', lw=2, linestyle=ls)
                        ax_dolp.add_patch(circ_dolp)
            except Exception:
                # If something goes wrong (missing values), fallback to adding nothing
                pass
            # Ajout du centroïde (croix rouge) sur la figure DoLP isolée
            ax_dolp.scatter([center_x], [center_y], color='red', marker='x', s=80, zorder=10)
            # Barre d'échelle en R* (identique au panel)
            if distance_star_pc is not None and radius_star_rsun is not None and radius_star_rsun > 0 and distance_star_pc > 0:
                scale_val = custom_scale_bar.get(star_name, 10)  # 10 R* par défaut
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
                ax_dolp.text(x_bar_axes + bar_length_axes/2, y_bar_axes + 0.01, f'{scale_val:.0f}{legend_unit}', color='white', fontsize=label_size, ha='center', va='bottom',  transform=ax_dolp.transAxes, path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
            else:
                # Barre de secours en mas si pas de distance/rayon
                try:
                    fallback_scale_mas = (nSubDim_DOLP / 10.0) * pix2mas
                    bar_length_axes = fallback_scale_mas / (x_max_DOLP - x_min_DOLP)
                    x_bar_axes = 0.90 - bar_length_axes
                    y_bar_axes = 0.04
                    ax_dolp.plot([x_bar_axes, x_bar_axes + bar_length_axes], [y_bar_axes, y_bar_axes], color='white', lw=3, transform=ax_dolp.transAxes, solid_capstyle='butt')
                    ax_dolp.text(x_bar_axes + bar_length_axes/2, y_bar_axes + 0.01, f'{int(fallback_scale_mas)} mas', color='white', fontsize=label_size, ha='center', va='bottom', transform=ax_dolp.transAxes, path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
                except Exception:
                    pass
            # Flèches N-W en haut à droite (identique au panel)
            arrow_len = 0.04 * (x_max_DOLP - x_min_DOLP)
            x_arrow = x_max_DOLP - 0.04 * (x_max_DOLP - x_min_DOLP)# vers la gauche
            y_arrow = y_max_DOLP - 0.17 * (y_max_DOLP - y_min_DOLP)# vers le bas
            ax_dolp.arrow(x_arrow, y_arrow, 0, arrow_len, head_width=0.02*arrow_len, head_length=0.04*arrow_len, fc='white', ec='white', lw=2)
            ax_dolp.arrow(x_arrow, y_arrow, -arrow_len, 0, head_width=0.02*arrow_len, head_length=0.04*arrow_len, fc='white', ec='white', lw=2)
            offset_label = 15
            ax_dolp.text(x_arrow, y_arrow + arrow_len + offset_label, 'N', color='white', fontsize=label_size, ha='center', va='bottom', path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
            ax_dolp.text(x_arrow - arrow_len - offset_label, y_arrow, 'E', color='white', fontsize=label_size, ha='right', va='center', path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
            # Carte DoLP seule : avec labels et graduations
            ax_dolp.set_xlabel('Relative RA (mas)', fontsize=label_size)
            ax_dolp.set_ylabel('Relative Dec (mas)', fontsize=label_size)
            ax_dolp.tick_params(axis='both', labelsize=label_size, width=1.2)
            ax_dolp.locator_params(axis='x', nbins=5)
            ax_dolp.locator_params(axis='y', nbins=5)
            
            # Boîte en haut-gauche pour le nom de l'étoile (figure DoLP seule)
            ax_dolp.text(0.02, 0.95, f'{star_name2}', transform=ax_dolp.transAxes, fontsize=label_size, color='white', va='top', path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
            
            # Boîte en bas-gauche pour le filtre (figure DoLP seule)
            ax_dolp.text(0.02, 0.02, f'{fltr_arr[z]}', transform=ax_dolp.transAxes, fontsize=label_size, color='white', va='bottom', path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
            
            # ax_dolp.set_xticks([])
            # ax_dolp.set_yticks([])
            # ax_dolp.tick_params(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)

            divider_dolp = make_axes_locatable(ax_dolp)
            # if star_name == 'R_Crt':
            #     cax_dolp = divider_dolp.append_axes('right', size='5%', pad=0.03)
            #     cb_dolp = fig_dolp.colorbar(im_dolp, cax=cax_dolp, orientation='vertical')
            # else:
            
            cax_dolp = divider_dolp.append_axes('right', size='5%', pad=0.03)
            cb_dolp = fig_dolp.colorbar(im_dolp, cax=cax_dolp, orientation='vertical')
            cb_dolp.ax.tick_params(labelsize=label_size)

            # Détecte l'exposant commun
            vmin, vmax = cb_dolp.vmin, cb_dolp.vmax
            if vmax != 0:
                common_exp = int(np.floor(np.log10(np.abs(vmax))))
            else:
                common_exp = 0

            # Formatter: affiche mantisse uniquement (1 décimale)
            def mantissa_1decimal(x, pos):
                if x == 0:
                    return ''  # Masquer la graduation 0.0
                mantissa = x / (10 ** common_exp)
                return f'{mantissa:.1f}'

            cb_dolp.ax.yaxis.set_major_formatter(FuncFormatter(mantissa_1decimal))

            # Affiche l'exposant en haut à droite (forcé)
            if label_size_great_panel != label_size_small_panel:
                exp_x = -2.4 + (label_size - label_size_small_panel) * (-4.0 + 2.4) / (label_size_great_panel - label_size_small_panel)
            else:
                exp_x = -2.4
            cb_dolp.ax.yaxis.get_offset_text().set_visible(False)
            cb_dolp.ax.text(exp_x, 1.001, f'×1e{common_exp}', 
                transform=cb_dolp.ax.transAxes, 
                fontsize=label_size, 
                verticalalignment='bottom',
                horizontalalignment='left',
                clip_on=False)
            
            cb_dolp.ax.yaxis.get_offset_text().set(size=label_size)
            plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10)
            fig_dolp_name = f'{star_name}_{obsmod}_{fltr_arr[z]}_{z}_DoLP_only'
            
            plt.savefig(os.path.join(outdir, fig_dolp_name + '.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(outdir, fig_dolp_name + '.pdf'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(f'{outdir_dolp_only}/pdf', fig_dolp_name + '.pdf'), dpi=300, bbox_inches='tight')
            if star_name== 'V854_Cen':
                plt.savefig(os.path.join(outdir, fig_dolp_name + '.eps'), format='eps', dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(outdir_dolp_only, fig_dolp_name + '.png'), dpi=300, bbox_inches='tight')

            # === Figure PI/I pour BK_Vir et RT_Vir (comparaison avec DoLP) ===
            if star_name in ['BK_Vir', 'RT_Vir']:
                fig_pi_over_i, ax_pi_over_i = plt.subplots(1, 1, figsize=(6, 5))
                
                # Calculer PI/I sur la grande découpe (comme DoLP)
                cutout_I_large = Cutout2D(I_z, position=position, size=size_DOLP)
                sub_v_I_large = cutout_I_large.data.astype(float)
                
                # Découpe PI sur la même taille
                cutout_PI_large = Cutout2D(intensity, position=position, size=size_DOLP)
                sub_v_PI_large = cutout_PI_large.data.astype(float)
                
                # Calcul PI/I
                eps = 1e-12
                with np.errstate(divide='ignore', invalid='ignore'):
                    pi_over_i = sub_v_PI_large / (sub_v_I_large + eps)
                    pi_over_i[~np.isfinite(pi_over_i)] = 0.0
                
                # Affichage avec les mêmes paramètres que DoLP
                if USE_AUTO_VMAX:
                    from scipy.ndimage import gaussian_filter
                    center = size_DOLP[0] // 2
                    y_grid, x_grid = np.ogrid[:size_DOLP[0], :size_DOLP[1]]
                    r_grid = np.sqrt((x_grid - center)**2 + (y_grid - center)**2)
                    r_max = size_DOLP[0] // 2
                    mask_bruit = (r_grid > 0.7 * r_max) & (r_grid < 0.9 * r_max)
                    pixels_bruit = pi_over_i[mask_bruit]
                    median_bruit = np.nanmedian(pixels_bruit)
                    mad_bruit = np.nanmedian(np.abs(pixels_bruit - median_bruit))
                    sigma_bruit = 1.4826 * mad_bruit
                    seuil_signal = median_bruit + 1 * sigma_bruit
                    pi_over_i_smooth = gaussian_filter(pi_over_i, sigma=1.0)
                    pi_over_i_clean = pi_over_i_smooth.copy()
                    pi_over_i_clean[pi_over_i_clean < seuil_signal] = np.nan
                    pixels_signal = pi_over_i_clean[~np.isnan(pi_over_i_clean)]
                    if len(pixels_signal) > 0:
                        vmax_pi_i = np.nanpercentile(pixels_signal, 97)
                    else:
                        vmax_pi_i = np.nanpercentile(pi_over_i, 97)
                    vmin_pi_i = 0
                    pi_over_i_display = pi_over_i_clean
                    ax_pi_over_i.set_facecolor('#2a2a2a')
                else:
                    vmin_pi_i = 0
                    vmax_pi_i = np.nanpercentile(pi_over_i, 97)
                    pi_over_i_display = pi_over_i
                
                im_pi_i = ax_pi_over_i.imshow(pi_over_i_display, cmap='plasma', origin='lower', 
                                              vmin=vmin_pi_i, vmax=vmax_pi_i, extent=extent_dolp, interpolation='bilinear')
                
                # Contours
                pi_i_min = np.nanmin(pi_over_i_display)
                pi_i_max = np.nanmax(pi_over_i_display)
                contour_levels_pi_i = np.linspace(pi_i_min, pi_i_max, 5)
                cs_pi_i = ax_pi_over_i.contour(pi_over_i_display, levels=contour_levels_pi_i, colors='white', 
                                               linewidths=1., origin='lower', extent=[x_min_DOLP, x_max_DOLP, y_min_DOLP, y_max_DOLP])
                ax_pi_over_i.clabel(cs_pi_i, inline=True, fontsize=10, fmt='%.2f')
                
                # Flèches N-E
                arrow_len = 0.04 * (x_max_DOLP - x_min_DOLP)
                x_arrow = x_max_DOLP - 0.04 * (x_max_DOLP - x_min_DOLP)
                y_arrow = y_max_DOLP - 0.17 * (y_max_DOLP - y_min_DOLP)
                ax_pi_over_i.arrow(x_arrow, y_arrow, 0, arrow_len, head_width=0.02*arrow_len, head_length=0.04*arrow_len, fc='white', ec='white', lw=2)
                ax_pi_over_i.arrow(x_arrow, y_arrow, -arrow_len, 0, head_width=0.02*arrow_len, head_length=0.04*arrow_len, fc='white', ec='white', lw=2)
                offset_label = 0
                ax_pi_over_i.text(x_arrow, y_arrow + arrow_len + offset_label, 'N', color='white', fontsize=label_size, ha='center', va='bottom', path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
                ax_pi_over_i.text(x_arrow - arrow_len - offset_label, y_arrow, 'E', color='white', fontsize=label_size, ha='right', va='center', path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
                
                # Barre d'échelle
                if distance_star_pc is not None and radius_star_rsun is not None and radius_star_rsun > 0 and distance_star_pc > 0:
                    # Pour PI/I, utiliser la même valeur de scale que dans le panneau PI du grand plot (0.5 * scale_val)
                    scale_val_base = custom_scale_bar.get(star_name, 10)
                    scale_val = 0.5 * scale_val_base  # Cohérent avec le panneau PI de la figure combinée
                    R_sun_cm = 6.957e10
                    radius_star_cm = radius_star_rsun * R_sun_cm
                    distance_star_cm = distance_star_pc * 3.0857e18
                    radius_star_rad = radius_star_cm / distance_star_cm
                    radius_stellar_mas = radius_star_rad * 206265 * 1000
                    scale_mas = scale_val * radius_stellar_mas
                    legend_unit = 'R$_\\star$'
                    bar_length_axes = scale_mas / (x_max - x_min)
                    x_bar_axes = 0.90 - bar_length_axes
                    y_bar_axes = 0.04
                    ax_pi_over_i.plot([x_bar_axes, x_bar_axes + bar_length_axes], [y_bar_axes, y_bar_axes], color='white', lw=3, transform=ax_pi_over_i.transAxes, solid_capstyle='butt')
                    ax_pi_over_i.text(x_bar_axes + bar_length_axes/2, y_bar_axes + 0.01, f'{scale_val:.1f}{legend_unit}', color='white', fontsize=label_size, ha='center', va='bottom', transform=ax_pi_over_i.transAxes, path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
                
                # Labels et ticks
                if label_size == label_size_great_panel:
                    ax_pi_over_i.tick_params(axis='both', labelsize=label_size, width=1.2)
                    ax_pi_over_i.locator_params(axis='x', nbins=5)
                    ax_pi_over_i.locator_params(axis='y', nbins=5)
                else:
                    ax_pi_over_i.tick_params(axis='both', labelsize=label_size, width=1.2)
                    ax_pi_over_i.set_xticks([])
                    ax_pi_over_i.set_yticks([])
                    ax_pi_over_i.tick_params(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)
                
                ax_pi_over_i.text(0.02, 0.95, f'{star_name2}', transform=ax_pi_over_i.transAxes, fontsize=label_size, color='white', va='top', path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
                ax_pi_over_i.text(0.02, 0.02, f'{fltr_arr[z]} (PI/I)', transform=ax_pi_over_i.transAxes, fontsize=label_size, color='white', va='bottom', path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
                
                # Colorbar
                divider_pi_i = make_axes_locatable(ax_pi_over_i)
                cax_pi_i = divider_pi_i.append_axes('right', size='5%', pad=0.03)
                cb_pi_i = fig_pi_over_i.colorbar(im_pi_i, cax=cax_pi_i, orientation='vertical')
                cb_pi_i.ax.tick_params(labelsize=label_size)
                vmin_cb, vmax_cb = cb_pi_i.vmin, cb_pi_i.vmax
                if vmax_cb != 0:
                    common_exp_pi_i = int(np.floor(np.log10(np.abs(vmax_cb))))
                else:
                    common_exp_pi_i = 0
                def mantissa_1decimal_pi_i(x, pos):
                    if x == 0:
                        return ''
                    mantissa = x / (10 ** common_exp_pi_i)
                    return f'{mantissa:.1f}'
                cb_pi_i.ax.yaxis.set_major_formatter(FuncFormatter(mantissa_1decimal_pi_i))
                cb_pi_i.ax.text(-2.4, 1.001, f'×1e{common_exp_pi_i}', transform=cb_pi_i.ax.transAxes, 
                               fontsize=label_size, verticalalignment='bottom', horizontalalignment='left')
                
                plt.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.10)
                fig_pi_i_name = f'{star_name}_{obsmod}_{fltr_arr[z]}_{z}_PI_over_I'
                plt.savefig(os.path.join(outdir, fig_pi_i_name + '.png'), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(outdir, fig_pi_i_name + '.pdf'), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(outdir_dolp_only, fig_pi_i_name + '.png'), dpi=300, bbox_inches='tight')
                print(f"✓ Figure PI/I sauvegardée pour {star_name}: {fig_pi_i_name}")
                plt.close(fig_pi_over_i)
                
                # === Enregistrement des profils radiaux (DoLP et PI/I) de 0 à 200 mas ===
                # Calculer les profils radiaux
                radial_dolp = radial_profile(sub_v_dolp_display)
                radial_pi_i = radial_profile(pi_over_i_display)
                
                # Normaliser les profils radiaux (division par la valeur maximale)
                max_dolp = np.nanmax(radial_dolp)
                max_pi_i = np.nanmax(radial_pi_i)
                radial_dolp_norm = radial_dolp / max_dolp if max_dolp > 0 else radial_dolp
                radial_pi_i_norm = radial_pi_i / max_pi_i if max_pi_i > 0 else radial_pi_i
                
                # Convertir les pixels en mas (0 à 200 mas)
                min_radius_mas = 0.0
                max_radius_mas = 200.0
                min_radius_pix = int(min_radius_mas / pix2mas)
                max_radius_pix = int(max_radius_mas / pix2mas)
                
                # Créer les fichiers CSV pour les profils radiaux
                profil_dolp_csv = os.path.join(outdir, f'{star_name}_{obsmod}_{fltr_arr[z]}_{z}_DoLP_radial_profile.csv')
                profil_pi_i_csv = os.path.join(outdir, f'{star_name}_{obsmod}_{fltr_arr[z]}_{z}_PI_over_I_radial_profile.csv')
                
                # Écrire le profil DoLP normalisé
                with open(profil_dolp_csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Radius_pix', 'Radius_mas', 'DoLP_Value_Normalized'])
                    for pix in range(min_radius_pix, min(max_radius_pix, len(radial_dolp_norm))):
                        radius_mas = pix * pix2mas
                        if min_radius_mas <= radius_mas <= max_radius_mas:
                            writer.writerow([pix, f'{radius_mas:.3f}', f'{radial_dolp_norm[pix]:.6f}'])
                
                # Écrire le profil PI/I normalisé
                with open(profil_pi_i_csv, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Radius_pix', 'Radius_mas', 'PI_over_I_Value_Normalized'])
                    for pix in range(min_radius_pix, min(max_radius_pix, len(radial_pi_i_norm))):
                        radius_mas = pix * pix2mas
                        if min_radius_mas <= radius_mas <= max_radius_mas:
                            writer.writerow([pix, f'{radius_mas:.3f}', f'{radial_pi_i_norm[pix]:.6f}'])
                
                print(f"✓ Profils radiaux (normalisés) sauvegardés pour {star_name}:")
                print(f"  - DoLP: {profil_dolp_csv}")
                print(f"  - PI/I: {profil_pi_i_csv}")
                
                # === Création d'une figure 2x2 avec les images et profils radiaux ===
                fig_combined = plt.figure(figsize=(14, 10))
                gs = fig_combined.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
                
                # Haut-gauche : Image DoLP
                ax_img_dolp = fig_combined.add_subplot(gs[0, 0])
                im_dolp_img = ax_img_dolp.imshow(sub_v_dolp_display, cmap='plasma', origin='lower', 
                                                 vmin=0, vmax=np.nanpercentile(sub_v_dolp_display, 95), 
                                                 extent=extent_dolp, interpolation='bilinear')
                # ax_img_dolp.set_xlabel('RA (mas)', fontsize=label_size)
                # ax_img_dolp.set_ylabel('Dec (mas)', fontsize=label_size)
                # ax_img_dolp.tick_params(labelsize=label_size)
                ax_img_dolp.set_xticks([])
                ax_img_dolp.set_yticks([])
                ax_img_dolp.tick_params(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)
                txt1 = ax_img_dolp.text(0.02, 0.95, f'{star_name2}', transform=ax_img_dolp.transAxes, fontsize=label_size, color='white', va='top')
                txt1.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
                txt2 = ax_img_dolp.text(0.02, 0.02, f'DoLP', transform=ax_img_dolp.transAxes, fontsize=label_size, color='white', va='bottom')
                txt2.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
                txt3 = ax_img_dolp.text(0.98, 0.02, f'{fltr_arr[z]}', transform=ax_img_dolp.transAxes, fontsize=label_size, color='white', va='bottom', ha='right')
                txt3.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
                divider_dolp_img = make_axes_locatable(ax_img_dolp)
                cax_dolp_img = divider_dolp_img.append_axes('right', size='5%', pad=0.05)
                cbar_dolp_img = fig_combined.colorbar(im_dolp_img, cax=cax_dolp_img)
                cbar_dolp_img.ax.tick_params(labelsize=label_size)
                # Notation scientifique pour colorbar DoLP
                vmin_dolp_img, vmax_dolp_img = cbar_dolp_img.vmin, cbar_dolp_img.vmax
                if vmax_dolp_img != 0:
                    exp_dolp_img = int(np.floor(np.log10(np.abs(vmax_dolp_img))))
                else:
                    exp_dolp_img = 0
                def fmt_dolp_img(x, pos):
                    if x == 0:
                        return ''
                    return f'{x / (10 ** exp_dolp_img):.1f}'
                cbar_dolp_img.ax.yaxis.set_major_formatter(FuncFormatter(fmt_dolp_img))
                if label_size_great_panel != label_size_small_panel:
                    exp_x = -2.4 + (label_size - label_size_small_panel) * (-4.0 + 2.4) / (label_size_great_panel - label_size_small_panel)
                else:
                    exp_x = -2.4
                cbar_dolp_img.ax.yaxis.get_offset_text().set_visible(False)  # Désactiver offset auto
                cbar_dolp_img.ax.text(exp_x, 1.001, f'×1e{exp_dolp_img}', transform=cbar_dolp_img.ax.transAxes, 
                                     fontsize=label_size, va='bottom', ha='left', clip_on=False)
                
                # Haut-droite : Image PI/I
                ax_img_pi_i = fig_combined.add_subplot(gs[0, 1])
                im_pi_i_img = ax_img_pi_i.imshow(pi_over_i_display, cmap='plasma', origin='lower', 
                                                vmin=0, vmax=np.nanpercentile(pi_over_i_display, 95), 
                                                extent=extent_dolp, interpolation='bilinear')
                # ax_img_pi_i.set_xlabel('RA (mas)', fontsize=label_size)
                # ax_img_pi_i.tick_params(labelsize=label_size)
                ax_img_pi_i.set_xticks([])
                ax_img_pi_i.set_yticks([])
                ax_img_pi_i.tick_params(left=False, right=False, bottom=False, top=False, labelleft=False, labelbottom=False)
                txt4 = ax_img_pi_i.text(0.02, 0.95, f'{star_name2}', transform=ax_img_pi_i.transAxes, fontsize=label_size, color='white', va='top')
                txt4.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
                txt5 = ax_img_pi_i.text(0.02, 0.02, f'PI/I', transform=ax_img_pi_i.transAxes, fontsize=label_size, color='white', va='bottom')
                txt5.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
                txt6 = ax_img_pi_i.text(0.98, 0.02, f'{fltr_arr[z]}', transform=ax_img_pi_i.transAxes, fontsize=label_size, color='white', va='bottom', ha='right')
                txt6.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
                divider_pi_i_img = make_axes_locatable(ax_img_pi_i)
                cax_pi_i_img = divider_pi_i_img.append_axes('right', size='5%', pad=0.05)
                cbar_pi_i_img = fig_combined.colorbar(im_pi_i_img, cax=cax_pi_i_img)
                cbar_pi_i_img.ax.tick_params(labelsize=label_size)
                # Notation scientifique pour colorbar PI/I
                vmin_pi_i_img, vmax_pi_i_img = cbar_pi_i_img.vmin, cbar_pi_i_img.vmax
                if vmax_pi_i_img != 0:
                    exp_pi_i_img = int(np.floor(np.log10(np.abs(vmax_pi_i_img))))
                else:
                    exp_pi_i_img = 0
                def fmt_pi_i_img(x, pos):
                    if x == 0:
                        return ''
                    return f'{x / (10 ** exp_pi_i_img):.1f}'
                cbar_pi_i_img.ax.yaxis.set_major_formatter(FuncFormatter(fmt_pi_i_img))
                cbar_pi_i_img.ax.yaxis.get_offset_text().set_visible(False)  # Désactiver offset auto
                cbar_pi_i_img.ax.text(-2.4, 1.001, f'×1e{exp_pi_i_img}', transform=cbar_pi_i_img.ax.transAxes, 
                                     fontsize=label_size, va='bottom', ha='left')
                
                # Axes x en mas pour les profils
                radius_pix_range = np.arange(min_radius_pix, min(max_radius_pix, len(radial_dolp_norm)))
                radius_mas_range = radius_pix_range * pix2mas
                
                # Bas-gauche : Profil DoLP normalisé
                ax_radial_dolp = fig_combined.add_subplot(gs[1, 0])
                # Ajouter le point (0, 1.0) au début
                radius_mas_with_center = np.concatenate([[0], radius_mas_range])
                radial_dolp_with_center = np.concatenate([[1.0], radial_dolp_norm[radius_pix_range]])
                ax_radial_dolp.plot(radius_mas_with_center, radial_dolp_with_center, 'o-', color='white', linewidth=2.5, markersize=5, label='DoLP (normalized)')
                # ax_radial_dolp.set_xlabel('Radius (mas)', fontsize=label_size)
                # ax_radial_dolp.set_ylabel('Normalized DoLP', fontsize=label_size)
                #ax_radial_dolp.set_title('DoLP Radial Profile', fontsize=label_size)
                ax_radial_dolp.set_xlim([0, 200])
                ax_radial_dolp.set_ylim([0, 1.1])
                ax_radial_dolp.grid(True, alpha=0.3)
                # ax_radial_dolp.tick_params(labelsize=label_size)
                ax_radial_dolp.legend(loc='best', fontsize=label_size)
                
                # Bas-droite : Profil PI/I normalisé
                ax_radial_pi_i = fig_combined.add_subplot(gs[1, 1])
                radial_pi_i_with_center = np.concatenate([[1.0], radial_pi_i_norm[radius_pix_range]])
                ax_radial_pi_i.plot(radius_mas_with_center, radial_pi_i_with_center, 'o-', color='white', linewidth=2.5, markersize=5, label='PI/I (normalized)')
                # ax_radial_pi_i.set_xlabel('Radius (mas)', fontsize=label_size)
                # ax_radial_pi_i.set_ylabel('Normalized PI/I', fontsize=label_size, loc='center')
                ax_radial_pi_i.yaxis.set_label_position('right')
                #ax_radial_pi_i.set_title('PI/I Radial Profile', fontsize=label_size)
                ax_radial_pi_i.set_xlim([0, 200])
                ax_radial_pi_i.set_ylim([0, 1.1])
                ax_radial_pi_i.grid(True, alpha=0.3)
                # ax_radial_pi_i.tick_params(labelsize=label_size, right=True, left=False, labelright=True, labelleft=False)
                # ax_radial_pi_i.yaxis.tick_right()
                ax_radial_pi_i.legend(loc='best', fontsize=label_size)
                
                fig_combined_name = f'{star_name}_{obsmod}_{fltr_arr[z]}_{z}_DoLP_vs_PI_over_I'
                fig_combined.savefig(os.path.join(outdir, fig_combined_name + '.png'), dpi=300, bbox_inches='tight')
                fig_combined.savefig(os.path.join(outdir, fig_combined_name + '.pdf'), dpi=300, bbox_inches='tight')
                fig_combined.savefig(os.path.join(outdir_dolp_only, fig_combined_name + '.png'), dpi=300, bbox_inches='tight')
                print(f"✓ Figure combinée (images + profils) sauvegardée: {fig_combined_name}")
                plt.close(fig_combined)


            # Figure PI seule avec contours et ellipse ajustée pour analyse des PA
            plt.figure(figsize=(6, 5))
            ax = plt.gca()
            custom_pi_contours = {
                'V854_Cen': [1.23, 1.275, 1.30, 1.59, 1.88, 2.00, 2.05],
                }
            im = ax.imshow(np.log10(sub_v + np.abs(np.min(sub_v)) + 10), cmap='plasma', origin='lower', extent=extent_small)
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

            # Barre d'échelle en R* (identique au panel)
            if distance_star_pc is not None and radius_star_rsun is not None and radius_star_rsun > 0 and distance_star_pc > 0:
                scale_val = custom_scale_bar.get(star_name, 10)  # 10 R* par défaut
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
                ax.plot([x_bar_axes, x_bar_axes + bar_length_axes], [y_bar_axes, y_bar_axes], color='white', lw=3, transform=ax.transAxes, solid_capstyle='butt')
                ax.text(x_bar_axes + bar_length_axes/2, y_bar_axes + 0.01, f'{scale_val:.0f}{legend_unit}', color='white', fontsize=label_size, ha='center', va='bottom',  transform=ax.transAxes, path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
            else:
                # Barre de secours en mas si pas de distance/rayon
                try:
                    fallback_scale_mas = (nSubDim_DOLP / 10.0) * pix2mas
                    bar_length_axes = fallback_scale_mas / (x_max_DOLP - x_min_DOLP)
                    x_bar_axes = 0.90 - bar_length_axes
                    y_bar_axes = 0.04
                    ax.plot([x_bar_axes, x_bar_axes + bar_length_axes], [y_bar_axes, y_bar_axes], color='white', lw=3, transform=ax.transAxes, solid_capstyle='butt')
                    ax.text(x_bar_axes + bar_length_axes/2, y_bar_axes + 0.01, f'{int(fallback_scale_mas)} mas', color='white', fontsize=label_size, ha='center', va='bottom', transform=ax.transAxes, path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
                except Exception:
                    pass
            # Flèches N-W en haut à droite (identique au panel)
            arrow_len = 0.04 * (x_max_DOLP - x_min_DOLP)
            x_arrow = x_max_DOLP - 0.005 * (x_max_DOLP - x_min_DOLP)# vers la gauche
            y_arrow = y_max_DOLP - 0.05 * (y_max_DOLP - y_min_DOLP)# vers le bas
            ax.arrow(x_arrow, y_arrow, 0, arrow_len, head_width=0.02*arrow_len, head_length=0.04*arrow_len, fc='white', ec='white', lw=2)
            ax.arrow(x_arrow, y_arrow, -arrow_len, 0, head_width=0.02*arrow_len, head_length=0.04*arrow_len, fc='white', ec='white', lw=2)
            offset_label = 15
            ax.text(x_arrow, y_arrow + arrow_len + offset_label, 'N', color='white', fontsize=label_size, ha='center', va='bottom', path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
            ax.text(x_arrow - arrow_len - offset_label, y_arrow, 'E', color='white', fontsize=label_size, ha='right', va='center', path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
            # appliquer le signe aux coordonnées X tracées
            x_contour_mas = x_contour_mas * x_sign
            x_centroid_mas = x_centroid_mas * x_sign
            #ax.plot(x_contour_mas, y_contour_mas, color='cyan', linewidth=2, linestyle='--')
            ax.scatter([x_centroid_mas], [y_centroid_mas], color='red', marker='x')
            ax.set_xlabel("Relative RA (mas)", fontsize=label_size)
            ax.set_ylabel("Relative Dec (mas)", fontsize=label_size)
            ax.tick_params(axis='both', labelsize=label_size, width=1.2)
            ax.locator_params(axis='x', nbins=5)
            ax.locator_params(axis='y', nbins=5)
            ax.text(0.02, 0.95, f'{star_name2}', transform=ax.transAxes, fontsize=label_size, color='white', va='top', path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
            ax.text(0.02, 0.02, f'{fltr_arr[z]}', transform=ax.transAxes, fontsize=label_size, color='white', va='bottom', path_effects=[PathEffects.withStroke(linewidth=2, foreground='black')])
            try:
                pi_offset = np.abs(np.nanmin(sub_v)) + 10.0
                log_PI = np.log10(np.clip(sub_v + pi_offset, a_min=1e-12, a_max=None))
                # niveaux personnalisés possibles via dictionnaire custom_pi_contours (fallback : 5 niveaux)
                pi_min = np.nanmin(log_PI)
                pi_max = np.nanmax(log_PI)
                pi_levels = custom_pi_contours.get(star_name, np.linspace(pi_min, pi_max, 5))
                cs_pi = ax.contour(log_PI, levels=pi_levels, colors='white', linewidths=1.2, origin='lower', extent=[x_min+1, x_max, y_min+1, y_max])
                ax.clabel(cs_pi, inline=True, fontsize=10, fmt='%.2f', colors='white')
                # --- Ajuster une ellipse sur le contour le plus grand (si présent) ---
                                # --- Fit ALL contours (tous les chemins) et collecte des paramètres ---
                
                cmap = plt.get_cmap('tab20')  # palette distincte pour chaque contour
                ncols_cmap = cmap.N
                ellipse_fits = []
                dx = 0.03 * (x_max - x_min)
                dy = 0.03 * (y_max - y_min)
                idx = 0
                for coll in cs_pi.collections:
                    for p in coll.get_paths():
                        verts = p.vertices  # (N,2) en unités de l'extent (mas)
                        if verts.shape[0] <= 6:
                            continue
                        try:
                            model = EllipseModel()
                            ok = model.estimate(verts)
                            if ok:
                                xc, yc, a_e, b_e, theta_e = model.params
                                method = 'EllipseModel'
                            else:
                                # fallback PCA
                                cen = verts.mean(axis=0)
                                cov = np.cov(verts.T)
                                w, v = np.linalg.eigh(cov)
                                order = w.argsort()[::-1]
                                w = w[order]; v = v[:, order]
                                a_e = 2.0 * np.sqrt(w[0])
                                b_e = 2.0 * np.sqrt(w[1])
                                theta_e = np.arctan2(v[1, 0], v[0, 0])
                                xc, yc = cen[0], cen[1]
                                method = 'PCA_fallback'

                            # choisir une couleur distincte pour ce contour
                            color_rgba = cmap(idx % ncols_cmap)
                            color_hex = to_hex(color_rgba)
                            linestyle = '-.' if method == 'EllipseModel' else '--'

                            # tracer une ellipse ajustée à chaque contour
                            t_ell = np.linspace(0, 2*np.pi, 300)
                            Rmat = np.array([[np.cos(theta_e), -np.sin(theta_e)], [np.sin(theta_e), np.cos(theta_e)]])
                            ellipse = np.vstack((a_e*np.cos(t_ell), b_e*np.sin(t_ell)))
                            pts_ell = (Rmat @ ellipse) + np.array([[xc],[yc]])
                            ax.plot(pts_ell[0, :], pts_ell[1, :], color=color_hex, linewidth=1.6, linestyle=linestyle, alpha=0.95)
                            

                            # Annotation empilée en haut‑droite : même abscisse (x_text), ordonnées différentes
                            pa_deg = float((np.degrees(np.pi/2.0 + theta_e)) % 180.0)
                            # Compute PA uncertainty for this contour ---
                            pa_med, pa_sigma, pa_lo, pa_hi = pa_uncertainty_from_verts(verts, nboot=1000, noise_sigma_px=0.5)
                            # if function failed (nan), fallback to empirical small sigma
                            if np.isnan(pa_sigma):
                                pa_sigma = 1.0
                            
                            
                            txt = f'{idx}: a={a_e:.1f}, b={b_e:.1f}, PA={pa_deg:.1f}°'
                            # position fixe en abscisse (marge relative à droite) et ordonnée empilée vers le bas
                            x_margin = 0.02 * (x_max - x_min)
                            x_text = x_max - x_margin
                            y_top = y_max - 0.02 * (y_max - y_min)
                            line_spacing = 0.035 * (y_max - y_min)  # espacement vertical entre labels
                            max_labels = 200  # sécurité pour éviter débordement
                            # if idx < max_labels:
                            #     y_text = y_top - idx * line_spacing
                            #     ax.text(x_text, y_text, txt, color=color_hex, fontsize=8, va='top', ha='right', zorder=13,
                            #             bbox=dict(facecolor='black', alpha=0.40, edgecolor='none', pad=1))
                            # else:
                            #     # si trop de labels, on s'arrête d'en dessiner (ou on pourrait basculer sur un fichier)
                            #     pass


                            # stocker les paramètres
                            ellipse_fits.append({
                                'contour_idx': idx,
                                'method': method,
                                'xc_mas': float(xc),
                                'yc_mas': float(yc),
                                'a_mas': float(a_e),
                                'b_mas': float(b_e),
                                'theta_rad': float(theta_e),
                                'theta_deg': pa_deg,
                                'pa_sigma_deg': float(pa_sigma),
                                'pa_lo68_deg': float(pa_lo),
                                'pa_hi68_deg': float(pa_hi),
                                'n_points': int(verts.shape[0])
                            })
                            
                            rows = []
                            for lvl, ef in zip(levels_expanded, ellipse_fits):
                                rows.append({
                                    'star_name': star_name,
                                    'Filtre': fltr_arr[z],
                                    'pi_level': float(lvl) if lvl is not None else np.nan,
                                    'contour_idx': ef.get('contour_idx'),
                                    'theta_deg': pa_deg,
                                    'z': int(z)
                                })
                            idx += 1
                            
                        except Exception as e:
                            print(f"⚠️ Fit contour échoué (pts={verts.shape[0]}): {e}", flush=True)
                        df_theta = pd.DataFrame(rows)
                        theta_csv = os.path.join(outdir, f'{star_name}_{fltr_arr[z]}_{z}_theta_vs_level.csv')
                        df_theta.to_csv(theta_csv, index=False)
                        print(f"✅ theta vs level saved: {theta_csv}", flush=True)
                        # sauvegarde table des fits (niveau, PA, xc,yc,a,b,color)
                                # --- après la boucle qui a rempli `ellipse_fits` ---
                try:
                    # construire la liste étendue des niveaux (une entrée par chemin de contour)
                    if hasattr(cs_pi, 'levels') and len(cs_pi.levels) > 0 and len(ellipse_fits) > 0:
                        levels_expanded = []
                        for i, coll in enumerate(cs_pi.collections):
                            npaths = len(coll.get_paths())
                            lvl = cs_pi.levels[i] if i < len(cs_pi.levels) else None
                            levels_expanded += [lvl] * npaths
                        # tronquer ou compléter pour correspondre à len(ellipse_fits)
                        if len(levels_expanded) < len(ellipse_fits):
                            levels_expanded += [None] * (len(ellipse_fits) - len(levels_expanded))
                        levels_expanded = levels_expanded[:len(ellipse_fits)]

                        # récupérer valeurs PA et indices
                        pa_vals = [ef.get('theta_deg', float(np.degrees(ef.get('theta_rad', 0.0))) % 180.0) for ef in ellipse_fits]
                        contour_idx = [ef.get('contour_idx', i) for i, ef in enumerate(ellipse_fits)]

                        # plot scatter PA vs level (inchangé)
                        fig_pa, ax_pa = plt.subplots(figsize=(6, 5))
                        sc = ax_pa.scatter(levels_expanded, pa_vals, c=np.arange(len(pa_vals)), cmap='tab20', s=60, edgecolor='k')
                        ax_pa.set_xlabel('log10(PI)', fontsize=label_size)
                        ax_pa.set_ylabel('PA (deg)', fontsize=label_size)
                        ax_pa.set_ylim(0, 120)
                        ax_pa.tick_params(axis='both', which='major', labelsize=label_size, width=1.2)
                        plt.tight_layout()
                        fig_pa_name = f'{star_name}_{obsmod}_{fltr_arr[z]}_{z}_PA_vs_PIlevel'
                        fig_pa.savefig(os.path.join(outdir, fig_pa_name + '.png'), dpi=300, bbox_inches='tight')
                        fig_pa.savefig(os.path.join(outdir, fig_pa_name + '.pdf'), dpi=300, bbox_inches='tight')
                        if star_name == 'V854_Cen':
                            fig_pa.savefig(os.path.join(outdir, fig_pa_name + '.eps'), format='eps', dpi=300, bbox_inches='tight')
                        plt.close(fig_pa)

                        # Construire et sauvegarder la table theta_vs_level à partir d'ellipse_fits
                        rows = []
                        for lvl, ef in zip(levels_expanded, ellipse_fits):
                            rows.append({
                                'star_name': star_name,
                                'Filtre': fltr_arr[z],
                                'pi_level': float(lvl) if (lvl is not None and lvl != '') else np.nan,
                                'contour_idx': ef.get('contour_idx'),
                                'theta_deg': ef.get('theta_deg'),
                                'pa_sigma_deg': ef.get('pa_sigma_deg', np.nan),
                                'pa_lo68_deg': ef.get('pa_lo68_deg', np.nan),
                                'pa_hi68_deg': ef.get('pa_hi68_deg', np.nan),
                                'xc_mas': ef.get('xc_mas'),
                                'yc_mas': ef.get('yc_mas'),
                                'a_mas': ef.get('a_mas'),
                                'b_mas': ef.get('b_mas'),
                                'method': ef.get('method'),
                                'z': int(z)
                            })
                        df_theta = pd.DataFrame(rows)
                        theta_csv = os.path.join(outdir, f'{star_name}_{fltr_arr[z]}_{z}_theta_vs_level.csv')
                        df_theta.to_csv(theta_csv, index=False)
                        print(f"✅ theta vs level saved: {theta_csv}", flush=True)

                        # sauvegarde table des fits (niveau, PA, xc,yc,a,b, method, color, et incertitudes)
                        try:
                            import csv
                            csv_pa = os.path.join(outdir, f'{star_name}_{fltr_arr[z]}_{z}_PA_vs_PIlevel.csv')
                            # diagnostic rapide
                            print(f"Writing PA CSV to {csv_pa} (n_fits={len(ellipse_fits)})", flush=True)
                            if len(ellipse_fits) > 0:
                                print("Sample fit keys:", list(ellipse_fits[0].keys()), flush=True)

                            with open(csv_pa, 'w', newline='') as cf:
                                writer = csv.writer(cf)
                                # header forcé : utilise theta_deg comme fallback pour pa_deg_0_180
                                header = ['contour_idx', 'pi_level', 'pa_deg_0_180', 'pa_sigma_deg', 'pa_lo68_deg', 'pa_hi68_deg',
                                        'xc_mas', 'yc_mas', 'a_mas', 'b_mas', 'method', 'color', 'z']
                                writer.writerow(header)
                                for lvl, ef in zip(levels_expanded, ellipse_fits):
                                    pa_val = ef.get('pa_deg_0_180', ef.get('theta_deg', ''))
                                    pa_sigma = ef.get('pa_sigma_deg', '')
                                    pa_lo = ef.get('pa_lo68_deg', '')
                                    pa_hi = ef.get('pa_hi68_deg', '')
                                    row = [
                                        ef.get('contour_idx', ''),
                                        float(lvl) if (lvl is not None and lvl != '') else '',
                                        pa_val,
                                        pa_sigma,
                                        pa_lo,
                                        pa_hi,
                                        ef.get('xc_mas', ''),
                                        ef.get('yc_mas', ''),
                                        ef.get('a_mas', ''),
                                        ef.get('b_mas', ''),
                                        ef.get('method', ''),
                                        ef.get('color', ''),
                                        int(z)
                                    ]
                                    writer.writerow(row)
                            print(f"✅ PA_vs_PIlevel CSV saved: {csv_pa}", flush=True)
                        except Exception as e_csv:
                            print(f"⚠️ Échec sauvegarde CSV PA_vs_PIlevel : {e_csv}", flush=True)

                    else:
                        print("⚠️ Aucun fit/contour valide pour tracer PA vs PI level.", flush=True)
                except Exception as e_pa:
                    print(f"⚠️ Erreur lors du tracé PA vs PI level : {e_pa}", flush=True)

            except Exception as e:
                print(f"⚠️ Impossible d'ajouter les isocontours PI pour {star_name}: {e}", flush=True)


            

            plt.tight_layout()
            fig_name = f'{star_name}_{obsmod}_{fltr_arr[z]}_{z}_unique_max_contour_for_Pol_Intensity'
            plt.savefig(os.path.join(outdir, fig_name + '.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(outdir, fig_name + '.pdf'), dpi=300, bbox_inches='tight')
            if star_name=='V854_Cen':
                plt.savefig(os.path.join(outdir, fig_name + '.eps'), dpi=300, bbox_inches='tight') 
            plt.savefig(os.path.join(outdir_uniq, fig_name + '.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(outdir_uniq, fig_name + '.pdf'), dpi=300, bbox_inches='tight')
            if star_name=='V854_Cen':
                plt.savefig(os.path.join(outdir_uniq, fig_name + '.eps'), dpi=300, bbox_inches='tight')
            print(f"Figure contour sauvegardée : {os.path.join(outdir_uniq, fig_name + '.png')}", flush=True)
            #plt.savefig(os.path.join(outdir, fig_name + '.eps'), format='eps', dpi=300, bbox_inches='tight')
            #plt.show() 
            plt.close()

            
            # Sauvegarde conditionnelle dans le dossier spécifique DoLP (une seule image par filtre/étoile, et seulement si clairement résolue)            
            dolp_specific_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Morphologies_contours/DoLP_specific'
            fig_path_pdf = os.path.join(dolp_specific_dir+'/pdf')
            fig_path_png = os.path.join(dolp_specific_dir+'/png')
            fig_dolp.savefig(fig_path_pdf, dpi=300, bbox_inches='tight')
            fig_dolp.savefig(fig_path_png, dpi=300, bbox_inches='tight')
            print(f"Figure DoLP spécifique sauvegardée : {fig_path_pdf} et {fig_path_png}", flush=True)
            os.makedirs(dolp_specific_dir, exist_ok=True)
            if fltr_arr[z] == star_specific_filter.get(star_name, None):
                # On ne sauvegarde que si aucune image pour ce filtre n'existe déjà
                already_exists = any(str(fltr_arr[z]) in fname and star_name in fname for fname in os.listdir(fig_path_pdf))
                if not already_exists and star_name in clearly_resolved:
                    fig_path1 = os.path.join(fig_path_pdf, fig_dolp_name + '.pdf')
                    plt.savefig(fig_path1, dpi=300, bbox_inches='tight')
                    print(f"Figure panel DoLP seul sauvegardée : {fig_path1}", flush=True)    
                else:
                    print(f"Panel déjà présent pour {star_name} [{fltr_arr[z]}], non sauvegardé.", flush=True)
                already_exists = any(str(fltr_arr[z]) in fname and star_name in fname for fname in os.listdir(fig_path_png))
                if not already_exists and star_name in clearly_resolved:
                    fig_path2 = os.path.join(fig_path_png, fig_dolp_name + '.png')
                    plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
                    print(f"Figure panel DoLP seul sauvegardée : {fig_path2}", flush=True)    
                else:
                    print(f"Panel déjà présent pour {star_name} [{fltr_arr[z]}], non sauvegardé.", flush=True)
                if fig_panel is not None:
                    plt.close(fig_panel)
            else:
                if fig_panel is not None:
                    plt.close(fig_panel)
                print(f"Panel ignoré pour {star_name} [{fltr_arr[z]}] : filtre non spécifique.", flush=True)

            plt.close(fig_dolp)
        else:
            if fig_panel is not None:
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
            if star_name=='V854_Cen':
                plt.savefig(os.path.join(outdir, fig_name + '.eps'), dpi=300, bbox_inches='tight')
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
    if structure is not None and star_name not in structure:
        continue   #filtre uniquement les étoiles bipolaires
    star_path = os.path.join(large_log_dir, star_name)
    if not os.path.isdir(star_path):
        continue
    if star_name in star_filters:
        mode, filters = star_filters[star_name]
        print(f"🌟 Traitement de {star_name} | Mode : {mode} | Filtres : {filters}")
        star_results = log_image(fold_name, star_name, mode, star_specific_filter=star_specific_filter)
        if star_results:
            print(f"   ✅ {len(star_results)} résultats obtenus pour {star_name}")
            all_results.extend(star_results)  # Ajoute les résultats de cette étoile à la liste globale
        else:
            print(f"   ⚠️ Aucun résultat pour {star_name}")
    else:
        print(f"❌ {star_name} n'est pas dans le dictionnaire star_filters, ignoré.")

print(f"\n📊 RÉSUMÉ FINAL :")
print(f"📈 Total de résultats collectés : {len(all_results)}")

# === Compilation et affichage de la liste globale des rayons du grand cercle ===
radii_global = []
seen_stars = set()
for star_name in os.listdir(large_log_dir):
    star_path = os.path.join(large_log_dir, star_name)
    outdir = os.path.join(star_path, 'plots', 'fits', 'log_scale', 'fully_automatic')
    radii_csv = os.path.join(outdir, 'radii_grand_cercle.csv')
    if os.path.exists(radii_csv):
        import csv
        with open(radii_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # récupérer nom étoile et normaliser
                etoile = (row.get('Étoile') or row.get('Star') or '').strip()
                if not etoile:
                    # fallback : prendre premier champ si headers inattendus
                    vals = [v for v in row.values()]
                    if len(vals) >= 1:
                        etoile = str(vals[0]).strip()
                if not etoile or etoile in seen_stars:
                    continue
                # tenter lecture champs complets (5-col) sinon fallback 3-col
                petit_rstar = (row.get('Rayon_petit_cercle_Rstar') or '').strip()
                petit_mas = (row.get('Rayon_petit_cercle_mas') or '').strip()
                grand_rstar = (row.get('Rayon_grand_cercle_Rstar') or '').strip()
                grand_mas = (row.get('Rayon_grand_cercle_mas') or '').strip()
                # si les clés manquent (format 3-col), extraire depuis les valeurs
                if not grand_rstar and not grand_mas:
                    vals = [v for v in row.values()]
                    if len(vals) >= 3:
                        # on suppose: [Étoile, grand_rstar, grand_mas]
                        grand_rstar = str(vals[1]).strip()
                        grand_mas = str(vals[2]).strip()
                radii_global.append((etoile, petit_rstar or None, petit_mas or None, grand_rstar or None, grand_mas or None))
                seen_stars.add(etoile)

if radii_global:
    print("\nListe des rayons des cercles pour toutes les étoiles :")
    for etoile, petit_rstar, petit_mas, grand_rstar, grand_mas in radii_global:
        if petit_rstar or petit_mas:
            print(f"  - {etoile} : Petit cercle = {petit_rstar} R* | {petit_mas} mas ; Grand cercle = {grand_rstar} R* | {grand_mas} mas")
        else:
            print(f"  - {etoile} : Grand cercle = {grand_rstar} R* | {grand_mas} mas (pas de petit cercle)")

    global_radii_csv = '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_tables/morpho/radii_cercles_all_stars.csv'
    import csv
    with open(global_radii_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Étoile', 'Rayon_petit_cercle_Rstar', 'Rayon_petit_cercle_mas', 'Rayon_grand_cercle_Rstar', 'Rayon_grand_cercle_mas'])
        for row in radii_global:
            if len(row) == 5:
                writer.writerow(row)
            elif len(row) == 3:
                etoile, grand_rstar, grand_mas = row
                writer.writerow([etoile, '', '', grand_rstar, grand_mas])
    print(f"\n✅ Liste globale sauvegardée dans : {global_radii_csv}")

# Sauvegarde de la table globale
if all_results:
    global_df = pd.DataFrame(all_results)

    # Une seule ligne par étoile : garder le D_maj_mas maximum
    if {'Étoile', 'D_maj_mas'}.issubset(global_df.columns):
        # s'assurer que D_maj_mas est bien numérique
        global_df['D_maj_mas'] = pd.to_numeric(global_df['D_maj_mas'], errors='coerce')
        idx_max = global_df.groupby('Étoile')['D_maj_mas'].idxmax()
        global_df = global_df.loc[idx_max].sort_values('Étoile').reset_index(drop=True)
    
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
    # --- Affichage compact de la liste des PA trouvés (uniques) et par étoile ---
        # --- Affichage compact de la liste des PA trouvés (uniques) et par étoile ---
    try:
        pa_ser = pd.to_numeric(global_df.get('theta_deg', pd.Series()), errors='coerce').dropna()
        unique_pa = sorted(pa_ser.round(1).unique().tolist())
        print("\nListe des PA trouvés (uniques, deg):", unique_pa)

        print("\nPA par étoile :")
        from collections import defaultdict
        for star, grp in global_df.groupby('Étoile'):
            pa_map = defaultdict(set)
            for _, row in grp.iterrows():
                try:
                    pa = round(float(row.get('theta_deg', np.nan)), 1)
                except Exception:
                    continue
                if np.isnan(pa):
                    continue
                filt = str(row.get('Filtre', '')).strip()
                if filt == 'nan' or filt == '':
                    filt = '(no-filter)'
                pa_map[pa].add(filt)
            if pa_map:
                entries = [f"{pa}° [{', '.join(sorted(list(fset)))}]" for pa, fset in sorted(pa_map.items())]
                print(f" - {star}: {entries}")
    except Exception as e:
        print(f"Erreur lors de l'affichage des PA : {e}")
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
# Création du DataFrame et filtrage sécurisé
results_df = pd.DataFrame(results)
if not results_df.empty and 'Étoile' in results_df.columns:
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

    hist_source_df = filtered_results_df
    if fold_name == 'large_log_+':
        # Ne tracer que les étoiles résolues dans le cas du dossier large_log_+
        hist_source_df = filtered_results_df[filtered_results_df['Étoile'].isin(clearly_resolved)].copy() 

    plt.figure(figsize=(7, 5))
    hist_vals, bins, patches = plt.hist(
        hist_source_df['e'],
        bins=20,
        color='#4A90E2',
        edgecolor='#bbbbbb',
        linewidth=1.1,
        alpha=0.85,
        rwidth=0.92
    )
    for patch in patches:
        patch.set_linewidth(1.1)
        patch.set_edgecolor('#bbbbbb')
    median_e = hist_source_df['e'].median()
    plt.axvline(median_e, color='#D0021B', linestyle='--', linewidth=2, label=f'Median = {median_e:.2f}')
    plt.xlabel(r'$\rm \varepsilon$', fontsize=14)
    plt.ylabel("Nombre d'étoiles", fontsize=14)
    plt.legend(fontsize=14, loc='upper right')
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(all_tables_dir + '/histogramme_ellipticite_max_diam.png', dpi=300, bbox_inches='tight')
    plt.savefig(all_tables_dir + '/histogramme_ellipticite_max_diam.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print("✅ Histogramme des ellipticités stylé sauvegardé sous 'histogramme_ellipticite_max_diam.png' et '.pdf'")

    # Histogramme des tailles physiques (D_maj_UA)
    plt.figure(figsize=(7, 5))
    hist_vals_phys, bins_phys, patches_phys = plt.hist(
        hist_source_df['D_maj_UA'].dropna(),
        bins=20,
        color='#50B878',
        edgecolor='#bbbbbb',
        linewidth=1.1,
        alpha=0.85,
        rwidth=0.92
    )
    for patch in patches_phys:
        patch.set_linewidth(1.1)
        patch.set_edgecolor('#bbbbbb')
    median_phys = hist_source_df['D_maj_UA'].median()
    plt.axvline(median_phys, color='#D0021B', linestyle='--', linewidth=2, label=f'Median = {median_phys:.1f} AU')
    plt.xlabel('$D_{maj}$ (AU)', fontsize=14)
    plt.ylabel("Nombre d'étoiles", fontsize=14)
    plt.legend(fontsize=14, loc='upper right')
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(all_tables_dir + '/histogramme_taille_physique_max_diam.png', dpi=300, bbox_inches='tight')
    plt.savefig(all_tables_dir + '/histogramme_taille_physique_max_diam.pdf', dpi=300, bbox_inches='tight')
    plt.show()
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
else:
    print("Aucun résultat à filtrer pour le diamètre maximal (table vide ou colonne manquante).")
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
# for val, patch in zip(hist_vals, patches):
#     if val > 0:
#         plt.text(patch.get_x() + patch.get_width()/2, val + 0.05, f'{int(val)}', ha='center', va='bottom', fontsize=12, color='#333333')
# plt.xlabel('$e$', fontsize=14)# Anglais
# plt.ylabel('N. stars', fontsize=14)
plt.xlabel(r'$\rm \varepsilon$', fontsize=14)  # Français
plt.ylabel("Nombre d'étoiles", fontsize=14)
#plt.title('Histogram of ellipticities\n(max diameter per star)', fontsize=17, fontweight='bold', color='#4A90E2', pad=18)
#plt.grid(alpha=0.3, linestyle='--')
plt.legend(fontsize=14)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
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
# Ajout d'annotations sur chaque barre
# for val, patch in zip(hist_vals_phys, patches_phys):
#     if val > 0:
#         plt.text(patch.get_x() + patch.get_width()/2, val + 0.05, f'{int(val)}', ha='center', va='bottom', fontsize=12, color='#333333')
# plt.xlabel('$D_{maj}$ (AU)', fontsize=14)# Anglais
# plt.ylabel('N. stars', fontsize=14)
plt.xlabel('$D_{maj}$ (AU)', fontsize=14)# Français
plt.ylabel("Nombre d'étoiles", fontsize=14)
#plt.title('Histogram of physical sizes\n(max diameter per star)', fontsize=16, fontweight='bold', color='#50B878', pad=16)
plt.legend(fontsize=14)
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
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
# def assemble_grand_panels_paginated(panel_dir, output_path_base, ncols=2, nrows=7):
#     """
#     Crée plusieurs grands panels paginés, chaque panel contenant au maximum ncols*nrows images.
#     - panel_dir: dossier où sont stockés les panels individuels (DoLP+contours & PI)
#     - output_path_base: base du chemin de sortie (sans extension ni numéro de page)
#     - ncols: nombre de colonnes
#     - nrows: nombre de lignes
#     """
#     import matplotlib.image as mpimg
#     import matplotlib.pyplot as plt
#     import os
#     image_files = [f for f in os.listdir(panel_dir) if f.endswith('PI_DoLP_panel.png')]
#     n_panels_per_page = ncols * nrows
#     total = len(image_files)
#     page = 0
#     for start in range(0, total, n_panels_per_page):
#         page += 1
#         end = min(start + n_panels_per_page, total)
#         fig, axes = plt.subplots(nrows, ncols, figsize=(12*ncols, 5*nrows))
#         if nrows == 1:
#             axes = np.array([axes])
#         axes = axes.reshape(nrows, ncols)
#         for idx, fname in enumerate(image_files[start:end]):
#             row = idx // ncols
#             col = idx % ncols
#             img_path = os.path.join(panel_dir, fname)
#             img = mpimg.imread(img_path)
#             axes[row, col].imshow(img)
#             axes[row, col].axis('off')
#             #axes[row, col].set_title(fname.replace('_PI_DoLP_panel.png',''), fontsize=18)
#         # Désactive les axes vides
#         for idx in range(end-start, n_panels_per_page):
#             row = idx // ncols
#             col = idx % ncols
#             axes[row, col].axis('off')
#         panel_name = 'PI_panel'
#         # Ajuste les espacements pour un remplissage optimal
#         plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.10, hspace=0.01)
#         plt.savefig(f"{output_path_base}png/{panel_name}_page_{page}.png", dpi=300, bbox_inches='tight')
#         plt.savefig(f"{output_path_base}pdf/{panel_name}_page_{page}.pdf", dpi=300, bbox_inches='tight')
#         plt.close(fig)
#         print(f"✅ Grand panel page {page} sauvegardé sous {output_path_base}png/{panel_name}_page_{page}.png et {output_path_base}pdf/{panel_name}_page_{page}.pdf")
#         # Appel de la fonction pour assembler les panels paginés
# panels_directory = '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Morphologies_contours/Panels/specific/'
# output_base = '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Morphologies_contours/Panels/grand_panel/'
# os.makedirs(output_base + 'png', exist_ok=True)
# os.makedirs(output_base + 'pdf', exist_ok=True)
# assemble_grand_panels_paginated(panels_directory, output_base, ncols=2, nrows=7) # Appel de la fonction pour assembler les panels paginés

def assemble_grand_panels_paginated(panel_dir, output_path_base, ncols=2, nrows=7,
                                     suffix='PI_DoLP_panel.png', panel_name='PI_panel',
                                     cell_size=(12, 5)):
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import os, numpy as np
    image_files = sorted(f for f in os.listdir(panel_dir) if f.endswith(suffix))
    if not image_files:
        print(f"Aucune image trouvée dans {panel_dir} avec suffixe {suffix}")
        return
    n_panels_per_page = ncols * nrows
    total = len(image_files)
    page = 0
    for start in range(0, total, n_panels_per_page):
        page += 1
        end = min(start + n_panels_per_page, total)
        fig, axes = plt.subplots(nrows, ncols, figsize=(cell_size[0]*ncols, cell_size[1]*nrows))
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
        for idx in range(end-start, n_panels_per_page):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.10, hspace=0.01)
        plt.savefig(f"{output_path_base}png/{panel_name}_page_{page}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_path_base}pdf/{panel_name}_page_{page}.pdf", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ {panel_name} page {page} → {output_path_base}png/{panel_name}_page_{page}.png")
# dolp_specific_dir = '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Morphologies_contours/DoLP_specific/png/'
# output_base_dolp = '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Morphologies_contours/DoLP_specific/grand_panel/'
# os.makedirs(output_base_dolp + 'png', exist_ok=True)
# os.makedirs(output_base_dolp + 'pdf', exist_ok=True)

# assemble_grand_panels_paginated(dolp_specific_dir, output_base_dolp, ncols=3, nrows=1,
#                                 suffix='_DoLP_only.png', panel_name='DoLP_only',
#                                 cell_size=(6, 5))


        

# panels_directory = '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Morphologies_contours/Panels/specific/'
# output_base = '/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Morphologies_contours/Panels/grand_panel/'
# os.makedirs(output_base + 'png', exist_ok=True)
# os.makedirs(output_base + 'pdf', exist_ok=True)
# assemble_grand_panels_paginated(panels_directory, output_base, ncols=2, nrows=2,
#                                 suffix='PI_DoLP_panel.png', panel_name='PI_panel',
#                                 cell_size=(12, 5)) # Appel de la fonction pour assembler les panels paginés



def circ_mean_std_deg(angles_deg, period=180.0):
    # angles in degrees, period=180 for PA convention [0,180)
    ang = np.deg2rad(np.asarray(angles_deg) * (180.0/period))  # map to radians (if using 180° period)
    # if angles already in 0-180 and you want treat them as circular on 180°, above scaling keeps consistency.
    s = np.sum(np.sin(ang))
    c = np.sum(np.cos(ang))
    mu = np.arctan2(s, c)            # mean angle in rad
    R = np.hypot(s, c) / len(ang)
    circ_std = np.sqrt(-2.0 * np.log(np.clip(R, 1e-12, 1.0)))
    # convert back to degrees in original period
    mu_deg = (np.rad2deg(mu) * (period/180.0)) % period
    circ_std_deg = np.rad2deg(circ_std) * (period/180.0)
    se_mean_deg = circ_std_deg / np.sqrt(len(ang))
    return mu_deg, circ_std_deg, se_mean_deg, R

def bootstrap_mean_ci_deg(angles_deg, nboot=2000, period=180.0, ci=68):
    arr = np.asarray(angles_deg)
    boots = []
    for _ in range(nboot):
        samp = np.random.choice(arr, size=len(arr), replace=True)
        mu, _, _, _ = circ_mean_std_deg(samp, period=period)
        boots.append(mu)
    boots = np.array(boots)
    lo = np.percentile(boots, 50 - ci/2)
    hi = np.percentile(boots, 50 + ci/2)
    med = np.median(boots)
    return med, lo, hi, boots

# usage minimal
angles = [92.23810036046609, 93.81594665297884, 92.43494686058115,
          82.89436649059326, 72.07300442372502, 81.79825878133693, 83.2742812522207]
mu, circ_std, se, R = circ_mean_std_deg(angles, period=180.0)
med, lo, hi, boots = bootstrap_mean_ci_deg(angles, nboot=5000, period=180.0)
print(f"PA_mean={mu:.2f}°  circ_std={circ_std:.2f}°  SE≈{se:.2f}°  68% CI ≈ [{lo:.2f},{hi:.2f}]")

# --- BEGIN ADD: estimation Monte‑Carlo de l'incertitude sur PA par perturbation des vertices ---
import numpy as _np
from skimage.measure import EllipseModel as _EllipseModel

def pa_uncertainty_from_verts(verts, nboot=1000, noise_sigma_px=0.5, return_dist=False):
    """
    Estimate PA uncertainty by Monte‑Carlo perturbation of contour vertices.
    - verts: (N,2) array of contour vertices (in same units as used for fit, e.g. mas).
    - nboot: number of realizations.
    - noise_sigma_px: gaussian stddev applied to x,y perturbation in same units as verts (typ. 0.5 px).
    Returns: (pa_med_deg, pa_sigma_deg, lo68_deg, hi68_deg) or additionally the distribution.
    """
    verts = _np.asarray(verts)
    pa_list = []
    for _ in range(int(nboot)):
        pts = verts + _np.random.normal(scale=noise_sigma_px, size=verts.shape)
        model = _EllipseModel()
        try:
            ok = model.estimate(pts)
        except Exception:
            ok = False
        if not ok:
            continue
        xc, yc, a_e, b_e, theta_e = model.params
        pa_deg = float((_np.degrees(_np.pi/2.0 + theta_e)) % 180.0)
        pa_list.append(pa_deg)
    pa_arr = _np.array(pa_list)
    if pa_arr.size == 0:
        if return_dist:
            return _np.nan, _np.nan, _np.nan, _np.nan, pa_arr
        return _np.nan, _np.nan, _np.nan, _np.nan
    # recentre en évitant le wrap modulo 180
    med = _np.median(pa_arr)
    diff = ((_np.asarray(pa_arr) - med + 90.0) % 180.0) - 90.0
    pa_centered = med + diff
    sigma = float(_np.std(pa_centered))
    lo = float(_np.percentile(pa_centered, 16))
    hi = float(_np.percentile(pa_centered, 84))
    pa_med = float(_np.median(pa_centered)) % 180.0
    if return_dist:
        return pa_med, sigma, lo, hi, pa_arr
    return pa_med, sigma, lo, hi
