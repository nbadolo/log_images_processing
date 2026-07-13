#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 17 19:42:43 2025

@author: nbadolo
"""



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'extraction de données FITS :
- Récupère des infos depuis les headers FITS
- Trie les données par date d'observation
- Affiche la date une seule fois par jour
- Place la colonne 'Etoile' en 3e position

Auteur : nbadolo
Date : 16 mai 2025
"""

import os
import csv
from astropy.io import fits
from datetime import datetime
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# === Dossiers ===

#folder = "V854_Cen"
folder = "large_log_+"
input_path = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian"
racine = os.path.join(input_path, "Input", folder)
dossier_logs = os.path.join(input_path, "Output", folder, "logs")
os.makedirs(dossier_logs, exist_ok=True)

# === Fichier de log horodaté ===
#timestamp = datetime.now().strftime('%Y%m%d_%H%M')
log_file = os.path.join(dossier_logs, f'log_observations_{folder}.csv')

# === FITS header keywords to extract === francçais
keywords = {
    'Observation date'    : 'DATE-OBS',
    'Program ID'          : 'HIERARCH ESO OBS PROG ID',
    'Target name'         : 'OBJECT',
    'Neutral density filter (ND)' : 'HIERARCH ESO INS3 OPTI2 NAME',
    'Filter 1'            : 'HIERARCH ESO INS3 OPTI5 NAME',
    'Filter 2'            : 'HIERARCH ESO INS3 OPTI6 NAME',
    'Seeing'              : 'HIERARCH ESO OBS AMBI FWHM',
    'Airmass'             : 'HIERARCH ESO OBS AIRM',
}

# # === Champs à extraire des en-têtes FITS === anglais
# keywords = {
#     'Date d’observation'     : 'DATE-OBS',
#     'Programme d’observation': 'HIERARCH ESO OBS PROG ID',
#     'PI du programme'        : 'HIERARCH ESO OBS PI-COI',

#     'Objet observé'          : 'OBJECT',
#     'Filtre neutre (ND)'     : 'HIERARCH ESO INS3 OPTI2 NAME',
#     'Filtre 1'               : 'HIERARCH ESO INS3 OPTI5 NAME',
#     'Filtre 2'               : 'HIERARCH ESO INS3 OPTI6 NAME',
#     'Seeing'                 : 'HIERARCH ESO OBS AMBI FWHM',
#     'Airmass'                : 'HIERARCH ESO OBS AIRM',
# }

# === Fonction de normalisation des noms d'étoiles ===
def normalize_star_name(name):
    """
    Normalise un nom d'étoile pour faciliter la correspondance :
    - Supprime les espaces
    - Remplace les tirets par rien
    - Met en majuscules
    - Remplace 'l' par 'I' dans certains cas (COl -> COI)
    """
    if not isinstance(name, str):
        return str(name)
    normalized = name.upper().replace(' ', '').replace('-', '').replace('_', '')
    # Correction spécifique pour SW COl -> SW COI
    normalized = normalized.replace('SWCOL', 'SWCOI')
    return normalized

# === Dictionnaire de correction des noms mal orthographiés ===
# Mappe les noms normalisés vers les bons noms
name_corrections = {
    'SWCOL': 'SW Col',
    'SWCOI': 'SW Col',
    'ALPHAHER': 'Alpha Her',
    'CHICYG': 'Chi Cyg',
    'ZERI': 'Z Eri',
    'RPEG': 'R Peg',
    'BWOCT': 'BW Oct',
    'ACCET': 'AC Cet',
    'DZAQR': 'DZ Aqr',
    'ZPEG': 'Z Peg',
    'WPEG': 'W Peg',
    'RTVIR': 'RT Vir',
    'RXLEP': 'RX Lep',
    'BETAGRU': 'Beta Gru',
    'TMIC': 'T Mic',
    'RDOR': 'R Dor',
    'AKHYA': 'AK Hya',
    'RLEO': 'R Leo',
    'BKVIR': 'BK Vir',
    'TCET': 'T Cet',
    'MIRACETI': 'Mira Ceti',
    'UDEL': 'U Del',
    'UHER': 'U Her',
    'PI1GRU': r'$\pi^1$ Gru',
    'Π1GRU': r'$\pi^1$ Gru',
    'WAQL': 'W Aql',
    'VPSA': 'V PsA',
    'RHYA': 'R Hya',
    'RAQL': 'R Aql',
    'SPAV': 'S Pav',
    'GYAQL': 'GY Aql',
    'SVAQR': 'SV Aqr',
    'UPSCET': 'Ups Cet',
    'V1943SGR': 'V1943 Sgr',
    'VV1943SGR': 'V1943 Sgr',  # VV1943 Sgr -> V1943 Sgr
    'PSIPHE': 'Psi Phe',
    'VSLEP': 'V S Lep',
    'SSLEP': 'SS Lep',
    '17LEP': 'SS Lep',  # 17 Lep est en fait SS Lep
    'L2PUP': 'L_2 Pup',
    'L02PUP': 'L_2 Pup',  # L02 Pup -> L2 Pup
    'CWCNC': 'CW Cnc',
    'VCWCNC': 'CW Cnc',  # VCW Cnc -> CW Cnc
    'YPAV': 'Y Pav',
    'RHOR': 'R Hor',
    'RSCL': 'R Scl',
    'YSCL': 'Y Scl',
    'RCRT': 'R Crt',
    'WHYA': 'W Hya',
    'SWVIR': 'SW Vir',
    'VHYA': 'V Hya',
    'ups Cet': 'Ups Cet',
}

def correct_star_name(name):
    """
    Corrige les noms d'étoiles mal orthographiés en utilisant le dictionnaire
    """
    if not isinstance(name, str):
        return str(name)
    
    normalized = normalize_star_name(name)
    
    # Si on trouve une correction, on l'utilise
    if normalized in name_corrections:
        return name_corrections[normalized]
    
    # Sinon on retourne le nom original
    return name.strip()

# === Dictionnaire PSF pour chaque observation ===
# Format: (Date, Time, Target normalisé) -> "Yes" ou "No"
psf_dict_raw = {
    ('2015-09-22', '06:00:05', 'Y Pav'): 'No',
    ('2015-09-23', '06:33:06', 'R Hor'): 'Yes',
    ('2015-09-23', '05:22:59', 'R Scl'): 'Yes',
    ('2015-09-23', '04:32:28', 'Y Scl'): 'Yes',
    ('2015-12-19', '06:56:44', 'R Crt'): 'Yes',
    ('2016-02-19', '02:00:31', 'SW Col'): 'No',
    ('2016-03-08', '07:53:40', 'W Hya'): 'No',
    ('2016-03-09', '07:35:37', 'SW Vir'): 'No',
    ('2016-04-22', '01:37:40', 'V Hya'): 'Yes',
    ('2016-04-30', '06:08:35', 'Alpha Her'): 'Yes',
    ('2016-06-30', '05:27:44', 'Chi Cyg'): 'Yes',
    ('2016-07-22', '10:13:16', 'Z Eri'): 'No',
    ('2016-07-22', '07:33:48', 'R Peg'): 'Yes',
    ('2016-07-22', '06:48:41', 'BW Oct'): 'Yes',
    ('2016-07-22', '09:09:48', 'AC Cet'): 'Yes',
    ('2016-07-22', '07:12:00', 'DZ Aqr'): 'Yes',
    ('2016-07-22', '08:40:12', 'Z Peg'): 'Yes',
    ('2016-07-22', '08:13:34', 'W Peg'): 'Yes',
    ('2016-07-29', '23:20:37', 'RT Vir'): 'No',
    ('2016-10-04', '06:33:30', 'RX Lep'): 'Yes',
    ('2016-10-12', '02:07:05', 'Beta Gru'): 'Yes',
    ('2016-10-12', '01:05:09', 'T Mic'): 'Yes',
    ('2016-11-09', '04:13:02', 'R Dor'): 'No',
    ('2016-12-07', '06:06:15', 'AK Hya'): 'No',
    ('2016-12-14', '08:47:36', 'R Leo'): 'No',
    ('2017-03-07', '04:08:12', 'BK Vir'): 'No',
    ('2017-07-31', '07:44:43', 'T Cet'): 'No',
    ('2017-11-27', '03:03:26', 'Mira Ceti'): 'Yes',
    ('2019-07-05', '05:09:14', 'U Del'): 'No',
    ('2019-07-08', '01:55:17', 'U Her'): 'No',
    ('2019-07-08', '09:20:26', 'π¹ Gru'): 'No',
    ('2019-07-09', '04:53:58', 'W Aql'): 'No',
    ('2019-07-09', '08:51:53', 'V PsA'): 'No',
    ('2019-07-27', '01:35:53', 'R Hya'): 'No',
    ('2019-07-30', '00:53:26', 'R Aql'): 'Yes',
    ('2019-09-24', '03:38:22', 'S Pav'): 'No',
    ('2019-09-28', '03:18:41', 'GY Aql'): 'Yes',
    ('2019-09-29', '03:02:18', 'SV Aqr'): 'Yes',
    ('2019-10-16', '03:26:11', 'ups Cet'): 'Yes',
    ('2019-10-16', '01:42:05', 'V1943 Sgr'): 'Yes',
    ('2019-10-16', '02:46:51', 'Psi Phe'): 'Yes',
    ('2020-02-29', '01:51:51', 'V S Lep'): 'No',
    ('2020-02-29', '00:10:45', 'SS Lep'): 'Yes',
    ('2020-02-29', '00:48:30', 'L₂ Pup'): 'No',
    ('2020-02-29', '02:20:39', 'CW Cnc'): 'No',
}

# Créer un dictionnaire avec des clés normalisées
psf_dict = {}
for (date, time, target), value in psf_dict_raw.items():
    normalized_target = normalize_star_name(target)
    psf_dict[(date, time, normalized_target)] = value

# === Initialisation ===
donnees = []
nb_total = nb_ok = nb_erreurs = nb_sans_fits = 0

# === Parcours des dossiers étoiles ===
for dossier in os.listdir(racine):
    nb_total += 1
    dossier_etoile = os.path.join(racine, dossier)

    if not os.path.isdir(dossier_etoile):
        continue

    dossier_star = os.path.join(dossier_etoile, 'Intensity', 'star')
    if not os.path.isdir(dossier_star):
        print(f"[!] Dossier manquant pour {dossier}")
        nb_sans_fits += 1
        continue

    found_fits = False

    for root, _, files in os.walk(dossier_star):
        for file in files:
            if file.endswith('.fits') or file.endswith('.fit'):
                chemin_fits = os.path.join(root, file)
                try:
                    with fits.open(chemin_fits) as hdul:
                        header = hdul[0].header

                        # 1) Date/Heure : privilégie DATE-OBS, sinon variants ESO fréquents
                        datetime_obs = (
                            header.get('DATE-OBS')
                            or header.get(keywords['Observation date'])
                            or header.get('HIERARCH ESO OBS START')
                            or header.get('HIERARCH ESO TPL START')
                            or 'N/A'
                        )
                        if isinstance(datetime_obs, bytes):
                            datetime_obs = datetime_obs.decode(errors='ignore')
                        if isinstance(datetime_obs, str) and 'T' in datetime_obs:
                            date_part, time_part = datetime_obs.split('T', 1)
                        else:
                            date_part, time_part = (str(datetime_obs), '')

                        # Formatage heure: retire fractions/zone et garde HH:MM:SS, avec espaces autour des :
                        time_display = ''
                        if time_part:
                            t = time_part.strip()
                            # retire suffixes éventuels (Z, timezone)
                            t = t.split('Z')[0].split('+')[0].split('-')[0]
                            # garde uniquement HH:MM:SS
                            t = t.split('.')[0]
                            # complète si HH:MM seulement
                            if len(t.split(':')) == 2:
                                h, m = t.split(':')
                                t = f"{h}:{m}:00"
                            # espaces autour des deux :
                            time_display = t.replace(':', ':')

                        # 2) Nom de cible : OBJECT avec retombes possibles
                        nom_objet = (
                            header.get('OBJECT')
                            or header.get(keywords['Target name'])
                            or header.get('HIERARCH ESO OBS TARG NAME')
                            or 'N/A'
                        )
                        if isinstance(nom_objet, bytes):
                            nom_objet = nom_objet.decode(errors='ignore')
                        nom_objet = nom_objet.strip()
                        
                        # Corriger le nom de l'objet si mal orthographié
                        nom_objet_corrige = correct_star_name(nom_objet)

                        # 3) Program metadata
                        prog_id = (
                            header.get('HIERARCH ESO OBS PROG ID')
                            or header.get(keywords['Program ID'])
                            or 'N/A'
                        )
                        # pi_name = (
                        #     #header.get('HIERARCH ESO OBS PI-COI ID')
                        #     header.get('HIERARCH ESO OBS PI-COI NAME')
                        #     or header.get('HIERARCH ESO OBS PI-COI')
                        #     or 'N/A'
                        # )
                        if isinstance(prog_id, bytes):
                            prog_id = prog_id.decode(errors='ignore')
                        # if isinstance(pi_name, bytes):
                        #     pi_name = pi_name.decode(errors='ignore')
                        prog_id = str(prog_id).strip()
                        #pi_name = str(pi_name).strip()

                        # 4) Construire la ligne dans l'ordre de header_final
                        #ligne = [date_part, time_part, nom_objet, prog_id]
                        ligne = [date_part, time_display, nom_objet_corrige, prog_id]
                        
                        # Ajouter la colonne PSF basée sur le dictionnaire
                        # Normaliser le nom de l'objet corrigé pour la recherche
                        normalized_name = normalize_star_name(nom_objet_corrige)
                        psf_key = (date_part, time_display, normalized_name)
                        psf_status = psf_dict.get(psf_key, 'N/A')
                        ligne.append(psf_status)
                        
                        champs = [
                            'Neutral density filter (ND)',
                            'Filter 1',
                            'Filter 2',
                            'Seeing',
                            'Airmass',
                        ]
                        for k in champs:
                            val = header.get(keywords[k], 'N/A')
                            if isinstance(val, bytes):
                                val = val.decode(errors='ignore')
                            ligne.append(val)

                        donnees.append(ligne)
                        print(f"[✓] {nom_objet_corrige} traité : {file}")
                        nb_ok += 1
                        found_fits = True
                        break

                except Exception as e:
                    print(f"[!] Erreur avec {dossier} : {e}")
                    nb_erreurs += 1
                    found_fits = True
                    break
        if found_fits:
            break

    if not found_fits:
        print(f"[!] Aucun FITS trouvé pour {dossier}")
        nb_sans_fits += 1

# === Tri par date ===
donnees.sort(key=lambda x: x[0])  # x[0] = Date

# === Masquer les dates répétées pour les observations du même jour ===
derniere_date = ""
for ligne in donnees:
    if ligne[0] == derniere_date:
        ligne[0] = ''
    else:
        derniere_date = ligne[0]

# === En-têtes dans le bon ordre ===
header_final = [
    'Date', 'Time', 'Target',
    'Program ID', 'Dedicated PSF',
    'ND', 'Filter 1', 'Filter 2', 'Seeing', 'Airmass'
]
# === Écriture CSV ===
with open(log_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header_final)
    writer.writerows(donnees)

# === Résumé final ===
print("\n========== RÉSUMÉ DE L’EXÉCUTION ==========")
print(f"🔢 Total d’étoiles analysées : {nb_total}")
print(f"✅ FITS traités : {nb_ok}")
print(f"❌ Erreurs FITS : {nb_erreurs}")
print(f"🚫 Sans fichier FITS : {nb_sans_fits}")
print(f"📄 Log sauvegardé : {log_file}")

#%%
# =============================================================================
# Conversion du fichier de sortie .csv en latex
# =============================================================================

# Rechargement du CSV en DataFrame
# Rechargement du CSV en DataFrame
df = pd.read_csv(log_file)
pd.set_option('display.max_columns', None)  # Affiche toutes les colonnes
print(df)

# Générer la table LaTeX avec escape=True pour bien gérer les underscores
# (on ne s'en sert plus ici mais on conserve le df)
# latex_table = df.to_latex(index=False, escape=True)

# Construction du tableau tabular seul (sans environnement table)
latex_table = "\\begin{tabular}{ll" + "c" * (df.shape[1] - 2) + "}\n"
latex_table += "\\hline\n\\hline\n"

# En-têtes
columns = df.columns.tolist()
header_row = " & ".join(columns) + " \\\\\n"
latex_table += header_row
latex_table += "\\hline\n"

# Lignes de données
for _, row in df.iterrows():
    # On échappe les underscores manuellement (sécurité)
    # Remplacer 'nan' par chaîne vide pour la colonne Date
    line = " & ".join(str(val).replace('_', '\\_').replace('nan', '') for val in row.tolist()) + " \\\\\n"
    latex_table += line

latex_table += "\\hline\n\\hline\n"
latex_table += "\\end{tabular}\n"

# Sauvegarde dans un fichier .tex
latex_table_path = os.path.splitext(log_file)[0] + "_styled.tex"
with open(latex_table_path, 'w') as f:
    f.write(latex_table)

print(f"📎 Tableau LaTeX stylisé (tabular seul) sauvegardé dans : {latex_table_path}")

# === Version française du tableau ===
# Dictionnaire de traduction des en-têtes
french_headers = {
    'Date'          : 'Date',
    'Time'          : 'Heure',
    'Program ID'    : 'ID Programme',
    'Target'        : 'Cible',
    'Dedicated PSF' : 'PSF Dédiée',
    'ND'            : 'Filtre ND',
    'Filter 1'      : 'Filtre 1',
    'Filter 2'      : 'Filtre 2',
    'Seeing'        : 'Seeing',
    'Airmass'       : 'Airmass',
}

# Construction du tableau tabular français
latex_table_fr = "\\begin{tabular}{ll" + "c" * (df.shape[1] - 2) + "}\n"
latex_table_fr += "\\hline\n\\hline\n"

# En-têtes en français
columns_fr = [french_headers.get(col, col) for col in df.columns.tolist()]
header_row_fr = " & ".join(columns_fr) + " \\\\\n"
latex_table_fr += header_row_fr
latex_table_fr += "\\hline\n"

# Lignes de données (identiques)
for _, row in df.iterrows():
    # On échappe les underscores manuellement (sécurité)
    # Remplacer 'nan' par chaîne vide pour la colonne Date
    line = " & ".join(str(val).replace('_', '\\_').replace('nan', '') for val in row.tolist()) + " \\\\\n"
    latex_table_fr += line

latex_table_fr += "\\hline\n\\hline\n"
latex_table_fr += "\\end{tabular}\n"

# Sauvegarde dans un fichier .tex français
latex_table_fr_path = os.path.splitext(log_file)[0] + "_french_styled.tex"
with open(latex_table_fr_path, 'w') as f:
    f.write(latex_table_fr)

print(f"📎 Tableau LaTeX français sauvegardé dans : {latex_table_fr_path}")



# Conversion de la table des paramètres stellaires des étoiles de mon échantillon.
folder_name = "Hipparcos/Hip"  # Nom du dossier contenant les données
file_name = "Sample_summary_table_hip.csv" # Échantillon sélectionné de 45 étoiles
main_path = f"/home/nbadolo/Bureau/Aymard/Tables/ML/{folder_name}/"

# Charger la table des noms
names_name = "star_names.csv"  # Nom du fichier contenant les noms des étoile
df_names = pd.read_csv(main_path + names_name)

df_sample = pd.read_csv(main_path + file_name)
print(df_sample.columns)
pd.set_option('display.max_columns', None)

# Harmoniser le format des identifiants Hipparcos pour le cross-match
hip_ids = df_sample["Object"].astype(str).str.replace('_', ' ').str.strip()
hip_to_name = dict(zip(df_names["HIP_SIMBAD"].astype(str).str.strip(), df_names["Nom"]))
object_names = [hip_to_name.get(hip_id, "") for hip_id in hip_ids]

# Ajout de la colonne "Object name"
if "Object name" not in df_sample.columns:
    df_sample.insert(0, "Object name", object_names)
# Création du dictionnaire de correspondance HIP_SIMBAD → Nom
hip_to_name = dict(zip(df_names["HIP_SIMBAD"].astype(str), df_names["Nom"]))

# Ajouter la colonne "Object name" au début de df_sample
# Utilise la colonne d'origine pour les identifiants Hipparcos
hip_ids = df_sample["Object"].astype(str).str.strip()
object_names = [hip_to_name.get(hip_id, "") for hip_id in hip_ids]
# Renommer la colonne Hipparcos ID (ancienne première colonne, maintenant deuxième)
df_sample.rename(columns={df_sample.columns[1]: r"\begin{tabular}{c} Hipparcos\\ID \end{tabular}"}, inplace=True)
df_sample.rename(columns={df_sample.columns[0]: r"\begin{tabular}{c} Object\\name \end{tabular}"}, inplace=True)
# Colonnes à inclure (toutes sauf 8 à 11 et l'avant-dernière)
cols_to_include = [i for i in range(len(df_sample.columns)) if (i < 8 or i > 11) and i != len(df_sample.columns) - 2]
columns_filtered = [df_sample.columns[i] for i in cols_to_include]
columns_filtered[-1] = r"\begin{tabular}{c} Evolved\\envelope ? \end{tabular}"
for idx, col in enumerate(columns_filtered):
    if col == "Teff":
        columns_filtered[idx] = r"\begin{tabular}{c} Teff\\(K) \end{tabular}"
    elif col == "Lum":
        columns_filtered[idx] = r"\begin{tabular}{c} Lum\\($L_\odot$) \end{tabular}"
    elif col == "Distance":
        columns_filtered[idx] = r"\begin{tabular}{c} Distance\\(pc) \end{tabular}"
    elif col == "Av":
        columns_filtered[idx] = r"\begin{tabular}{c} Av\\(mag) \end{tabular}"
    elif col == "logg":
        columns_filtered[idx] = r"\begin{tabular}{c} logg\\(dex) \end{tabular}"

sample_params = "\\begin{tabular}{l" + "c" * (len(columns_filtered) - 1) + "}\n"
sample_params += "\\toprule\n\\hline\n"

header_row_filtered = " & ".join(columns_filtered) + " \\\\\n"
sample_params += header_row_filtered
sample_params += "\\midrule\n"

for _, row in df_sample.iterrows():
    formatted_row = []
    for i in cols_to_include:
        col = df_sample.columns[i]
        val = row[i]
        # Première colonne et deux dernières : pas de modification
        if i == 0:
            # Première colonne (Object name) : remplacer underscore par espace
            formatted_val = str(val).replace('_', ' ')
        elif i == 1:
            # Deuxième colonne (Hipparcos ID) : remplacer underscore par espace
            formatted_val = str(val).replace('_', ' ')
        elif i >= len(df_sample.columns) - 2:
            formatted_val = str(val)
        elif col == "LIR/L*":
            try:
                formatted_val = f"{float(val):.4f}"
            except Exception:
                formatted_val = str(val)
        elif col in ["Teff", "Lum", "Distance"]:
            try:
                formatted_val = f"{int(round(float(val)))}"
            except Exception:
                formatted_val = str(val)
        else:
            try:
                formatted_val = f"{float(val):.2f}"
            except Exception:
                formatted_val = str(val)
        formatted_row.append(formatted_val.replace('_', '\\_'))
    line = " & ".join(formatted_row) + " \\\\\n"
    sample_params += line

sample_params += "\\bottomrule\n"
sample_params += "\\end{tabular}\n"

latex_sample = os.path.splitext(main_path + file_name)[0] + "_styled.tex"
with open(latex_sample, 'w') as f:
    f.write(sample_params)

print(f"📎 Tableau LaTeX stylisé (tabular seul) sauvegardé dans : {latex_sample}")
print("Identifiants dans df_sample :")
print(df_sample.iloc[:, 0].astype(str).str.strip().head(10).to_list())

print("Identifiants dans df_names :")
print(df_names["HIP_SIMBAD"].astype(str).str.strip().head(10).to_list())