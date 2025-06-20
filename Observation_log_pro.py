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
folder = "McD_interrompu"
input_path = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian"
racine = os.path.join(input_path, "Input", folder)
dossier_logs = os.path.join(input_path, "Output", folder, "logs")
os.makedirs(dossier_logs, exist_ok=True)

# === Fichier de log horodaté ===
#timestamp = datetime.now().strftime('%Y%m%d_%H%M')
log_file = os.path.join(dossier_logs, f'log_observations_{folder}.csv')

# === Champs à extraire des headers FITS ===
keywords = {
    'DateTime': 'DATE-OBS',
    'Object': 'OBJECT',
    'ND': 'HIERARCH ESO INS3 OPTI2 NAME',
    'Filter 1': 'HIERARCH ESO INS3 OPTI5 NAME',
    'Filter 2': 'HIERARCH ESO INS3 OPTI6 NAME',
    'Seeing': 'HIERARCH ESO OBS AMBI FWHM',
    'Airmass': 'HIERARCH ESO OBS AIRM',
}

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

                        # Extraire DATE-OBS et séparer date/heure
                        datetime_obs = header.get(keywords['DateTime'], 'N/A')
                        if 'T' in datetime_obs:
                            date_part, time_part = datetime_obs.split('T')
                        else:
                            date_part, time_part = datetime_obs, ''

                        # Nom de l'objet depuis le FITS (pas depuis le nom du dossier)
                        nom_objet = header.get(keywords['Object'], 'N/A').strip()

                        # Organiser les champs : Date, Time, Etoile, autres
                        ligne = [date_part, time_part, nom_objet]
                        for key in list(keywords.keys())[2:]:  # à partir de ND
                            val = header.get(keywords[key], 'N/A')
                            ligne.append(val)

                        donnees.append(ligne)
                        print(f"[✓] {nom_objet} traité : {file}")
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
header_final = ['Date', 'Time', 'Target', 'ND', 'Filter 1', 'Filter 2', 'Seeing', 'Airmass']

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
df = pd.read_csv(log_file)

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
    line = " & ".join(str(val).replace('_', '\\_') for val in row.tolist()) + " \\\\\n"
    latex_table += line

latex_table += "\\hline\n\\hline\n"
latex_table += "\\end{tabular}\n"

# Sauvegarde dans un fichier .tex
latex_table_path = os.path.splitext(log_file)[0] + "_styled.tex"
with open(latex_table_path, 'w') as f:
    f.write(latex_table)

print(f"📎 Tableau LaTeX stylisé (tabular seul) sauvegardé dans : {latex_table_path}")