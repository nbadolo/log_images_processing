#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 18:56:02 2025

@author: nbadolo
"""



import os
import csv
from astropy.io import fits
from datetime import datetime

# Dossier racine contenant tous les dossiers d’étoiles
input_path = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian"
racine = f"{input_path}/Input/large_log_+"

# Dossier où seront sauvegardés les fichiers de log
dossier_logs = f"{input_path}/Output/large_log_+/logs/"
os.makedirs(dossier_logs, exist_ok=True)  # Crée le dossier s’il n'existe pas

# ======= CONFIGURATION =======

# Dossier racine contenant les sous-dossiers pour chaque étoile

# Dossier où seront sauvegardés les fichiers logs (créé s'il n'existe pas)

# Création d’un nom de fichier basé sur la date et l’heure actuelle
timestamp = datetime.now().strftime('%Y%m%d_%H%M')
log_file = os.path.join(dossier_logs, f'log_observations_{timestamp}.csv')

# Dictionnaire définissant les champs à extraire depuis les headers FITS
# La clé est le nom qui sera utilisé dans le CSV,
# la valeur est la clé exacte du header FITS à chercher
keywords = {
    'Date': 'DATE-OBS',                               # Date et heure d’observation
    'Object': 'OBJECT',                               # Nom de l’objet observé
    'ND': 'HIERARCH ESO INS3 OPTI2 NAME',            # Nom du filtre ND
    'Filter 1': 'HIERARCH ESO INS3 OPTI5 NAME',      # Nom du premier filtre
    'Filter 2': 'HIERARCH ESO INS3 OPTI6 NAME',      # Nom du deuxième filtre
    'Seeing': 'HIERARCH ESO OBS AMBI FWHM',          # Valeur du seeing ambiant (FWHM)
    'Airmass': 'HIERARCH ESO OBS AIRM',               # Masse d’air à l’observation
}

# ======= INITIALISATION DES COMPTEURS =======

nb_total = 0      # Nombre total de dossiers étoile analysés
nb_ok = 0         # Nombre de fichiers FITS traités avec succès
nb_erreurs = 0    # Nombre d’erreurs rencontrées à la lecture des FITS
nb_sans_fits = 0  # Nombre de dossiers étoile sans fichier FITS trouvé

# ======= OUVERTURE DU FICHIER CSV POUR ÉCRITURE =======

with open(log_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Écriture de la ligne d’en-tête avec "Etoile" puis les clés définies dans keywords
    writer.writerow(['Etoile'] + list(keywords.keys()))

    # Parcours des dossiers dans le dossier racine (un dossier par étoile)
    for etoile in os.listdir(racine):
        nb_total += 1
        dossier_etoile = os.path.join(racine, etoile)

        # Ignorer si ce n’est pas un dossier
        if not os.path.isdir(dossier_etoile):
            continue

        # Chemin vers le dossier qui contient les fichiers FITS (Intensity/star)
        dossier_star = os.path.join(dossier_etoile, 'Intensity', 'star')

        # Si ce dossier n’existe pas, on le signale et on passe au suivant
        if not os.path.isdir(dossier_star):
            print(f"[!] Dossier manquant pour {etoile} : {dossier_star}")
            nb_sans_fits += 1
            continue

        found_fits = False  # Booléen pour savoir si on a trouvé au moins un FITS

        # Recherche récursive dans dossier_star pour trouver un fichier FITS
        for root, _, files in os.walk(dossier_star):
            for file in files:
                if file.endswith('.fits') or file.endswith('.fit'):
                    chemin_fits = os.path.join(root, file)
                    try:
                        # Ouverture du fichier FITS avec astropy
                        with fits.open(chemin_fits) as hdul:
                            header = hdul[0].header  # Récupération du header du premier HDU

                            # Extraction des valeurs correspondant aux clés dans keywords
                            ligne = [etoile]  # Première colonne = nom de l’étoile
                            for key in keywords:
                                # Récupérer la valeur dans le header, 'N/A' si absente
                                val = header.get(keywords[key], 'N/A')
                                ligne.append(val)

                            # Écriture de la ligne dans le fichier CSV
                            writer.writerow(ligne)

                            print(f"[✓] {etoile} traité : {file}")
                            nb_ok += 1
                            found_fits = True
                            break  # On ne traite qu’un FITS par étoile

                    except Exception as e:
                        # En cas d’erreur à l’ouverture ou la lecture du FITS
                        print(f"[!] Erreur lecture FITS ({etoile}) : {e}")
                        nb_erreurs += 1
                        found_fits = True  # On considère que le fichier est trouvé même s’il y a erreur
                        break
            if found_fits:
                break

        # Si aucun FITS n’a été trouvé dans le dossier de l’étoile
        if not found_fits:
            print(f"[!] Aucun fichier FITS trouvé pour {etoile}")
            nb_sans_fits += 1

# ======= AFFICHAGE DU RÉSUMÉ =======

print("\n========== RÉSUMÉ DE L’EXÉCUTION ==========")
print(f"🔢 Total d’étoiles trouvées : {nb_total}")
print(f"✅ Observations traitées avec succès : {nb_ok}")
print(f"❌ Erreurs lors de la lecture : {nb_erreurs}")
print(f"🚫 Aucun FITS trouvé : {nb_sans_fits}")
print(f"📄 Log sauvegardé dans : {log_file}")
