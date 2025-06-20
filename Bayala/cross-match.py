#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 13:21:11 2025

@author: nbadolo
"""


import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
import numpy as np
# === 1. Lire les deux fichiers CSV ===

# Table 1 : liste des galaxies à étudier (RA/DEC en sexagésimal)
table1 = pd.read_csv("/home/nbadolo/Bureau/Bayala/extractions/spirales_classe_F.csv")

# Affiche les valeurs aberrantes de DEC
dec_values = table1['DEC']
invalid_mask = (dec_values < -90) | (dec_values > 90)

# Afficher le nombre de lignes invalides
print(f"Nombre de lignes avec une DEC invalide : {invalid_mask.sum()}")

# Afficher les lignes problématiques
if invalid_mask.sum() > 0:
    print("Voici les lignes avec une DEC hors de la plage [-90, 90] :")
    print(table1[invalid_mask])
else:
    print("Toutes les valeurs de DEC sont valides.")
    
# Table 2 : catalogue complet (RA/DEC en degrés + paramètres supplémentaires)
table2 = pd.read_csv("/home/nbadolo/Bureau/Bayala/extractions/galaxies_spirales_morpho_dr17.csv")

# === 2. Convertir les coordonnées de la table 1 en objets SkyCoord ===

# Astropy attend des strings du type 'HH:MM:SS' et 'DD:MM:SS'
coords1 = SkyCoord(ra=table1['RA'], dec=table1['DEC'], unit=(u.hourangle, u.deg))

# Convertir les RA/DEC de la table 2 en SkyCoord (en degrés)
coords2 = SkyCoord(ra=table2['RA']*u.deg, dec=table2['DEC']*u.deg)

# === 3. Cross-match : associer chaque galaxie de table1 à la plus proche dans table2 ===

# match_to_catalog_sky renvoie l'indice du match le plus proche, la distance angulaire et 3e valeur inutilisée ici
idx, d2d, _ = coords1.match_to_catalog_sky(coords2)

# Tolérance : ici 1 arcseconde
tolerance = 1 * u.arcsec
matched_mask = d2d < tolerance

# Filtrage : galaxies de table1 qui ont un match dans table2
matched_table1 = table1[matched_mask].reset_index(drop=True)
matched_table2 = table2.iloc[idx[matched_mask]].reset_index(drop=True)

# === 4. Créer la 3e table contenant les paramètres de table2 pour les galaxies communes ===

# Vous pouvez ajouter d'autres colonnes si besoin
final_table = matched_table2.copy()

# Ajouter les colonnes d'identifiants si elles existent (nom MANGA, Simbad, etc.)
# Vérifiez que ces colonnes existent dans votre fichier (sinon commentez-les)
# Exemple :
# final_table = final_table[['manga_id', 'simbad_name', 'RA', 'DEC', ...]]

# === 5. Enregistrer la table résultante en CSV ===
final_table.to_csv("galaxies_croisees.csv", index=False)

# === 6. Afficher les 5 premières lignes dans Spyder (ou toute console IPython) ===
print(final_table.head())
