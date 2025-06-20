#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 23:42:49 2025

@author: nbadolo
"""
"""
Code pour faire le cross-matche deux tables.
               
                code okay !!!
"""
import pandas as pd
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
import numpy as np

# === 1. Charger les deux fichiers CSV ===

# Table 1 : Galaxies à étudier (coordonnées RA/DEC en sexagésimal)
table1 = pd.read_csv("/home/nbadolo/Bureau/Bayala/extractions/spirales_classe_G.csv")

# Table 2 : Catalogue complet avec RA/DEC en degrés
table2 = pd.read_csv("/home/nbadolo/Bureau/Bayala/extractions/galaxies_spirales_morpho_dr17.csv")

# === 2. Conversion des coordonnées RA/DEC de table1 en degrés décimaux ===

# RA est au format 'hh mm ss.ss' → convertir en degrés
ra_deg = Angle(table1['RA'], unit=u.hourangle).degree

# DEC est au format '±dd mm ss.s' → convertir en degrés
dec_deg = Angle(table1['DEC'], unit=u.deg).degree

# Ajouter les colonnes converties à table1
table1['RA_deg'] = ra_deg
table1['DEC_deg'] = dec_deg

# === 3. Vérification des valeurs aberrantes (DEC hors plage -90 à +90) ===

invalid_mask = (dec_deg < -90) | (dec_deg > 90)

print(f"Nombre de lignes avec une DEC invalide : {invalid_mask.sum()}")

if invalid_mask.sum() > 0:
    print("Voici les lignes avec une DEC hors de la plage [-90, 90] :")
    print(table1[invalid_mask])
else:
    print("✅ Toutes les valeurs de DEC sont valides.")

# === 4. Créer les objets SkyCoord pour le cross-match ===

# Table 1 : coordonnées en degrés
coords1 = SkyCoord(ra=table1['RA_deg'] * u.deg, dec=table1['DEC_deg'] * u.deg)

# Table 2 : coordonnées déjà en degrés
coords2 = SkyCoord(ra=table2['RA'] * u.deg, dec=table2['DEC'] * u.deg)

# === 5. Cross-match (associer chaque galaxie de table1 à la plus proche de table2) ===

# match_to_catalog_sky retourne les indices, distances et indicateurs inutilisés ici
idx, d2d, _ = coords1.match_to_catalog_sky(coords2)

# Tolérance de match : 1 arcseconde
tolerance = 1 * u.arcsec
matched_mask = d2d < tolerance

# Galaxies de table1 avec un match dans table2
matched_table1 = table1[matched_mask].reset_index(drop=True)
matched_table2 = table2.iloc[idx[matched_mask]].reset_index(drop=True)

# === 6. Fusionner les informations dans une nouvelle table ===

# Ici on garde les colonnes de table2 (mais on peut aussi ajouter des colonnes de table1)
final_table = matched_table2.copy()

# Exemple : si vous voulez inclure les RA/DEC de table1 aussi :
# final_table['RA_source'] = matched_table1['RA']
# final_table['DEC_source'] = matched_table1['DEC']

# === 7. Sauvegarder la table croisée ===

output_path = "/home/nbadolo/Bureau/Bayala/extractions/galaxies_croiseesG.csv"
final_table.to_csv(output_path, index=False)

# === 8. Affichage d’un aperçu ===

print("✅ Cross-match terminé. Voici les 5 premières lignes :")
print(final_table.head())
print('le nombre de ligne du tableau final =' + str(len(final_table)))