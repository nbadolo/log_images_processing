#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 23:19:43 2025

@author: nbadolo
"""

import pandas as pd

# === 1. Charger le fichier CSV ===
file_path = "/home/nbadolo/Bureau/Bayala/extractions/spirales_fusionnees_redressees.csv"
df = pd.read_csv(file_path)

# === 2. Nettoyer la colonne 'galaxy' ===
# - Retirer le "J" au début
# - Supprimer tous les espaces pour unifier le format
df['RA_DEC'] = df['galaxy'].str.replace('^J', '', regex=True)
df['RA_DEC'] = df['RA_DEC'].str.replace(r'\s+', '', regex=True)

# === 3. Extraire RA et DEC (format hhmmss.ss et ±ddmmss.s) ===
df[['RA', 'DEC']] = df['RA_DEC'].str.extract(r'(\d{6}\.\d+)([+\-]\d{6}\.\d+)')

# === 4. Fonctions de reformatage vers hh mm ss.ss et ±dd mm ss.s ===
def format_ra(ra_str):
    """Transforme hhmmss.ss -> hh mm ss.ss"""
    h = ra_str[0:2]
    m = ra_str[2:4]
    s = ra_str[4:]
    return f"{h} {m} {s}"

def format_dec(dec_str):
    """Transforme ±ddmmss.s -> ±dd mm ss.s"""
    sign = dec_str[0]
    d = dec_str[1:3]
    m = dec_str[3:5]
    s = dec_str[5:]
    return f"{sign}{d} {m} {s}"

# === 5. Appliquer le formatage RA/DEC lisible ===
df['RA'] = df['RA'].apply(format_ra)
df['DEC'] = df['DEC'].apply(format_dec)
# Vérification (facultative)
print(df[['RA_DEC', 'RA', 'DEC']].head(10))

# === 6. Supprimer les colonnes intermédiaires ===
df = df.drop(columns=['galaxy', 'RA_DEC'])

# === 7. Sauvegarder dans un nouveau fichier CSV ===
output_path = "/home/nbadolo/Bureau/Bayala/extractions/spirales_fusionnees_redressees_cor.csv"
df.to_csv(output_path, index=False)

print("✅ Fichier traité et enregistré avec RA et DEC formatés.")


# Charger le fichier final
df = pd.read_csv(output_path)

# Filtrer les lignes où la colonne 'classe' vaut exactement 'F'
df_f = df[df['class'] == 'F']

# Vérifier combien de lignes ont été extraites
print(f"{len(df_f)} lignes extraites avec classe == 'F'.")

# Sauvegarder dans un nouveau fichier CSV
output_f_path = "/home/nbadolo/Bureau/Bayala/extractions/spirales_classe_F.csv"
df_f.to_csv(output_f_path, index=False)

print("✅ Fichier des galaxies de classe F sauvegardé.")

# Filtrer les lignes où la colonne 'classe' vaut exactement 'G' grand design
df_f = df[df['class'] == 'G']

# Vérifier combien de lignes ont été extraites
print(f"{len(df_f)} lignes extraites avec classe == 'F'.")

# Sauvegarder dans un nouveau fichier CSV
output_f_path = "/home/nbadolo/Bureau/Bayala/extractions/spirales_classe_G.csv"
df_f.to_csv(output_f_path, index=False)

print("✅ Fichier des galaxies de classe F sauvegardé.")
