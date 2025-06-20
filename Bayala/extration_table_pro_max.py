#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 16:02:16 2025

@author: nbadolo
"""

"""
Extraction d'une table contenue  dans un fichier pdf. Si la table est repartie sur plusieurs pages,
le code ectrait la partie contenue dans chaque page et fusionne pour enfaire une seule page. 

                                                CODE OKAY !!
"""


import camelot
import pandas as pd

file_path = "/home/nbadolo/Bureau/Bayala/paper.pdf"
all_tables = []

def make_unique_columns(cols):
    seen = {}
    new_cols = []
    for col in cols:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    return new_cols

for page in range(14, 19):
    print(f"Traitement de la page {page}...")
    tables = camelot.read_pdf(file_path, pages=str(page), flavor='stream')

    if len(tables) > 0:
        for i, table in enumerate(tables):
            df = table.df

            # Sauter les tableaux vides ou trop petits
            if df.shape[0] < 2 or df.shape[1] < 2:
                continue

            header_found = False  # Variable pour savoir si un en-tête est trouvé

            # Parcourir les lignes de la table pour trouver "Galaxy" dans la première cellule
            for idx, row in df.iterrows():
                if any("galaxy" in str(cell) for cell in row):
                    # Utiliser cette ligne comme en-tête
                    header = make_unique_columns(row)
                    df.columns = header
                    df = df[idx + 1:].reset_index(drop=True)  # Enlever la ligne d'en-tête et réinitialiser les index
                    header_found = True
                    break  # Sortir de la boucle une fois l'en-tête trouvé

            if not header_found:
                print(f"-> Table ignorée (pas d'en-tête 'Galaxy') à la page {page}, table {i}")
                continue  # Passer à la prochaine table

            # Filtrer les lignes indésirables
            df = df[~df[header[0]].str.contains("Table A1", na=False)]

            # Ajouter info sur la page
            df["page"] = page

            # Sauvegarde individuelle
            output = f"/home/nbadolo/Bureau/Bayala/spirales_{page}_table{i}.csv"
            df.to_csv(output, index=False)

            all_tables.append(df)
    else:
        print(f"Aucune table trouvée à la page {page}.")

# Fusion de toutes les tables valides
if all_tables:
    merged_df = pd.concat(all_tables, ignore_index=True)
    merged_df.to_csv("/home/nbadolo/Bureau/Bayala/spirales_fusionnees.csv", index=False)
    print("Fusion réussie dans 'spirales_fusionnees.csv'")
else:
    print("Aucune table à fusionner.")
