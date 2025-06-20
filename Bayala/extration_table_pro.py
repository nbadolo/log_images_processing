#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:23:53 2025

@author: nbadolo
"""

import camelot
# Charger le PDF
file_path = "/home/nbadolo/Bureau/Bayala/paper.pdf"
for page in range(14,19):
    
# Extraction des tables du PDF
    tables = camelot.read_pdf(file_path, pages=str(page), flavor='stream')

    #tables = camelot.read_pdf(file_path, pages='14-18', flavor='stream')
    
    
    # Si une table est extraite, affichez-la et nettoyez-la
    if len(tables) > 0:
        df = tables[0].df
    
        # Exemple de suppression des lignes indésirables (par exemple, contenant des en-têtes ou légendes)
        df_cleaned = df[~df[0].str.contains("Table A1", na=False)]  # Ici, on filtre les lignes contenant "Table A1"
    
        # Afficher le DataFrame nettoyé
        print(df_cleaned)
    
        # Sauvegarder le DataFrame nettoyé dans un fichier CSV
        output = f"/home/nbadolo/Bureau/Bayala/spirales_{page}.csv"

        df_cleaned.to_csv(output, index=False)
