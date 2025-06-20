#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 13:39:19 2025

@author: nbadolo
"""

import camelot
import os
# Charger le PDF
file_path = "/home/nbadolo/Bureau/Bayala/paper.pdf"


# Extraction des tables du PDF
tables = camelot.read_pdf(file_path, pages='14-18', flavor='stream')

# Afficher le nombre de tables extraites
print(f"Nombre de tables extraites : {len(tables)}")

# Afficher la première table en DataFrame
if len(tables) > 0:
    print(tables[0].df)  # Utilisez 'df' pour accéder à la DataFrame

tables[0].to_csv('new_floculences.csv')
