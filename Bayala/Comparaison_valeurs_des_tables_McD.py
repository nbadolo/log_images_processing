#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 14 16:58:29 2025

@author: nbadolo
"""

import pandas as pd
import matplotlib.pyplot as plt

# Définir le chemin principal
main_path = "/home/nbadolo/Bureau/Aymard/Tables/comparaison/"

# Charger le premier tableau (CSV)
df1 = pd.read_csv(main_path + "McD_simple_summary_table.csv")

# Harmoniser les identifiants Hipparcos dans df1 (enlever le préfixe 'HIP_')
df1['Hip_clean'] = df1['HIP_'].astype(str).str.replace('HIP_', '').str.strip()

# Charger le second tableau (CSV)
df2 = pd.read_csv(main_path + "McD.csv")

# Gérer les valeurs manquantes dans df2 avant conversion
# Option 1 : Supprimer les lignes avec des valeurs manquantes dans 'HIP'
# df2 = df2.dropna(subset=['HIP'])

# Option 2 : Remplacer les valeurs manquantes par une valeur par défaut (par exemple 0)
df2['HIP'] = df2['HIP'].fillna(0)

# Harmoniser les identifiants Hipparcos dans df2 (convertir en chaînes et enlever les décimales)
df2['Hip_clean'] = df2['HIP'].astype(int).astype(str).str.strip()

# Afficher les 5 premières lignes de la première colonne de chaque fichier après harmonisation
# print("Premières lignes de la première colonne du tableau 1 (df1) après harmonisation :")
# print(df1['Hip_clean'].head())  # Affiche la colonne harmonisée du DataFrame df1

# print("\nPremières lignes de la première colonne du tableau 2 (df2) après harmonisation :")
# print(df2['Hip_clean'].head())  # Affiche la colonne harmonisée du DataFrame df2

# Fusionner les deux DataFrames sur les identifiants nettoyés
merged = pd.merge(df1, df2, on='Hip_clean', suffixes=('_1', '_2'))

# Vérifier si la fusion a réussi
if merged.empty:
    print("Aucune correspondance trouvée entre les identifiants Hipparcos des deux tableaux.")
else:
    # Indices pour l'axe x
    indices = range(len(merged))

    # Liste des paramètres à comparer
    parametres = ['Distance', 'Teff', 'E_IR', 'LIR/L*','Lum']
    
    
    # # Tracer les comparaisons
    # for param in parametres:
    #     param_1 = f"{param}_1"
    #     param_2 = f"{param}_2"

    #     # Vérifier si les colonnes existent dans le DataFrame fusionné
    #     if param_1 in merged.columns and param_2 in merged.columns:
    #         plt.figure(figsize=(10, 5))
    #         plt.plot(indices, merged[param_1], label=f'{param} new calculation', marker='o')
    #         plt.plot(indices, merged[param_2], label=f'{param} original table', marker='s')
    #         plt.title(f'Comparaison de {param} pour les étoiles communes')
    #         plt.xlabel("Indice de l'étoile")
    #         plt.ylabel(param)
    #         plt.legend()
    #         plt.grid(True)
    #         plt.tight_layout()
    #         plt.show()
    #     else:
    #         print(f"Les colonnes {param_1} et/ou {param_2} sont absentes du DataFrame fusionné.")

    # Dictionnaire des unités pour les étiquettes
    unites = {
        'Distance': 'Distance (pc)',
        'Teff': 'Teff (K)',
        'E_IR': 'E_IR ',  # Vous pouvez ajuster si ce n'est pas des magnitudes
        'LIR/L*': 'LIR / L* ',
        'Lum' :'Lum'
    }

    # Tracer les comparaisons avec unités dans les labels
    for param in parametres:
        param_1 = f"{param}_1"
        param_2 = f"{param}_2"
    
        if param_1 in merged.columns and param_2 in merged.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(indices, merged[param_1], label=f'{param} new calculation', marker='o')
            plt.plot(indices, merged[param_2], label=f'{param} original table (McD)', marker='s')
            plt.title(f'Comparaison de {param} pour les étoiles communes')
            plt.xlabel("Indice de l'étoile")
            plt.ylabel(unites[param])
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print(f"Les colonnes {param_1} et/ou {param_2} sont absentes du DataFrame fusionné.")
