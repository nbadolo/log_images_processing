#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:32:04 2025

@author: nbadolo
"""

import os

def sauvegarder_noms_etoiles(repertoire_etoiles, repertoire_sortie, fichier_sortie):
    # Vérifier si le répertoire de sortie existe, sinon le créer
    os.makedirs(repertoire_sortie, exist_ok=True)
    
    # Récupérer la liste des fichiers dans le répertoire des étoiles
    fichiers = os.listdir(repertoire_etoiles)
    
    # Construire le chemin complet du fichier de sortie
    chemin_fichier_sortie = os.path.join(repertoire_sortie, fichier_sortie)

    # Écrire les noms des fichiers dans le fichier texte
    with open(chemin_fichier_sortie, 'w', encoding='utf-8') as f:
        for fichier in fichiers:
            f.write(f"{fichier}\n")

    print(f"Liste des étoiles enregistrée dans {chemin_fichier_sortie}")

# Exemple d'utilisation
dir_name = 'large_log_+'
repertoire_etoiles = '/home/nbadolo/Bureau/Aymard/Donnees_sph/'+ dir_name+'/'  # Remplace par ton chemin
repertoire_sortie = '/home/nbadolo/Bureau/Aymard/Donnees_sph/sphere_files/txt_files/'  # Répertoire où enregistrer le fichier
fichier_sortie = "liste_etoiles_de_"+dir_name+".txt"

sauvegarder_noms_etoiles(repertoire_etoiles, repertoire_sortie, fichier_sortie)
