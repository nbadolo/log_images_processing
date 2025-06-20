#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 10:29:16 2025

@author: nbadolo
"""


import os
import shutil

def vider_et_preparer_sous_dossiers(dossier_input):
    """
    Vide le contenu de chaque sous-dossier dans le dossier `dossier_input`,
    puis crée les sous-dossiers 'Intensity' et 'Pol_intensity' dans chacun.
    
    :param dossier_input: Chemin vers le dossier principal contenant les sous-dossiers d'étoiles
    """
    for nom_sous_dossier in os.listdir(dossier_input):
        chemin_sous_dossier = os.path.join(dossier_input, nom_sous_dossier)
        
        if os.path.isdir(chemin_sous_dossier):
            # 🔥 Suppression du contenu existant
            for item in os.listdir(chemin_sous_dossier):
                chemin_item = os.path.join(chemin_sous_dossier, item)
                try:
                    if os.path.isfile(chemin_item) or os.path.islink(chemin_item):
                        os.remove(chemin_item)
                    elif os.path.isdir(chemin_item):
                        shutil.rmtree(chemin_item)
                    print(f"🗑️ Supprimé : {chemin_item}")
                except Exception as e:
                    print(f"❌ Erreur avec {chemin_item} : {e}")

            # 📁 Création des sous-dossiers 'Intensity' et 'Pol_intensity'
            for nom_nouveau in ['Intensity', 'Pol_intensity']:
                chemin_nouveau = os.path.join(chemin_sous_dossier, nom_nouveau)
                try:
                    os.makedirs(chemin_nouveau, exist_ok=True)
                    print(f"📂 Créé : {chemin_nouveau}")
                except Exception as e:
                    print(f"❌ Erreur lors de la création de {chemin_nouveau} : {e}")


chemin1 = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian/Input/large_log_+/"
chemin2 = "/home/nbadolo/Bureau/Aymard/Donnees_sph/Gaussian/Input/resolved_log/"
vider_et_preparer_sous_dossiers(chemin1)
vider_et_preparer_sous_dossiers(chemin2)