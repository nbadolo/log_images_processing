#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 14:25:28 2025

@author: nbadolo
"""

import os
import pandas as pd

# Filtres à vérifier sans préciser alone ou both
filtres_cibles = {
    "Cnt820": "alone",
    "I_PRIM": "alone",
    "R_PRIM": "alone",
    "Cnt820_Cnt748": "both",
    "I_PRIM_R_PRIM": "both"
}

def dossier_non_vide(path):
    """Retourne True si le dossier existe et contient des fichiers"""
    return os.path.isdir(path) and any(os.listdir(path))

def filtres_non_vides(star_path):
    """Retourne les filtres non vides pour une étoile"""
    filtres_present = []

    for filtre, sous_dossier in filtres_cibles.items():
        chemin = os.path.join(star_path, "psf", sous_dossier, filtre)
        if dossier_non_vide(chemin):
            filtres_present.append(filtre)

    return filtres_present

def analyser_dossier_etoiles(chemin_racine):
    """Analyse toutes les étoiles du dossier"""
    resultats = []

    for nom_etoile in os.listdir(chemin_racine):
        chemin_etoile = os.path.join(chemin_racine, nom_etoile)
        if not os.path.isdir(chemin_etoile):
            continue

        filtres = filtres_non_vides(chemin_etoile)
        resultats.append({
            "Nom_Étoile": nom_etoile,
            "Filtres_Présents": ", ".join(filtres) if filtres else "Aucun"
        })

    return pd.DataFrame(resultats)

# === À personnaliser ===
chemin_racine = "/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log_old/"  # ⇦ à modifier
nom_fichier_csv = "etoiles_filtres_non_vides.csv"

# === Traitement ===
df = analyser_dossier_etoiles(chemin_racine)
df.to_csv(nom_fichier_csv, index=False, encoding="utf-8-sig")
print(df)
print(f"\n✅ Analyse terminée : {len(df)} étoiles traitées.")
print(f"📄 Résultat enregistré dans : {nom_fichier_csv}")
