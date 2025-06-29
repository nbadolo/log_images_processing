"""
Script d'estimation de la masse stellaire à partir d'une table d'étoiles.

nbadolo
Date : 24/06/2025

Description : 
Ce script lit un fichier CSV contenant des données photométriques d'étoiles,
en particulier l'indice de couleur B-V, puis :

1. Estime la température effective (Teff) via la formule empirique de Ballesteros (2012),
   basée sur l'indice de couleur B-V.

2. Calcule la luminosité relative (L / L_sun) en supposant que
   la luminosité suit la loi de Stefan-Boltzmann et que le rayon est
   similaire au rayon solaire (hypothèse simplificatrice) :
   
   L / L_sun = (Teff / Teff_sun)^4

3. Estime la masse stellaire (M / M_sun) en utilisant la relation masse-luminosité
   empirique sur plusieurs plages de masse, avec un exposant α variable
   selon la luminosité (dérivé de Kippenhahn & Weigert 1990, Cox 2000) :

       M = (L)^(1/α)

4. Sauvegarde un nouveau fichier CSV contenant les colonnes d'origine
   ainsi que les colonnes calculées : Teff_K, Luminosite_Lsun, Masse_Msun.

---

Utilisation :
- Adapter la variable `chemin_dossier` et `nom_fichier` selon ton organisation.
- Le fichier CSV doit contenir au minimum la colonne 'B-V'.
"""

import pandas as pd
import os

# --- Fonctions ---

def estimer_teff(BV):
    """
    Estime la température effective Teff (en Kelvin) d'une étoile
    à partir de son indice de couleur B-V selon Ballesteros (2012).

    Formule :
        Teff = 4600 * (1 / (0.92 * (B-V) + 1.7) + 1 / (0.92 * (B-V) + 0.62))

    Plage de validité :
        -0.4 < B-V < 2.0
    
    Source :
        Ballesteros, F. J. (2012), "New insights into black bodies",
        EPL (Europhysics Letters), 97(3), 34008.
    """
    return 4600 * (1 / (0.92 * BV + 1.7) + 1 / (0.92 * BV + 0.62))


def estimer_luminosite(Teff):
    """
    Estime la luminosité relative (L / L_sun) en supposant une loi
    de Stefan-Boltzmann avec rayon constant (approximation).

    Formule :
        L / L_sun = (Teff / Teff_sun)^4
    
    où Teff_sun ≈ 5778 K est la température effective du Soleil.

    Limitation :
        Cette approximation néglige les variations de rayon stellaire.
    """
    Teff_sun = 5778  # Température effective solaire en Kelvin
    return (Teff / Teff_sun) ** 4


def estimer_masse_depuis_luminosite(L):
    """
    Estime la masse stellaire (M / M_sun) en fonction de la luminosité
    selon la relation masse-luminosité adaptée à plusieurs plages de masse.

    Formule générale :
        M = L^{1/α}
    
    avec α dépendant de L (et donc implicitement de M) selon :

    - Pour L < 0.03 : α = 2.3 (naines rouges)
    - Pour 0.03 ≤ L < 16 : α = 4.0 (étoiles type solaire)
    - Pour 16 ≤ L < 32000 : α = 3.5 (étoiles massives)
    - Pour L ≥ 32000 : α = 2.0 (supergéantes et très massives)

    Sources :
        - Kippenhahn & Weigert (1990), "Stellar Structure and Evolution"
        - Cox (2000), "Allen’s Astrophysical Quantities"
    """
    if L < 0.03:
        alpha = 2.3
    elif L < 16:
        alpha = 4.0
    elif L < 32000:
        alpha = 3.5
    else:
        alpha = 2.0
    return L ** (1 / alpha)


# --- Paramètres utilisateur ---

chemin_dossier = "/chemin/vers/le/dossier"  # <-- Modifie ici le chemin vers ton dossier
nom_fichier = "etoiles_M33.csv"             # <-- Nom du fichier CSV dans ce dossier

# Construction du chemin complet
chemin_fichier = os.path.join(chemin_dossier, nom_fichier)

# --- Chargement des données ---

df = pd.read_csv(chemin_fichier)

# Vérification de la présence de la colonne 'B-V'
if "B-V" not in df.columns:
    raise ValueError("La colonne 'B-V' est absente du fichier CSV. Veuillez vérifier le nom et le format.")

# --- Calculs ---

# Estimation de la température effective
df["Teff_K"] = df["B-V"].apply(estimer_teff)

# Estimation de la luminosité relative L/L_sun
df["Luminosite_Lsun"] = df["Teff_K"].apply(estimer_luminosite)

# Estimation de la masse stellaire M/M_sun
df["Masse_Msun"] = df["Luminosite_Lsun"].apply(estimer_masse_depuis_luminosite)

# --- Sauvegarde des résultats ---

nom_fichier_sortie = "etoiles_M33_avec_masse.csv"
chemin_fichier_sortie = os.path.join(chemin_dossier, nom_fichier_sortie)

df.to_csv(chemin_fichier_sortie, index=False)

print(f"Traitement terminé. Fichier enregistré ici : {chemin_fichier_sortie}")
