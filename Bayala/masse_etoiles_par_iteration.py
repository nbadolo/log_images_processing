"""
Script d'estimation itérative de la masse stellaire depuis B-V

nbadolo
Date :24/06/2025

Description :
Ce script lit un fichier CSV contenant une table d'étoiles avec l'indice
de couleur B-V. Pour chaque étoile, il estime :

1. La température effective (Teff) via la formule empirique de Ballesteros (2012).
2. La luminosité relative (L / L_sun) en utilisant la loi de Stefan-Boltzmann
   sous hypothèse simplificatrice de rayon constant.
3. La masse stellaire via une méthode itérative :
   - Initialisation avec la luminosité estimée à partir de Teff.
   - Calcul de la masse avec la relation masse-luminosité (exposant α variable).
   - Recalcul de la luminosité via la masse estimée.
   - Répétition jusqu'à convergence sur la masse.

La méthode améliore la cohérence entre masse et luminosité.

Le résultat est sauvegardé dans un nouveau fichier CSV.

---

Utilisation :
- Modifier les variables `chemin_dossier` et `nom_fichier` pour correspondre
  à ton environnement.
- Le fichier CSV doit contenir une colonne 'B-V' (indice de couleur).
"""

import pandas as pd
import os

def estimer_teff(BV):
    """
    Estime la température effective Teff (en K) à partir de l'indice B-V.

    Formule (Ballesteros 2012) :
    Teff = 4600 * [1/(0.92*BV + 1.7) + 1/(0.92*BV + 0.62)]

    Plage : -0.4 < B-V < 2.0

    Source :
    Ballesteros, F. J. (2012), "New insights into black bodies",
    EPL (Europhysics Letters), 97(3), 34008.
    """
    return 4600 * (1 / (0.92 * BV + 1.7) + 1 / (0.92 * BV + 0.62))


def estimer_masse_depuis_luminosite(L):
    """
    Estime la masse M (en M_sun) à partir de la luminosité L (en L_sun)
    en utilisant la relation masse-luminosité à plusieurs régimes :

    - L < 0.03      : alpha = 2.3 (naines rouges)
    - 0.03 ≤ L < 16 : alpha = 4.0 (étoiles de masse solaire)
    - 16 ≤ L < 32000: alpha = 3.5 (étoiles massives)
    - L ≥ 32000     : alpha = 2.0 (supergéantes)

    Formule générale :
    M = L^{1/alpha}

    Sources :
    Kippenhahn & Weigert (1990), "Stellar Structure and Evolution"
    Cox (2000), "Allen’s Astrophysical Quantities"
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


def masse_iterative(BV, tol=1e-5, max_iter=100):
    """
    Estime la masse stellaire de manière itérative à partir de B-V.

    Méthode :
    - Calcul initial de Teff via B-V.
    - Estimation initiale de la luminosité L via Teff.
    - Estimation initiale de la masse M via la relation masse-luminosité.
    - Boucle : recalcul de L à partir de M, recalcul de M à partir de L.
    - Arrêt si différence entre masses successive < tol ou max iterations.

    Retourne :
    - Masse stellaire (M/M_sun)
    - Luminosité relative (L/L_sun)
    - Température effective (K)
    """
    T_sun = 5778  # Température effective solaire (K)
    Teff = estimer_teff(BV)
    L = (Teff / T_sun) ** 4  # Luminosité initiale

    M = estimer_masse_depuis_luminosite(L)

    for _ in range(max_iter):
        # Choix de l'exposant alpha selon luminosité
        if L < 0.03:
            alpha = 2.3
        elif L < 16:
            alpha = 4.0
        elif L < 32000:
            alpha = 3.5
        else:
            alpha = 2.0

        M_new = L ** (1 / alpha)  # Masse mise à jour
        L_new = M_new ** alpha    # Luminosité mise à jour

        if abs(M_new - M) < tol:
            break  # Convergence atteinte

        M, L = M_new, L_new

    return M_new, L_new, Teff


# --- Partie principale ---

# Chemin vers le dossier contenant le fichier CSV (à modifier)
chemin_dossier = "/chemin/vers/le/dossier"

# Nom du fichier CSV d'entrée
nom_fichier = "etoiles_M33.csv"

# Construction du chemin complet
chemin_fichier = os.path.join(chemin_dossier, nom_fichier)

# Chargement des données
df = pd.read_csv(chemin_fichier)

# Vérification de la colonne 'B-V'
if "B-V" not in df.columns:
    raise ValueError("La colonne 'B-V' est manquante dans le fichier CSV.")

# Application de la fonction itérative sur chaque étoile
resultats = df["B-V"].apply(masse_iterative)

# Extraction des résultats dans des colonnes séparées
df["Masse_Msun"], df["Luminosite_Lsun"], df["Teff_K"] = zip(*resultats)

# Nom du fichier de sortie
nom_fichier_sortie = "etoiles_M33_avec_masse_iterative.csv"
chemin_fichier_sortie = os.path.join(chemin_dossier, nom_fichier_sortie)

# Sauvegarde du dataframe enrichi
df.to_csv(chemin_fichier_sortie, index=False)

print(f"Traitement terminé. Fichier sauvegardé ici : {chemin_fichier_sortie}")
