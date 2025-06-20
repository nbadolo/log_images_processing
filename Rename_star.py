#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 10:28:23 2025

@author: nbadolo
"""

"""
Code pour corriger le nom de mes étoiles 
"""

import os
from astropy.io import fits


"                                         "
"                                    "
"                                   "
# star=log_image('V854_Cen', 'alone')  
# star=log_image('V854_Cen', 'both')  



"alone ==  Cnt820  CntHa  I_PRIM  N_R  R_PRIM  V  VBB"

"both == Cnt820_Cnt748  CntHa_B_Ha  CntHa_N_Ha  I_PRIM_R_PRIM  V_Cnt748  V_N_R"

# Chemin vers le dossier de l'étoile
star_name = "V854_Cen"
mod = "both"
filt ="I_PRIM_R_PRIM"
#chemin_dossier = "/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log_+/"+star_name + "/"+ "star"+"/"+mod+"/"+ filt
chemin_dossier = "/home/nbadolo/Bureau/Aymard/Donnees_sph/First/"+star_name + "/"+ "star"+"/"+mod+"/"+ filt

# Nouveau nom à mettre dans l'en-tête
nouveau_nom = "V854 Cen"

# Mot-clé de l'en-tête à modifier
cle_nom_etoile = "OBJECT"  # Change si nécessaire

# Parcours de tous les fichiers .fits dans le dossier
for fichier in os.listdir(chemin_dossier):
    if fichier.endswith(".fits"):
        chemin_fits = os.path.join(chemin_dossier, fichier)
        with fits.open(chemin_fits, mode='update') as hdul:
            header = hdul[0].header
            if cle_nom_etoile in header:
                ancien_nom = header[cle_nom_etoile]
                header[cle_nom_etoile] = nouveau_nom
                print(f"{fichier} : '{ancien_nom}' remplacé par '{nouveau_nom}'")
            else:
                print(f"{fichier} : clé '{cle_nom_etoile}' non trouvée dans l'en-tête.")
