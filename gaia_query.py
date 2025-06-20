#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:31:12 2024

@author: nbadolo
"""
"""
Pour recupérer certains paramètres de mes étoiles dans le catalogue gaia.  
"""

import pandas as pd
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u

# Liste des étoiles avec leurs coordonnées (RA, DEC)
# Exemple de liste : [(RA1, DEC1), (RA2, DEC2), ...]
stars_coords = [
    (10.684, 41.268),  # Exemple de coordonnées (RA, DEC) en degrés (RA en degrés, DEC en degrés)
    (83.822, -5.391),
    # Ajouter d'autres étoiles ici
]

# Paramètres à récupérer
columns = [
    'source_id', 'ra', 'dec', 'phot_g_mean_mag', 'parallax', 'radial_velocity', 
    'pmra', 'pmdec', 'teff_val', 'logg_val'
]

# Initialiser la connexion à Gaia
Gaia.MAIN_GAIA_SERVER = "https://gea.esac.esa.int/tap-server/tap"

# Fonction pour récupérer les données
def get_gaia_data(ra, dec, radius=1.0):
    # Requête TAP (Table Access Protocol) pour récupérer les données
    coords = SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
    
    # Requête à l'archive Gaia pour les sources proches de ces coordonnées
    query = f"""
    SELECT {', '.join(columns)}
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {coords.ra.deg}, {coords.dec.deg}, {radius})) = 1
    """
    
    # Exécuter la requête
    job = Gaia.launch_job(query)
    
    # Récupérer les résultats sous forme de table pandas
    result = job.get_results()
    
    return result

# Récupérer les données pour chaque étoile
star_data = []
for ra, dec in stars_coords:
    data = get_gaia_data(ra, dec)
    star_data.append(data)

# Fusionner les résultats dans un DataFrame pandas
all_star_data = pd.concat(star_data, ignore_index=True)

# Afficher les résultats
print(all_star_data)
