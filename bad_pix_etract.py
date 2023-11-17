#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:55:42 2023

@author: nbadolo
"""


from astropy.io import fits
import numpy as np

# Charger le cube FITS
hdulist = fits.open('votre_cube.fits')
cube = hdulist[0].data

# Supprimer les pixels chauds en soustrayant la médiane de chaque couche
for i in range(cube.shape[0]):
    cube[i] -= np.nanmedian(cube[i])
#Normaliser chaque couche du cube en divisant par la valeur maximale
maxi = np.max(cube, axis=0)
for i in range(cube.shape[0]):
    cube[i] /= maxi

# Ajuster les valeurs pour qu'elles soient un peu plus lumineuses que l'arrière-plan
cube = cube * 0.9 + 0.1

# Mettre à zéro les valeurs négatives
cube[cube < 0] = 0

# Limiter les valeurs supérieures à 1 à 1
cube[cube > 1] = 1

# Appliquer une correction gamma en prenant la racine carrée
cube = np.sqrt(cube)

# Étendre l'échelle des valeurs de 0-255
cube *= 255

# Sauvegarder le cube FITS modifié
hdulist[0].data = cube
hdulist.writeto('cube_modifie.fits', overwrite=True)
hdulist.close()