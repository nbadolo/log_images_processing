#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb 24 14:34:30 2022

@author: nbadolo
"""


import os 
import log_images # pour toutes les étoiles avec une psf. le code est  okay
import log_images_wp #pour toutes les étoiles sans psf. le code est okay
import log_deconv #pour tous les objets resolus à deconvoluer
import log_gauss_full_ellips_fitting #pour tous les objets résolus deconvolués le code est okay
import log_radial_profile_at_given_orientation_star_psf # pour les objets resolus
from natsort import natsorted


# codes calling
lst_star = sorted(os.listdir('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'))
#lst_star = natsorted('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/')

#lst_star.sort()
lst_len = len(lst_star)
print(lst_star)
print(lst_len)

for i in range(lst_len): # affiche  les images des étoiles sans leur psf 
    #if i != 14 :
    print(lst_star[i])    
    # log_images_wp.log_image(lst_star[i], 'alone')
    # log_images_wp.log_image(lst_star[i], 'both')   
    # log_images.log_image(lst_star[i], 'alone')
    # log_images.log_image(lst_star[i], 'both')
    
    # log_gauss_full_ellips_fitting.log_image(lst_star[i], 'alone')
    # log_gauss_full_ellips_fitting.log_image(lst_star[i], 'both')
    
    log_radial_profile_at_given_orientation_star_psf.log_image(lst_star[i], 'alone')
    log_radial_profile_at_given_orientation_star_psf.log_image(lst_star[i], 'both')
