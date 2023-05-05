#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb 24 14:34:30 2022

@author: nbadolo
"""


import os 
import log_images # pour toutes les étoiles avec une psf
import log_images_wp # pour toutes les étoiles sans psf
import log_deconv #  pour tous les objets resolus à deconvoluer
import log_gauss_full_ellips_fitting # pour tous les objets résolus deconvolués
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
    log_images_wp.log_image(lst_star[i], 'alone')
    log_images_wp.log_image(lst_star[i], 'both')   
   
#     else :  # Affiche toutes les étoiles qui n'ont pas de psf
#         if i == 0 or i == 3 or i == 14 or i == 19 or i == 21 or i == 22:
#             log_agb_images_wp.log_image(lst_star[i], 'alone')
#             log_agb_images_wp.log_image(lst_star[i], 'both')
       
#         else: # Affiche toutes les étoiles normales
#             log_agb_images.log_image(lst_star[i], 'alone')
#             log_agb_images.log_image(lst_star[i], 'both')
           
#     if i == 9 or i == 10 or i == 13 or i == 15 or i == 24 : # affiche la deconvolution des objets resolus qui ont une psf
#         deconv.log_image(lst_star[i], 'alone')
#         deconv.log_image(lst_star[i], 'both')
    
#     if i == 9 or i == 10 or i == 13 or i == 14  or i == 15 or i == 24 : # affiche les fits gaussiens des objets resolus
     
      
#         gauss_ellips_full_automatic_log_fitting.log_image(lst_star[i], 'alone')
#         gauss_ellips_full_automatic_log_fitting.log_image(lst_star[i], 'both')
        
