#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb 24 14:34:30 2022

@author: nbadolo
"""


import os 
import log_agb_images
import log_agb_images_wp
import log_agb_images_wp2
import log_irc_images
#import deconv
#import gauss_ellips_full_automatic_log_fitting


#%%
# codes call

lst_star = os.listdir('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/')
lst_star.sort()
lst_len = len(lst_star)
#print(lst_star)

for i in range(lst_len): 
#     if i == 5 :    # affiche uniquement irc_10420
#         log_irc_images.log_image(lst_star[i], 'alone')
#         log_irc_images.log_image(lst_star[i], 'both')   
   
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
        
     if i == 8 :  # pour le deuxieme lot d'étoile. Elle sont dans le dossier News_stars lui meme etant dans log au rang 8
          lst_star2 = os.listdir('/home/nbadolo/Bureau/Aymard/Donnees_sph/log/'+ lst_star[i]+ '/')
          lst_star2.sort()
          lst_len2 = len(lst_star2)
          #print(lst_star2)
          for j in range(lst_len2):
              
              log_agb_images_wp2.log_image(lst_star2[j], 'alone')
              log_agb_images_wp2.log_image(lst_star2[j], 'both')
