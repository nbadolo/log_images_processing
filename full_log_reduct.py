#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Feb 24 14:34:30 2022

@author: nbadolo
"""
"""
# Ceci est le code prinicipal qui fait rouler tous les autres sur tout l'échantillon. 

"""

import os 
import log_images # pour toutes les étoiles avec une psf. le code est  okay
import log_images_wp #pour toutes les étoiles sans psf. le code est okay
import log_deconv #pour tous les objets resolus à deconvoluer
import log_gauss_full_ellips_fitting #pour tous les objets résolus deconvolués le code est okay
import log_radial_profile_at_given_orientation_star_psf # pour les objets resolus. code en ecriture
from natsort import natsorted

#%%
# codes calling
lst_star = sorted(os.listdir('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'))
#lst_star = natsorted('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/')

modes = ['alone','both']

# fichier .txt pour recuperer les étoiles qui pour lesquelles le code affiffche un erreur
txt_folder = 'sphere_txt_file'
file_path = '/home/nbadolo/Bureau/Aymard/Donnees_sph/' + txt_folder + '/'
file_name = 'not_run_star_lst.txt'
not_run_star_lst = open("{}/{}".format(file_path, file_name), "w")
not_run_star_lst.write("{}, {}, {}\n".format('Star_name', 'Mode', 'ErrorType'))
#lst_star.sort()
lst_len = len(lst_star)
print(lst_star)
print(lst_len)

for i in range(lst_len): # affiche  les images des étoiles sans leur psf 
 
    print(lst_star[i])
    for mod in modes : 
        try :
            log_radial_profile_at_given_orientation_star_psf.log_image(lst_star[i], mod)
            #log_radial_profile_at_given_orientation_star_psf.log_image(lst_star[i], 'both')
        except Exception as e:
            not_run_star_lst.write("{}, {}, {}\n".format(lst_star[i], mod, e))
            pass
    
        try :
            log_images_wp.log_image(lst_star[i], mod)
            log_images.log_image(lst_star[i], mod)
            log_gauss_full_ellips_fitting(lst_star[i], mod)
        except :
            pass
#%%
   #if i != 14 :
       
   # log_images_wp.log_image(lst_star[i], 'alone')
   # log_images_wp.log_image(lst_star[i], 'both')   
   # log_images.log_image(lst_star[i], 'alone')
   # log_images.log_image(lst_star[i], 'both')
   
   # log_gauss_full_ellips_fitting.log_image(lst_star[i], 'alone')
   # log_gauss_full_ellips_fitting.log_image(lst_star[i], 'both')