 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:43:57 2023

@author: nbadolo
"""


"""
# Ceci est le code prinicipal qui fait rouler tous les autres sur tout l'échantillon. 
# Version amélioré de full_log_reduct à laquelle on a ajouté les fichiers .txt pour recuperer 
# les eventuelles erreurs qui empechheraient le code de tourner.
"""

import os 
#import log_images # pour toutes les étoiles avec une psf. le code est  okay
#import log_images_wp #pour toutes les étoiles sans psf. le code est okay
#import log_deconv #pour tous les objets resolus à deconvoluer
#import log_gauss_full_ellips_fitting #pour tous les objets résolus deconvolués le code est okay
#import log_radial_profile_at_given_orientation_star_psf # pour les objets resolus. code en ecriture
#import paper1_log_images_fwhm # pour l'intensité etoile et psf. avec calcul de fwhm codde okay 
import paper1_log_images # pour l'intensité etoile et psf. codde okay
import paper2_log_images # pour le degré de polaristion et les vecteurs de polaristioarisation
from natsort import natsorted
import glob # pour changer l'extension des fichiers
import pandas as pd 
import shutil # pour les copies de fichiers et de repertoires


## codes calling
# lst_star0 = sorted(os.listdir('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/'))
# lst_star = sorted(os.listdir('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log_old/'))
lst_star_plus = sorted(os.listdir('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log_+/')) 
# lst_star_discard = sorted(os.listdir('/home/nbadolo/Bureau/Aymard/Donnees_sph/bad_&_discarted_log/'))
# resolved = sorted(os.listdir('/home/nbadolo/Bureau/Aymard/Donnees_sph/resolved_log/'))
#lst_star = natsorted('/home/nbadolo/Bureau/Aymard/Donnees_sph/large_log/')
#print(lst_star)
modes = ['alone','both']

# fichier .txt pour recuperer les étoiles pour lesquelles le code affiffche un erreur
txt_folder = 'sphere_files'
file_path = '/home/nbadolo/Bureau/Aymard/Donnees_sph/' + txt_folder + '/txt_files/'
file_name_suf = 'not_run_star_lst.txt'

input_directory = file_path #'chemin/vers/ton_repertoire_txt'  # Remplace par le chemin de ton répertoire
output_directory = '/home/nbadolo/Bureau/Aymard/Donnees_sph/' + txt_folder + '/csv_files/'# le chemin de ton répertoire de sortie

#log_images_wp (1)
# file_name1 = 'log_images_wp_' + file_name_suf
# n#ot_run_star_lst1 = open("{}/{}".format(file_path, file_name1), "w")
# not_run_star_lst1.write("{}, {}, {}\n".format('Star name', 'Mode', 'ErrorType'))

# log_images (2)
file_name2 = 'log_images_' + file_name_suf
#not_run_star_lst2 = open("{}/{}".format(file_path, file_name2), "w")
#not_run_star_lst2.write("{}, {}, {}\n".format('Star name', 'Mode', 'ErrorType'))

#log_gauss_full_ellips_fitting (3)
file_name3 = 'log_gauss_full_ellips_fitting_' + file_name_suf
#not_run_star_lst3 = open("{}/{}".format(file_path, file_name3), "w")
#not_run_star_lst3.write("{}, {}, {}\n".format('Star name', 'Mode', 'ErrorType'))


#log_radial_profile_at_given_orientation_star_psf (4)
file_name4 = 'log_radial_' + file_name_suf
#not_run_star_lst4 = open("{}/{}".format(file_path, file_name4), "w")
#not_run_star_lst4.write("{},{},{}\n".format('Star name', 'Mode', 'ErrorType'))
file_name44 = 'no_psf_lst1.txt'
#no_psf_star_lst4 = open("{}/{}".format(file_path, file_name44), "w")
#no_psf_star_lst4.write("{},{}\n".format('Star name', 'Mode'))
no_psf_star_lst = [] # creation d'une liste vide pour recupérer les étoiles sans psf

#paper1_log_images (5)
file_name5 = 'paper1_log_images_' + file_name_suf
# not_run_star_lst5 = open("{}/{}".format(file_path, file_name5), "w")
# not_run_star_lst5.write("{},{}\n".format('Star name', 'Mode'))
file_name55 = 'no_psf_lst2.txt'
#no_psf_star_lst5 = open("{}/{}".format(file_path, file_name55), "w")
#no_psf_star_lst5.write("{},{},{}\n".format('Star name', 'Mode', 'ErrorType'))
no_psf_star_lst2 = [] # creation d'une liste vide pour recupérer les étoiles sans psf

#paper1_log_images (6)
file_name6 = 'paper2_log_images_' + file_name_suf
#not_run_star_lst6 = open("{}/{}".format(file_path, file_name6), "w")
#not_run_star_lst6.write("{},{}\n".format('Star name', 'Mode'))
file_name66 = 'no_psf_lst3.txt'
#no_psf_star_lst6 = open("{}/{}".format(file_path, file_name66), "w")
#no_psf_star_lst6.write("{},{},{}\n".format('Star name', 'Mode', 'ErrorType'))
no_psf_star_lst3 = [] # creation d'une liste vide pour recupérer les étoiles sans psf

#paper1_log_images_fwhm (7)
file_name7 = 'paper1_log_images_fwhm_' + file_name_suf
file_name77 = 'fwhm_lst.txt'
#fwhm_lst7 = open("{}/{}".format(file_path, file_name77), "w")
#fwhm_lst7.write("{},{},{},{},{},{}\n".format('Star name ','FWHM_star ','FWHM_psf ','Ratio of FWHM ','Mode ','Filter'))
#not_run_star_lst7 = open("{}/{}".format(file_path, file_name7), "w")
#not_run_star_lst7.write("{},{},{}\n".format('Star name ','Mode ', 'ErrorType'))

#pour la conversion de txt en csv
file_name8 = 'no_conversion_lst.txt'
#no_conversion_lst = open("{}/{}".format(file_path, file_name66), "w")
#no_conversion_lst.write("{},{}\n".format('Star name', 'Mode', 'ErrorType'))


#lst_star.sort()

# lst_len = len(lst_star)
# lst_len0 = len(lst_star0)
lst_len_plus = len(lst_star_plus)
# print(lst_star)
# print(lst_len)

for star in lst_star_plus :  #affiche  les images des étoiles sans leur psf 
     
    # if star not in lst_star_plus :
    #     print(star)
        for mod in modes : 
            try :     
                no_psf = paper1_log_images.log_image(star, mod)
            #     no_psf[mod] = no_psf
            #     no_psf_star_lst2.append(no_psf)
            #     #no_psf_star_lst5.write("{},{},{}\n".format(no_psf[0], no_psf[1]))            
            # except Exception as e5:
            #     #not_run_star_lst5.write("{}, {}, {}\n".format(star, mod, e5))
            #     pass
            #     fwhm = paper1_log_images_fwhm.log_image(lst_star[i], mod)
            #     fwhm_lst7.write("{},{},{},{},{},{}\n".format(fwhm[0],fwhm[1],fwhm[2],fwhm[3], fwhm[4],fwhm[5]))
            # except Exception as e7 :
            #     not_run_star_lst7.write("{},{},{}\n".format(lst_star[i], mod, e7))
            #     pass
                paper2_log_images.log_image(star, mod) 
            except Exception as e6:
               # not_run_star_lst5.write("{}, {}, {}\n".format(star, mod, e6))
                pass
    
            #     log_images_wp.log_image(lst_star[i], mod)
            # except Exception as e1:
            #     not_run_star_lst1.write("{}, {}, {}\n".format(lst_star[i], mod, e1))
            #     pass
            #     log_images.log_image(lst_star[i], mod)
            # except Exception as e2:
            #     not_run_star_lst2.write("{}, {}, {}\n".format(lst_star[i], mod, e2))
            #     pass
            #     log_gauss_full_ellips_fitting(lst_star[i], mod)
            # except Exception as e3:
            #     not_run_star_lst3.write("{}, {}, {}\n".format(lst_star[i], mod, e3))
            #     pass
            #     no_psf_star = log_radial_profile_at_given_orientation_star_psf.log_image(lst_star[i], mod)
            #     no_psf_star_lst.append(no_psf_star) # recuperation des étoiles sans psf
            #     no_psf_star_lst4.write("{}\n".format(no_psf_star[0], no_psf_star[1]))
            # except Exception as e4:
            #     not_run_star_lst4.write("{}, {}, {}\n".format(lst_star[i], mod, e4))
            #     pass
            
        
    
    
    ## conversion des fichiers txt en format csv
    # path_txt = file_path
    # path_csv = '/home/nbadolo/Bureau/Aymard/Donnees_sph/' + txt_folder + '/csv_files/'
    # lst_txt = glob.glob(os.path.join(path_txt, "*.txt"))
    # print(lst_txt)
    # n_lst_txt = len(lst_txt)
    # print('le nombre de fichiers text est '+ str(n_lst_txt))
    
    # for fic in os.listdir(path_txt) :
    #     file_name = os.path.splitext(fic)[0]
    #     print(file_name)
    #     try :         
    #     # stop
    #         file_txt = pd.read_csv(path_txt +fic +'.txt', delimiter = ';') # ouverture du fichier txt avec csv
    #         #file_txt = pd.read_csv(path_txt +fic, delim_whitespace=True) # ouverture du fichier txt avec csv
    #         file_csv = file_txt.to_csv(path_csv + file_name+'.csv', index = None) # creation du fichier csv
    #     except Exception as e8 :
    #         no_conversion_lst.write("{},{}\n".format(file_name, e8))
    #         pass
     
    
#%%

# Assurez-vous que le répertoire de sortie existe
# os.makedirs(output_directory, exist_ok=True)

# # Parcourir tous les fichiers dans le répertoire
# for filename in os.listdir(input_directory):
#     if filename.endswith('.txt'):
#         # Chemin complet du fichier .txt
#         txt_file_path = os.path.join(input_directory, filename)
        
#         # Lire le fichier .txt dans un DataFrame pandas
#         try:
#             df = pd.read_csv(txt_file_path, sep='\t')  # Modifier le séparateur si nécessaire
#         except Exception as e:
#             print(f"Erreur lors de la lecture du fichier {filename}: {e}")
#             continue
        
#         # Créer le chemin pour le fichier .csv de sortie
#         csv_file_path = os.path.join(output_directory, filename.replace('.txt', '.csv'))
        
#         # Enregistrer le DataFrame en tant que fichier .csv
#         df.to_csv(csv_file_path, index=False)
#         print(f"Converti: {filename} -> {csv_file_path}")
