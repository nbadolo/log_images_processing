#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 13:22:54 2022

@author: nbadolo
"""


"""
Personal package that contain some usefull modules I have writed
"""

 

## used packages
import numpy as np
from math import pi, cos, sin, atan
import scipy.ndimage
from scipy.ndimage import median_filter
import os
from matplotlib import pyplot as plt
from astropy.nddata import Cutout2D
from astropy.io import fits
from scipy.stats import linregress
import scipy.optimize as opt
#from sklearn.linear_model import LinearRegression
#from skimage import io, color, measure, draw, img_as_bool
import pylab as plt
from scipy.signal import convolve2d as conv2
from scipy.ndimage import median_filter, gaussian_filter




# =============================================================================
#  For the radial profile at a given orientation, theta_f       
# =============================================================================


  
def EllRadialProf(x0, y0, a, b, theta, im, num):    
    
    """
    parameters
    ----------
     
     x0, y0 : cordinates of the centre of the ellipse
     
     a, b : major and minor axis
     
     theta : orientation of the ellipse
     
     im : image that will be used for making interpolation 
     
     num : number of points of x and y grid
     
    returns :
    ----------
     x0, y0 : cordinates of the centre of the ellipse
         
     
     x1, y1 : cordinates of first intersection of theta orientation and ellipse
     
     x1_, y1_: cordinates of seconde  intersection of theta orientation and ellipse
         
     x2, y2 : cordinates of first intersection of (theta + pi/2) orientation and ellipse
     
     x2_, y2_ : cordinates of seconde  intersection of (theta + pi/2) orientation and ellipse
     
     z : used image
          
     zi1 :
          
     zi2 :
            
        
     zi1_ :
         
     zi2_ :
    """
    
    
    #-- Extract the line...
    # Make a line with "num" points...
    
    theta_f = theta
    u, v = x0, y0 
    z = im
    theta1 = np.tan(theta) # orientation suivant le grand axe de l'ellipse
    theta2 = -1/theta1     # orientation suivant le petit axe de l'ellipse
    p1 = v - theta1*u
    p2 = v - theta2*u
  
    #--the second point of theta line
    # -- Solution du systeme d'equation(ellipse et droite de theta1)
    q1 = (b*(cos(theta) + theta1*sin(theta)))**2+(a*(sin(theta)-theta1*cos(theta)))**2
    qq1 = 0
    if q1 != 0 :
        qq1 = q1       
    else :
        qq1 = 0.000001         
    
    x1 = u -a*b/(np.sqrt(qq1))   # the second point of theta line
    x1_ = u + a*b/(np.sqrt(qq1))  # the second point of (theta + pi/2) line 
    y1 = theta1*x1 + p1
    y1_ = theta1*x1_ + p1
  
  
    #--the second point of (theta + pi/2) line 
    # -- Solution du systeme d'equation(ellipse et droite de theta2)
    q2=(b*(cos(theta) + theta2*sin(theta)))**2+(a*(sin(theta)-theta2*cos(theta)))**2
    qq2 = 0
    if q2 != 0 :
        qq2 = q2
    else :
        qq2 = 0.000001

    x2_ = u - a*b/(np.sqrt(qq2))
    x2 = u + a*b/(np.sqrt(qq2))  
    y2 = theta2*x2 + p2
    y2_ = theta2*x2_ + p2
    
    
    #num = 100
    # creation d'une grille des 100 points entre chaque couple de points
    
    x, y = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
    x_, y_ = np.linspace(x0, x1_, num), np.linspace(y0, y1_, num)
    xx, yy = np.linspace(x0, x2, num), np.linspace(y0, y2, num)
    xx_, yy_ = np.linspace(x0, x2_, num), np.linspace(y0, y2_, num)

    #-- Extract the values along the line, using cubic interpolation
    zi1 = scipy.ndimage.map_coordinates(z, np.vstack((x,y))) #  les données pour le profile radial
    zi1_ = scipy.ndimage.map_coordinates(z, np.vstack((x_,y_)))
    zi2 = scipy.ndimage.map_coordinates(z, np.vstack((xx,yy)))
    zi2_ = scipy.ndimage.map_coordinates(z, np.vstack((xx_,yy_)))
    #-- Plot...
    
    return(x0, y0, x1, y1, x2, y2, z, zi1, zi2, x1_, y1_, x2_, y2_, zi1_, zi2_)





#============================================================================#
#Fonction de deconvolution des cartes reduites du large_log agb mises à part #
#============================================================================#


def Margaux_RL_deconv(science_im, PSF_im, nb_iter):
    """
    Inputs :   
        science_im : image we want to applie the deconvolution
        PSF_im :     noise we want to extract from science image
        nb_iter :    nomber of iteration
        
    Outputs :
        
    """
    #register their sizes
    size_im = science_im.shape
    idim = size_im[0]
    jdim = size_im[1]
    nim = len(science_im)
    
    size_psf = PSF_im.shape 
    idim_psf = size_psf[0]
    jdim_psf = size_psf[1] 
    nim_psf = len(PSF_im)
    
    #normalize the images

    sci0 = science_im/np.sum(science_im) 
    psf0 = PSF_im/np.sum(PSF_im) 

    #start output image (a normalized grey image)
    lucy = np.ones_like(sci0)
    lucy = lucy/np.sum(lucy)
    
    #psf = psf0
    psf = psf0
    y = sci0+1.0E-12*np.ones_like(sci0) #L2 Pup reduced (observed image)
    alpha = 1.
    
    psft = abs(np.fft.fft2(np.fft.fft2(psf)))*idim*jdim # Fast Fourier Transform
    psft = psft/np.sum(psft)
    
    present = lucy                  # x^k (L2 Pup deconvolved)
    print("Richardson-Lucy algorithm starts\n")
    for k in range (nb_iter):
        print("In progress:",k+1,"/"+str(nb_iter))
        Hx = conv2(present,psf,mode='same')+1.0E-12*np.ones_like(sci0) #Hx^k
        corr = y / Hx        #y/ (Hx^k)
        grad = corr - np.ones_like(sci0)   #y/ (Hx^k) - 1
        d = present * conv2(psft,grad,mode='same') #x^k * H^T (y/(Hx^k) - 1)
        suivant = present + alpha*d
        present = suivant
    print("Map  recovered\n")
    return present



# =============================================================================
# utilisation d'une regression linéaire pour déterminer l'orientation dun nuage 
# de points d'une image 
# =============================================================================

def LinOrientation(image, Dim):
    
    """
    inputs:
        image : l'image dont on veut extraire les points
        Dim: les dimensions de l'image, sa taille en gros

    outputs:
        alpha_rad : l'orientation de la regressrion en radian
        alpha_deg : l'orientation de la regressrion en degre

    """
    index = np.argwhere(image) # recupère les indices  des points dont l'intensité est non nulle 
    X = index[:,1]
    Y = index[:,0]
   
   
    linear_reg = np.polyfit(X, Y, 1, full = False, cov = True)
    alpha_rad = atan(linear_reg[0][0])  # recupération de la pente de la regression
    alpha_deg = alpha_rad*180/pi
    aa = linear_reg[0][0]
    bb = linear_reg[0][1]
    xx = np.arange(Dim)
    yy = aa*xx + bb
    return(alpha_rad,alpha_deg )


# =============================================================================
# Suppression de pixels chauds dans une image Fits
# =============================================================================

def DelHotPix(image) :
    
    """
    Cette fonction permet  de supprimer les pixels chauds d'une image Fits'
    
    inputs:
        image : image that we want to extract the hots pixels
        
        
    outputs : final image without the hots pixels 
    
    """
    
    # Charger le cube FITS
    hdulist = fits.open(image)
    cube = hdulist[0].data

    # Supprimer les pixels chauds en soustrayant la médiane de chaque couche
    for i in range(cube.shape[0]):
        cube[i] -= np.nanmedian(cube[i])
    #Normaliser chaque couche du cube en divisant par la valeur maximale
    # maxi = np.max(cube, axis=0)
    # for i in range(cube.shape[0]):
    #     cube[i] /= maxi

    # # Ajuster les valeurs pour qu'elles soient un peu plus lumineuses que l'arrière-plan
    # cube = cube * 0.9 + 0.1

    # # Mettre à zéro les valeurs négatives
    # cube[cube < 0] = 0

    # # Limiter les valeurs supérieures à 1 à 1
    # cube[cube > 1] = 1

    # # Appliquer une correction gamma en prenant la racine carrée
    # cube = np.sqrt(cube)

    # # Étendre l'échelle des valeurs de 0-255
    # cube *= 255

    # # Sauvegarder le cube FITS modifié
    hdulist[0].data = cube
    hdulist.writeto('cube_modifie.fits', overwrite=True)
    
    return(hdulist)
    hdulist.close()




# =============================================================================
# Clacul de la largeur à mi-hauteur d'un profile gaussien complet
# =============================================================================

def calculate_fwhm(x, y):
    # Vérifier que les entrées ne sont pas vides et ont la même longueur
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        raise ValueError("Les tableaux x et y doivent être non vides et de même longueur.")
    
    # Vérifier que y est un tableau 1D
    if y.ndim != 1:
        raise ValueError("Le tableau 'y' doit être un tableau 1D.")
    
    # Trouver le maximum
    max_y = np.max(y)
    max_x = x[np.argmax(y)]
    
    # Calculer la valeur à mi-hauteur
    half_max = max_y / 2.0
    
    # Trouver les indices les plus proches de la mi-hauteur
    indices = np.where(np.diff(np.sign(y - half_max)))[0]
    
    if len(indices) < 2:
        raise ValueError("Impossible de trouver deux points de croisement à mi-hauteur.")
    
    # Interpolation linéaire pour obtenir une estimation plus précise
    x_interp = []
    for index in indices:
        if index + 1 < len(y):  # Vérifie que l'index suivant est valide
            x_interp.append(np.interp(half_max, [y[index], y[index + 1]], [x[index], x[index + 1]]))
        else:
            raise IndexError("L'indice dépasse la longueur du tableau y.")
    
    if len(x_interp) < 2:
        raise ValueError("Pas assez de points interpolés pour calculer la FWHM.")
    
    # Calculer la largeur à mi-hauteur
    fwhm = abs(x_interp[1] - x_interp[0])
    
    return fwhm, max_x, max_y


# =============================================================================
# Clacul de la  largeur à mi-hauteur d'un demi-profile gaussien (cas des profiles radiaux)
# =============================================================================

def calculate2_fwhm(x, y):
    # Trouver le maximum
    """"
    cette fonction calcule la largeur à mi-hauteur d'un profil radiale de la moyenne d'intensité.  
    
    inpputs :
        x : les indices sur esquels s'étale l'étoile (r_mas dans le cas de mes données)
        y : l'inensité correspondante (mean_subva_arr dan le cas de mes données)
    
    """
    try:
        max_y = np.max(y)
        max_x = x[np.argmax(y)]
        
        half_max = max_y / 2.0
        indices = np.where(np.diff(np.sign(y - half_max)))[0]

        # if len(indices) < 2:
        #     print("Pas assez de points de croisement à mi-hauteur.")
        #     return None
        
        x_interp = [np.interp(half_max, [y[index], y[index + 1]], [x[index], x[index + 1]]) for index in indices]

        if len(x_interp) < 1:
            print("x_interp est vide.")
            return None

        fwhm = 2 * abs(x_interp[0] - max_x)
        return fwhm
    except Exception as e:
        print(f"Erreur dans calculate2_fwhm : {e}")
        return None  # Retourne None en cas d'erreur





# =============================================================================
# Clacul de la  largeur à une certaine hauteur d'un demi-profile gaussien (cas des profiles radiaux)
# =============================================================================
def calculate_fwm_f(x, y, h): # Celle ci est la bonne fonction 
    # Trouver le maximum
    """
    Cette fonction calcule la largeur à une certaine hauteur (par rapport au maximum) 
    d'un profil radial de la moyenne d'intensité d'une étoile.  
    
    inpputs :
        x :  les indices sur lesquels s'étale l'étoile (r_mas dans le cas de mes données)
        y :  l'inensité correspondante (mean_subv_arr dan le cas de mes données)
        h :  la hauteur à laquelle on calcule la largeur : 0<h<1
    outputs :
        fwm_h :  la largeur du profile à la hauteur h. 
        
    """
    try:
        max_y = np.max(y)
        max_x = x[np.argmax(y)]
        if h> 0 and h < 1:
            hh = h
        else :
            print("Error: The height value is not compatible")
        
        #half_max = max_y / 2.0
        height = hh*max_y
        indices = np.where(np.diff(np.sign(y - height)))[0]

        # if len(indices) < 2:
        #     print("Pas assez de points de croisement à mi-hauteur.")
        #     return None
        
        x_interp = [np.interp(height, [y[index], y[index + 1]], [x[index], x[index + 1]]) for index in indices]

        if len(x_interp) < 1:
            print("x_interp est vide.")
            return None

        fwm_h = 2 * abs(x_interp[0] - max_x)
        return fwm_h
    except Exception as e:
        print(f"Erreur dans calculate2_fwhm : {e}")
        return None  # Retourne None en cas d'erreur


# def calculate_fwm_f(x, y, h):
#     """
#     Calcule la largeur à une certaine hauteur (par rapport au maximum) d'un profil radial.

#     Inputs :
#         x : indices (ex. r_mas)
#         y : intensité correspondante (ex. mean_subv_arr)
#         h : hauteur relative (0 < h < 1)

#     Outputs :
#         fwm_h : largeur du profil à la hauteur h, ou None si échec.
#     """
#     try:
#         # Vérification des entrées
#         if h <= 0 or h >= 1:
#             print("Error: The height value is not compatible.")
#             return None
        
#         if np.any(np.isnan(y)) or np.any(np.isinf(y)):
#             print("Les données contiennent des NaN ou des valeurs infinies dans y.")
#             return None
        
#         if np.any(np.isnan(x)) or np.any(np.isinf(x)):
#             print("Les données contiennent des NaN ou des valeurs infinies dans x.")
#             return None

#         if np.max(y) == 0 or np.min(y) == np.max(y):
#             print("Profil radial plat ou sans intensité significative.")
#             return None

#         # Lissage des données
#         from scipy.ndimage import gaussian_filter
#         y = gaussian_filter(y, sigma=2)

#         # Trouver le maximum et sa position
#         max_y = np.max(y)
#         max_x = x[np.argmax(y)]
#         height = h * max_y

#         # Trouver les indices où le profil croise la hauteur spécifiée
#         indices = np.where(np.diff(np.sign(y - height)))[0]
#         if len(indices) < 1:
#             print("Pas de croisement détecté à la hauteur spécifiée.")
#             return None

#         # Interpolation des positions des croisements
#         x_interp = [np.interp(height, [y[index], y[index + 1]], [x[index], x[index + 1]]) for index in indices]
#         if len(x_interp) < 2:
#             print("Interpolation insuffisante pour calculer la largeur.")
#             return None

#         # Calcul de la largeur
#         fwm_h = 2 * abs(x_interp[0] - max_x)
#         return fwm_h

#     except Exception as e:
#         print(f"Erreur dans calculate_fwm_f : {e}")
#         return None



# =============================================================================
#   Calcul de l'écart type d'un profil radial moyen
# =============================================================================

def calculate2_stddev(x, y):
    """
    Cette fonction calcule l'écart type d'un profil radial de la moyenne d'intensité.

    Inputs :
        x : les indices sur lesquels s'étale l'étoile (r_mas dans le cas de mes données)
        y : l'intensité correspondante (mean_subva_arr dans le cas de mes données)
    """
    try:
        # On ne considère que les points valides
        valid_indices = np.where(y > 0)[0]
        if len(valid_indices) == 0:
            print("Aucun point valide trouvé.")
            return None
        
        # Calculer l'écart type uniquement sur les valeurs valides
        stddev = np.std(y[valid_indices])
        return stddev
    except Exception as e:
        print(f"Erreur dans calculate2_stddev : {e}")
        return None  # Retourne None en cas d'erreur



# =============================================================================
# Foncion pour soustraire les mauvais pixels d'une image
# =============================================================================

# Fonction pour charger une image FITS
def load_fits_image(im_a_nettoy):
    # hdulist = fits.open(file_path)
    # image_data = hdulist[0].data  # L'image est dans la première extension
    # image_data = image_data[0,:,:]
    # hdulist.close()
    image_data = im_a_nettoy
    return image_data

# Calcul du seuil des pixels chauds basé sur la moyenne et l'écart-type
def compute_threshold_hot(image_data, multiplier=3):
    mean_intensity = np.mean(image_data)
    std_intensity = np.std(image_data)
    threshold_hot = mean_intensity + multiplier * std_intensity  # ajuster avec le multiplicateur
    return threshold_hot
# Calcul du seuil des pixels mort basé sur la moyenne et l'écart-type
def compute_threshold_dead(image_data, multiplier=5):
    mean_intensity = np.mean(image_data)
    std_intensity = np.std(image_data)
    threshold_dead = mean_intensity - multiplier * std_intensity  # ajuster avec le multiplicateur
    return threshold_dead


# Fonction pour détecter les mauvais pixels (en fonction du seuil)
def detect_bad_pixels(image_data, threshold):
    # Détecte les pixels où la valeur dépasse le seuil
    bad_pixels = np.where(image_data > threshold)
    return bad_pixels

# Fonction pour remplacer les mauvais pixels par la médiane locale
def replace_bad_pixels(image_data, bad_pixels):
    # Utiliser un filtre médian pour lisser l'image autour des mauvais pixels
    # Cela remplace les mauvais pixels par des valeurs voisines plus réalistes
    cleaned_image = image_data.copy()
    cleaned_image[bad_pixels] = median_filter(image_data, size=3)[bad_pixels]
    return cleaned_image

# # Fonction pour sauvegarder l'image traitée dans un nouveau fichier FITS
# def save_fits_image(image_data, output_file):
#     hdu = fits.PrimaryHDU(image_data)
#     hdu.writeto(output_file, overwrite=True)

# Fonction principale pour charger, nettoyer et sauvegarder l'image

def process_fits_image(input_file, multiplier=3):
    """
    Cette fonction nettoie une image en retirant :
        1. le bruit avec un filtre median (d'une fenêtre de 3x3);
        2. davantage le bruit avec un filtre gaussien (cette étape a été finalement sauté car induit trop de flou);
        3. les pixels morts (dont la valeur est < 10) en les remplaçant par la médiane des pixels voisins;
        4. les pixel chauds en les remplaçant par une valeur mediane locale. Elle calcule un seuil
        d'intensité à ne pas depasser par les pixels chauds et évalue ensuite le pixel en fonction de ce seuil.
        Le calcul du seuil est basé sur un calcul de la moyenne et de l'écart-type. 
    
    input : l'image à traitée
    
    output : l'image traitée 
    
    """
    #--- Étape 1: Charger l'image FITS
    image_data = load_fits_image(input_file)
    
    # --- Étape 2: Suppression du bruit avec un filtre médian ---
    # Appliquer un filtre médian pour réduire le bruit impulsif (type sel et poivre)
    # Le paramètre "size" détermine la taille de la fenêtre du filtre
    image_data = median_filter(image_data, size=3)  # Fenêtre de 3x3 pixels

    # --- Étape 3: Appliquer un flou gaussien pour réduire davantage le bruit ---
    # Le flou gaussien est une méthode courante pour lisser l'image et réduire le bruit
    # "sigma" détermine l'intensité du flou
    #image_data = gaussian_filter(image_data, sigma=2)  # Un flou modéré

    # --- Étape 4: Traitement des pixels morts ---
    # Supposons que les pixels morts aient une valeur proche de zéro.
    # Nous allons les identifier et les remplacer par la médiane des pixels voisins.
    # Identifier les pixels morts (ici, les pixels avec une valeur inférieure à 10.En general les mixel morts ont une valeur proche de 0)
    
    # Définir le seuil pour les pixels morts basé sur 3 écarts-types sous la moyenne
    threshold_dead = compute_threshold_dead(image_data, multiplier)

    # Identifier les pixels morts avec ce seuil ajusté
    dead_pixels_mask = image_data < threshold_dead
    # Identifier les pixels morts (ici, les pixels avec une valeur inférieure à 10)
    dead_pixels_mask = image_data < threshold_dead  # Ajuster cette valeur en fonction de tes données


    # # Remplacer les pixels morts par la médiane des pixels voisins
    # image_data[dead_pixels_mask] = np.median(image_data[~dead_pixels_mask])


    # Définir le seuil pour les pixels chauds basé sur la moyenne et l'écart-type
    threshold_hot = compute_threshold_hot(image_data, 10)
    #threshold_hot = 5*np.std(image_data)

    # Détecter les  pixels chauds
    hot_pixels = detect_bad_pixels(image_data, 12)

    # Remplacer les pixels chauds par la médiane locale
    cleaned_image = replace_bad_pixels(image_data, hot_pixels)

    # # Sauvegarder l'image nettoyée
    # save_fits_image(cleaned_image, output_file)

    # # Afficher l'image avant et après nettoyage
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # axs[0].imshow(image_data, cmap='gray', origin='lower')
    # axs[0].set_title('Image Originale')
    # axs[1].imshow(cleaned_image, cmap='gray', origin='lower')
    # axs[1].set_title('Image Nettoyée')
    # plt.show()
    return cleaned_image
# # Exemple d'utilisation
# input_file = 'path_to_your_image.fits'  # Remplace par le chemin de ton fichier FITS
# output_file = 'cleaned_image.fits'  # Fichier de sortie

# process_fits_image(input_file, output_file, multiplier=5)






# =============================================================================
# Ce code fait la meme chose que le pecedent mais teste plusieurs multiplicateurs
# =============================================================================
# import numpy as np
# from astropy.io import fits
# from scipy.ndimage import median_filter
# import matplotlib.pyplot as plt

# # Fonction pour charger une image FITS
# def load_fits_image(file_path):
#     hdulist = fits.open(file_path)
#     image_data = hdulist[0].data  # L'image est dans la première extension
#     hdulist.close()
#     return image_data

# # Calcul du seuil basé sur la moyenne et l'écart-type
# def compute_threshold(image_data, multiplier=5):
#     mean_intensity = np.mean(image_data)
#     std_intensity = np.std(image_data)
#     threshold = mean_intensity + multiplier * std_intensity  # ajuster avec le multiplicateur
#     return threshold

# # Fonction pour détecter les mauvais pixels (en fonction du seuil)
# def detect_bad_pixels(image_data, threshold):
#     # Détecte les pixels où la valeur dépasse le seuil
#     bad_pixels = np.where(image_data > threshold)
#     return bad_pixels

# # Fonction pour remplacer les mauvais pixels par la médiane locale
# def replace_bad_pixels(image_data, bad_pixels):
#     # Utiliser un filtre médian pour lisser l'image autour des mauvais pixels
#     # Cela remplace les mauvais pixels par des valeurs voisines plus réalistes
#     cleaned_image = image_data.copy()
#     cleaned_image[bad_pixels] = median_filter(image_data, size=3)[bad_pixels]
#     return cleaned_image

# # Fonction pour sauvegarder l'image traitée dans un nouveau fichier FITS
# def save_fits_image(image_data, output_file):
#     hdu = fits.PrimaryHDU(image_data)
#     hdu.writeto(output_file, overwrite=True)

# # Fonction principale pour charger, nettoyer et sauvegarder l'image
# def process_fits_image(input_file, output_file, multipliers=[3, 5, 7]):
#     # Charger l'image FITS
#     image_data = load_fits_image(input_file)

#     # Test avec plusieurs multiplicateurs et afficher les résultats
#     fig, axs = plt.subplots(1, len(multipliers) + 1, figsize=(15, 6))
    
#     for i, multiplier in enumerate(multipliers):
#         threshold = compute_threshold(image_data, multiplier)
#         bad_pixels = detect_bad_pixels(image_data, threshold)
#         cleaned_image = replace_bad_pixels(image_data, bad_pixels)
#         axs[i].imshow(cleaned_image, cmap='gray', origin='lower')
#         axs[i].set_title(f"Multiplicateur {multiplier}")
    
#     axs[len(multipliers)].imshow(image_data, cmap='gray', origin='lower')
#     axs[len(multipliers)].set_title('Image Originale')
#     plt.show()

#     # Sauvegarde le résultat final avec un multiplicateur de ton choix
#     final_threshold = compute_threshold(image_data, multipliers[1])  # Choisir ici un multiplicateur
#     final_bad_pixels = detect_bad_pixels(image_data, final_threshold)
#     final_cleaned_image = replace_bad_pixels(image_data, final_bad_pixels)
#     save_fits_image(final_cleaned_image, output_file)

# # Exemple d'utilisation
# input_file = 'path_to_your_image.fits'  # Remplace par le chemin de ton fichier FITS
# output_file = 'cleaned_image.fits'  # Fichier de sortie

# process_fits_image(input_file, output_file, multipliers=[3, 5, 7])


# =============================================================================
# Sauvegarde et ouverture d'un fichier fit 
# =============================================================================


# save Q
# fpath_Q = '/home/nbadolo/SIM_CODES/RADMC3D/newest_version/radmc3d-2.0/AymardModels/PolData/Q_data.fits'
# fits.writeto(fpath_Q, Q, overwrite=True)
# var_Q = fits.getdata(fpath_Q)
# var_Q = np.reshape(var_Q,(200,200))
# #save U
# fpath_U = '/home/nbadolo/SIM_CODES/RADMC3D/newest_version/radmc3d-2.0/AymardModels/PolData/U_data.fits'
# fits.writeto(fpath_U, U, overwrite=True)
# var_U = fits.getdata(fpath_U)
# var_U = np.reshape(var_U, (200,200))
# #save CO
# fpath_CO = '/home/nbadolo/SIM_CODES/RADMC3D/newest_version/radmc3d-2.0/AymardModels/PolData/CO_data.fits'
# fits.writeto(fpath_CO, CO, overwrite=True)
# var_CO = fits.getdata(fpath_CO)

# #save SI
# fpath_SI = '/home/nbadolo/SIM_CODES/RADMC3D/newest_version/radmc3d-2.0/AymardModels/PolData/SI_data.fits'
# fits.writeto(fpath_SI, SI, overwrite=True)
# var_SI = fits.getdata(fpath_SI)


# =============================================================================
# Pour le scale bar sur mes images. les unités en arcseconde ou en ua         #
# =============================================================================
# import matplotlib.pyplot as plt
# import matplotlib.offsetbox
# from matplotlib.lines import Line2D
# import numpy as np; np.random.seed(42)

# x = np.linspace(-6,6, num=100)
# y = np.linspace(-10,10, num=100)
# X,Y = np.meshgrid(x,y)
# Z = np.sin(X)/X+np.sin(Y)/Y

# fig, ax = plt.subplots()
# ax.contourf(X,Y,Z, alpha=.1)
# ax.contour(X,Y,Z, alpha=.4)

# class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
#     """ size: length of bar in data units
#         extent : height of bar ends in axes units """
#     def __init__(self, size=1, extent = 0.03, label="", loc=2, ax=None,
#                  pad=0.4, borderpad=0.5, ppad = 0, sep=2, prop=None, 
#                  frameon=True, linekw={}, **kwargs):
#         if not ax:
#             ax = plt.gca()
#         trans = ax.get_xaxis_transform()
#         size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
#         line = Line2D([0,size],[0,0], **linekw)
#         vline1 = Line2D([0,0],[-extent/2.,extent/2.], **linekw)
#         vline2 = Line2D([size,size],[-extent/2.,extent/2.], **linekw)
#         size_bar.add_artist(line)
#         size_bar.add_artist(vline1)
#         size_bar.add_artist(vline2)
#         txt = matplotlib.offsetbox.TextArea(label, minimumdescent=False)
#         self.vpac = matplotlib.offsetbox.VPacker(children=[size_bar,txt],  
#                          align="center", pad=ppad, sep=sep) 
#         matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad, 
#                  borderpad=borderpad, child=self.vpac, prop=prop, frameon=frameon,
#                  **kwargs)

# ob = AnchoredHScaleBar(size=3, label="3 units", loc=4, frameon=True,
#                        pad=0.6,sep=4, linekw=dict(color="crimson"),)
# ikpc = lambda x: x*3.085e16 #x in kpc, return in km
# ob = AnchoredHScaleBar(size=ikpc(10), label="10kpc", loc=4, frameon=False,
#                        pad=0.6,sep=4, linekw=dict(color="k", linewidth=0.8))

# ax.add_artist(ob)
# plt.show()
