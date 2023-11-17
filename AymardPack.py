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
import os
from matplotlib import pyplot as plt
from astropy.nddata import Cutout2D
from astropy.io import fits
from scipy.stats import linregress
import scipy.optimize as opt
from sklearn.linear_model import LinearRegression
from skimage import io, color, measure, draw, img_as_bool
import pylab as plt
from scipy.signal import convolve2d as conv2



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
    x1 = u -a*b/(np.sqrt((b*(cos(theta) + theta1*sin(theta)))**2+(a*(sin(theta)-theta1*cos(theta)))**2))   # the second point of theta line
    x1_ = u + a*b/(np.sqrt((b*(cos(theta) + theta1*sin(theta)))**2+(a*(sin(theta)-theta1*cos(theta)))**2))  # the second point of (theta + pi/2) line 
    y1 = theta1*x1 + p1
    y1_ = theta1*x1_ + p1
  
  
    #--the second point of (theta + pi/2) line 
    # -- Solution du systeme d'equation(ellipse et droite de theta2)
  
    x2_ = u - a*b/(np.sqrt((b*(cos(theta) + theta2*sin(theta)))**2+(a*(sin(theta)-theta2*cos(theta)))**2))
    x2 = u + a*b/(np.sqrt((b*(cos(theta) + theta2*sin(theta)))**2+(a*(sin(theta)-theta2*cos(theta)))**2))
  
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
    Cette fonction permet supprimer les pixels chauds d'une image Fits'
    
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