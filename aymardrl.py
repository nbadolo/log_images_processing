#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:45:07 2022

@author: nbadolo
"""


import numpy as np
from scipy.signal import convolve2d as conv2



#==============================================
#Fonction de deconvolution des cartes reduites du log agb mises à part
#==============================================

def Margaux_RL_deconv(science_im, PSF_im, nb_iter):
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