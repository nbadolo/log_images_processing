#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 17:42:33 2017

Library to begin the analysis of ZIMPOL polarimetric data

@author: miguel
"""

#pylint: disable-msg=E1101,C0103

import sys
import os
import re
import csv
import warnings as wa
import locale
import datetime
from glob import glob
import numpy as np
import matplotlib.scale as mscale
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm, PowerNorm
from scipy.signal import correlate2d
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.simbad import Simbad
from uncertainties import unumpy
from uncertainties.umath import *

class SquareRootScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.
    """

    name = 'sqrt'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return  max(0., vmin), vmax

    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            a[a < 0] = 0
            return np.array(a)**0.5

        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            return np.array(a)**2

        def inverted(self):
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        return self.SquareRootTransform()

def manual_coord():
    """
    Prompting celestial coordinates to the user
    """

    print('\n Simbad query unsucessful.')
    target_ra = input('Enter RA manually (HH:MM:SS.XXX): ')
    target_ra = target_ra.replace(' ', ':')
    target_dec = input('Enter DEC manually (DD:MM:SS.XXX):')
    target_dec = target_dec.replace(' ', ':')

    return target_ra, target_dec


def apply_WCS(infile, pix_size=3.6006):
    """
    Applying FITS header to the FITS files

    Parameters
    ----------
    **infile** : Complete path to the FITS file

    Keywords
    ----------
    **pix_size** : ZIMPOL pixel size in mas/pixel
    """

    #Opening FITS and getting header
    hdu = fits.open(infile, mode='update')
    indir = os.path.dirname(infile)
    hdr = hdu[0].header

    #Traget name, removing _ and putting space before capitals
    target = hdr['OBJECT']
    target = target.replace('_', ' ')
    target = target.replace('HR', 'hr')
    target = target.replace('HD', 'hd')
    target = re.sub(r"(\w)([A-Z])", r"\1 \2", target)

    #CDS request
    try:
        target_cds = None
        target_cds = Simbad.query_object(target.strip())
        print('Target is '+target)

        #If the target is not recognized by SIMBAD, the fields are not updated
#        validate_cds = (target_cds == Simbad.query_object(target_cds['MAIN_ID']))

        if target_cds is not None:
            target_ra = target_cds['RA'][0].replace(' ', ':')
            target_dec = target_cds['DEC'][0].replace(' ', ':')
        #If target not recognized, manual coordinates
        else:
            try:
                with open(os.path.join( indir, 'corresp_WCS.csv'), mode='r') as infile:
                    reader = csv.reader(infile)
                    corresp_name_simbad = {rows[0]:rows[1] for rows in reader}
                target = corresp_name_simbad[target]
                target_cds = Simbad.query_object(target.strip())
            except FileNotFoundError:
                target_in = target
                target = input(target+' not found in Simbad, provide another name: ')
                target_cds = None
                target_cds = Simbad.query_object(target.strip())
                print('Target is '+target)
                if target_cds is not None:
                    with open(os.path.join(indir, 'corresp_WCS.csv'), mode='a') as outfile:
                        outfile.write(target_in+','+target)
            if target_cds is not None:
                target_ra = target_cds['RA'][0].replace(' ', ':')
                target_dec = target_cds['DEC'][0].replace(' ', ':')
            else:
                target_ra, target_dec = manual_coord()

    #If time out, manual coordinates
    except KeyError:
        print('Timeout on Simbad, provide manual coordinates for '+target)
        target_ra, target_dec = manual_coord()

    #Conversion in degrees
    coord = SkyCoord(target_ra+' '+target_dec, unit=(u.hourangle, u.deg))

    target_ra = coord.ra.value
    target_dec = coord.dec.value

    #Adding WCS to the FITS file
    hdr['CRPIX1'] = hdr['NAXIS1']/2
    hdr['CRPIX2'] = hdr['NAXIS2']/2
    hdr['CUNIT1'] = 'deg'
    hdr['CUNIT2'] = 'deg'
    hdr['CTYPE1'] = "RA---TAN"
    hdr['CTYPE2'] = "DEC--TAN"
    hdr['CRVAL1'] = target_ra
    hdr['CRVAL2'] = target_dec
    rotangle = 0#hdr['HIERARCH ESO INS4 DROT2 POSANG']*np.pi/180.
#    rotangle -= 1.6*np.pi/180.   #Rotation of the camera, taken into account by ESOREX
    hdr['CD1_1'] = -np.cos(rotangle)*pix_size*1e-3/3600.
    hdr['CD1_2'] = np.sin(rotangle)*pix_size*1e-3/3600.
    hdr['CD2_1'] = np.sin(rotangle)*pix_size*1e-3/3600.
    hdr['CD2_2'] = np.cos(rotangle)*pix_size*1e-3/3600.

    #Now
    now = datetime.datetime.now()
    now_history = now.strftime("%d %b %Y %H:%M:%S")

    #Updating header
    upd_hdr = "WCS on "+now_history+" by "+os.path.basename(os.getenv("HOME"))
    hdr['HISTORY'] = upd_hdr

    #Closing FITS
    hdu[0].header = hdr
    hdu.flush()
    hdu.close()

def radial_profiles(infile, sup_poldeg=None, sqrtInt=False):
    """
    A function to plot the radial profile from a *POLAR_COMPUTED.fits file produced by
    computepolar. The radial profile is centered on the maximum pixel of the intensity frame.

    **WARNING** Must be run after computepolar.

    Parameters
    ----------
    **infile** : Input *POLAR_COMPUTED.fits

    Keywords
    ----------
    **(sup_poldeg|None)** : If not set to *None*, this is the value that will be used as upper limit
    to display the degree of linear polarization

    **(sqrtInt|False)** : If set to *True*, the intensity plot as a sqrt scale.

    Returns
    ----------
    The figure instance.
    """

    #To be able to use the square root scale on the intensity
    mscale.register_scale(SquareRootScale)

    ##Opening FITS file    
    #with (fits.open(infile)) as hdu:

        # intens = hdu["INTENSITY.IMAGE"].data   # les lignes de Miguel
        # polflux = hdu["POLFLUX.IMAGE"].data
        # poldeg = hdu["POLDEG.IMAGE"].data
        # hdr = hdu[0].header
    
    # lst_files = glob(os.path.join(infile, "*.fits"))
    # n_files =len(lst_files)
    #
    n_files = len(infile)
    #print('le contenu de infile est ' + str(lst_files))
    #lists
    nSubDim = 200
    i_v_arr = np.empty((n_files,nSubDim,nSubDim)) # liste ajoutée pour recuperer les fits ouverts
    ##Opening FITS file
    for i in range (n_files):
          hdu = fits.open(infile)[0]   
          data = hdu.data   
          i_v = data[0,:,:] 
        
    intens = i_v_arr[0]
    polflux = i_v_arr[0]
    poldeg = i_v_arr[0]
    hdr = hdu.header
    print('la forme de lobjet est' + str(intens.shape))    


    #Locating intensity maximum
    loc_max = np.unravel_index(intens.argmax(), intens.shape)

    #Radial map
    y, x = np.indices((polflux.shape))
    r = np.sqrt((x - loc_max[1])**2 + (y - loc_max[0])**2)
    r = r.astype(np.int)
    nr = np.bincount(r.ravel())

    #Intensity profile
    tbin_i = np.bincount(r.ravel(), intens.ravel())
    tbin_i_sq = np.bincount(r.ravel(), (intens*intens).ravel())
    rp_i_mean = tbin_i / nr
    rp_i_std = np.sqrt(np.abs(tbin_i_sq/nr - rp_i_mean**2))

    #Polar flux profile
    tbin_f = np.bincount(r.ravel(), polflux.ravel())
    tbin_f_sq = np.bincount(r.ravel(), (polflux*polflux).ravel())
    rp_f_mean = tbin_f / nr
    rp_f_std = np.sqrt(np.abs(tbin_f_sq/nr - rp_f_mean**2))

    #Degree of linear polarization
    tbin_d = np.bincount(r.ravel(), poldeg.ravel())
    tbin_d_sq = np.bincount(r.ravel(), (poldeg*poldeg).ravel())
    rp_d_mean = tbin_d / nr
    rp_d_std = np.sqrt(np.abs(tbin_d_sq/nr - rp_d_mean**2))

    #X scale (for plots)
    x_scale = np.linspace(0, np.linalg.norm((hdr['CD1_1'], hdr['CD2_2']))*rp_f_mean.size, \
    rp_f_mean.size)*3600*1e3

    #Plotting
    fig, (ax_i, ax_f, ax_d) = plt.subplots(3, 1, sharex=True)
    #Intensity
    ax_i.errorbar(x_scale, rp_i_mean, yerr = rp_i_std, fmt = 'orange')
    if sqrtInt:
        ax_i.set_yscale('sqrt')
    ax_i.grid()
    ax_i.set_ylabel('Intensity')
    #Polflux
    ax_f.errorbar(x_scale, rp_f_mean, yerr = rp_f_std, fmt = 'r')
    ax_f.grid()
    ax_f.set_xlim(0, 250)
    ax_f.set_ylabel('Polarized flux')
    #Poldeg
    ax_d.errorbar(x_scale, rp_d_mean, yerr=rp_d_std, fmt='k')
    ax_d.grid()
    if sup_poldeg is not None:
        ax_d.set_ylim(0, sup_poldeg)
    ax_d.set_ylabel('Degree of lin. pol.')
    ax_d.set_xlabel('Distance to star center (mas)')

    return fig

def plot_all_radial_profiles(indir, form, sup_poldeg=None, sqrtInt=False):
    """
    This function plots the radial_profiles for an entire dataset reduced by ESOREFLEX
    (this mean all target/filters reduced by the pipeline).

    **WARNING** Must be run after computepolar.

    Parameters
    ----------
    **input_file** : Complete path to the product directory containing all the SPH*tpl directories
    for each target/filter.

    **form** : Format of the files containg the plots (pdf, jpg, png, ...)

    Keywords
    ----------
    **(sup_poldeg|None)** : If not set to *None*, this is the value that will be used as upper limit
    to display the degree of linear polarization

    **(sqrtInt|False)** : If set to *True*, the intensity plot as a sqrt scale.

    See also
    ----------
    *radial_profiles*
    """

    #Listing products files
    #list_files = glob(os.path.join(indir, "*/*POLAR_COMPUTED.fits")) # la ligne de Miguel
    list_files = glob(os.path.join(indir, "*.fits")) # ma ligne à moi


    #For each file
    if len(list_files) == 0:

        #print("\nNo SPH*tpl directories containing *POLAR_COMPUTED.fits") # la ligne de Miguel 
        print("\nNo SPH*tpl directories containing *.fits") # ma ligne à moi 
        print("in the current input directory.")
        print("\nCheck your path ! Aborting...\n")

    else:

        #plt.ioff()
        plt.clf()
        plt.figure()

        for cur_file in list_files:

            #Plotting
            fig = radial_profiles(cur_file, sup_poldeg=sup_poldeg, sqrtInt=sqrtInt)
            fig.set_size_inches(8, 15)
            fig.savefig(cur_file[:cur_file.rfind("_POLAR")] + "_RADIAL." + form, bbox_inches='tight', \
            dpi=300)
            plt.close(fig)

            #plt.savefig('/home/nbadolo/Bureau/Aymard/Donnees_sph/All_plots/Mig_prof/'+ cur_file + '.pdf', 
            #                dpi=100, bbox_inches ='tight')
            plt.tight_layout()
        #plt.ion()
        #plt.imshow()

def computepolar(reflex_product_dir, sep_files = False, merge_common = True):
    """
    Computation of the polarization quantities : intensity, polarized flux, degree of linear
    polarization, polarization angle.

    :math:`I = \\frac{i(+Q) \\ + \\ i(-Q) \\ + \\ i(+U) \\ + \\ i(-U)}{4}`

    :math:`pI = \\sqrt{p(Q)^2 + p(U)^2}`

    :math:`DoLP = \\sqrt{\\left(\\frac{p(Q)}{i(Q)}\\right)^2 +
    \\left(\\frac{p(U)}{i(U)}\\right)^2}`

    :math:`\\theta = 0.5*\\arctan\\left(\\frac{p(Q)}{i(Q)}; \\frac{p(U)}{i(U)}\\right)`

    With i the intensity frame and p the pframe of the different files.

    Parameters
    ----------
    **reflex_product_directory** : Directory containing the SPHER*tpl directories produced by
    ESOREFLEX.

    Keywords
    ----------
    **(sep_files|False)** : If set to *True*, the different observables mentionned above are put in
    separated FITS file and in a multiextension FITS file. If set to False (default), only the
    multiextension FITS file is written.

    **(merge_common|True)** If True, merge common filters on the 2 arms

    Output
    ----------
    The polarization observables are written in a FITS file(s) in the SPHER*tpl directories. Its
    name is *OB_name+filter_name+_POLAR_COMPUTED.fits*. The nature of each extension is provided in
    extname keyword.
    """

    plt.close('all')

    if not locale.getlocale(locale.LC_TIME) == ('en_US', 'UTF-8'):
        locale.setlocale(locale.LC_TIME, "en_US.utf8")

    #Listing templates
    list_prod = glob(os.path.join(reflex_product_dir, "SPH*tpl"))
    list_prod.sort()

    #For each template
    for cur_tpl in list_prod:

        #Listting product files
        tpl_file = glob(os.path.join(cur_tpl, "*ZPL_SCIENCE_P23_REDUCED_Q_CAM1.fits"))
        if len(tpl_file) == 0:
            continue
        tpl_file = tpl_file[0][:-11]

        for cam in range(2):

            #Opening input files
            try:
                hduq = fits.open(tpl_file+"Q_CAM"+str(cam+1)+".fits", mode='readonly')
                hduplusq = fits.open(tpl_file+"QPLUS_CAM"+str(cam+1)+".fits", mode='readonly')
                hduminusq = fits.open(tpl_file+"QMINUS_CAM"+str(cam+1)+".fits", mode='readonly')
                hduu = fits.open(tpl_file+"U_CAM"+str(cam+1)+".fits", mode='readonly')
                hduplusu = fits.open(tpl_file+"UPLUS_CAM"+str(cam+1)+".fits", mode='readonly')
                hduminusu = fits.open(tpl_file+"UMINUS_CAM"+str(cam+1)+".fits", mode='readonly')
            except FileNotFoundError:
                wa.warn(os.path.basename(cur_tpl)+" is missing some files")
                continue

            #Now
            now = datetime.datetime.now()
            now_history = now.strftime("%d %b %Y %H:%M:%S")


            #Writting output : 1 file per camera
            if cam == 0:
                filt = hduplusq[0].header['ESO INS3 OPTI5 NAME']
            else:
                filt = hduplusq[0].header['ESO INS3 OPTI6 NAME']
            name_out = tpl_file+filt+"_POLAR_COMPUTED"

            #Updating header
            upd_hdr = "Polarization computed on "+now_history+" by "+\
            os.path.basename(os.getenv("HOME"))
            hduplusq[0].header['HISTORY'] = upd_hdr
            hduminusq[0].header['HISTORY'] = upd_hdr
            hduplusu[0].header['HISTORY'] = upd_hdr
            hduminusu[0].header['HISTORY'] = upd_hdr

            # =============================================================================
            #                        INTENSITY
            # =============================================================================
            hduplusq_un = unumpy.uarray(hduplusq['IFRAME.IMAGE'].data, hduplusq['IFRAME.RMSMAP'].data)
            hduminusq_un = unumpy.uarray(hduminusq['IFRAME.IMAGE'].data, hduminusq['IFRAME.RMSMAP'].data)
            hduplusu_un = unumpy.uarray(hduplusu['IFRAME.IMAGE'].data, hduplusu['IFRAME.RMSMAP'].data)
            hduminusu_un = unumpy.uarray(hduminusu['IFRAME.IMAGE'].data, hduminusu['IFRAME.RMSMAP'].data)

            #Combining
            iframe_un = (hduplusq_un + hduminusq_un + hduplusu_un + hduminusu_un)/4
            iframe = unumpy.nominal_values(iframe_un)
            iframesig = unumpy.std_devs(iframe_un)
            hdr_primary = hduplusq[0].header
            hdr_primary['EXTNAME'] = "INTENSITY.IMAGE"

            #If same filter on both cameras
            if hduplusq[0].header['ESO INS3 OPTI5 NAME'] == hduplusq[0].header['ESO INS3 OPTI6 NAME'] and cam == 1 and merge_common:

                #Opening computed CAM1
                with fits.open(name_out+'.fits') as hdul1:
                    iframe1 = hdul1['INTENSITY.IMAGE'].data
                    iframesig1 = hdul1['INTENSITY.RMSMAP'].data
                    polflux1 = hdul1['POLFLUX.IMAGE'].data
                    polfluxsig1 = hdul1['POLFLUX.RMSMAP'].data
                    poldeg1 = hdul1['POLDEG.IMAGE'].data
                    poldegsig1 = hdul1['POLDEG.RMSMAP'].data
                    polangle1 = hdul1['POLANGLE'].data

                    # No need to change DIT or EXPTIME as we average
                    # hdr_primary['ESO DET SEQ1 EXPTIME'] += hdul1[0].header['ESO DET SEQ1 EXPTIME']
                    # hdr_primary['ESO DET DIT1'] += hdul1[0].header['ESO DET DIT1']
                    hdr_primary['COMMENT'] = 'This file is the result of the average of the two arms of ZIMPOL'

                #Find shift
                locmax_1 = np.unravel_index(np.argmax(iframe), iframe.shape)
                locmax_2 = np.unravel_index(np.argmax(iframe1), iframe1.shape)

                #Centring
                size_box = int((np.min([locmax_1[0], locmax_1[1], locmax_2[0], locmax_2[1], \
                iframe.shape[0]-locmax_1[0], iframe.shape[1]-locmax_1[1], \
                iframe1.shape[0]-locmax_2[0], iframe1.shape[1]-locmax_2[1]]))/2)
                iframe = iframe[locmax_1[0]-size_box:locmax_1[0]+size_box, locmax_1[1]-size_box:locmax_1[1]+size_box]
                iframe1 = iframe1[locmax_2[0]-size_box:locmax_2[0]+size_box, locmax_2[1]-size_box:locmax_2[1]+size_box]
                iframesig = iframesig[locmax_1[0]-size_box:locmax_1[0]+size_box, locmax_1[1]-size_box:locmax_1[1]+size_box]
                iframesig1 = iframesig1[locmax_2[0]-size_box:locmax_2[0]+size_box, locmax_2[1]-size_box:locmax_2[1]+size_box]

                if 0 in iframe.shape or 0 in iframe1.shape:
                    print("Error with "+name_out)
                    print(locmax_1, locmax_2)
                    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
                    axes[0].imshow(iframe1, cmap='cubehelix', origin='lower')
                    axes[1].imshow(iframe, cmap='cubehelix', origin='lower')
                    sys.exit(10)

                #Summing
                iframe_un = unumpy.uarray(iframe, iframesig)
                iframe1_un = unumpy.uarray(iframe1, iframesig1)
                iframe_avg_un = (iframe_un+iframe1_un)/2
                iframe = unumpy.nominal_values(iframe_avg_un)
                iframesig = unumpy.std_devs(iframe_avg_un)

            hdus = [fits.PrimaryHDU(iframe, header=hdr_primary)]
            hdus.append(fits.ImageHDU(iframesig, name="INTENSITY.RMSMAP"))
            if sep_files:
                sep_dir = os.path.join(os.path.dirname(name_out), "sep_files")
                if not os.path.isdir(sep_dir):
                    os.mkdir(sep_dir)
                hdu_i = fits.PrimaryHDU(iframe, header=hduplusq[0].header)
                hdu_i.writeto(os.path.join(sep_dir, os.path.basename(name_out)+"_i.fits"), \
                overwrite=True)
                hdu_irms = fits.PrimaryHDU(iframesig, header=hduplusq[0].header)
                hdu_irms.writeto(os.path.join(sep_dir, os.path.basename(name_out)+"_irms.fits"), \
                overwrite=True)

            # =============================================================================
            #                        POLARIZED FLUX
            # =============================================================================
            Qp_un = unumpy.uarray(hduq["PFRAME.IMAGE"].data, hduq["PFRAME.RMSMAP"].data)
            Up_un = unumpy.uarray(hduu["PFRAME.IMAGE"].data, hduu["PFRAME.RMSMAP"].data)
            polflux_un = unumpy.sqrt(unumpy.pow(Up_un, 2) + unumpy.pow(Qp_un, 2))
            polflux = unumpy.nominal_values(polflux_un)
            polfluxsig = unumpy.std_devs(polflux_un)

            #If same filter on both cameras
            if hduq[0].header['ESO INS3 OPTI5 NAME'] == hduq[0].header['ESO INS3 OPTI6 NAME'] and cam == 1 and merge_common:
                polflux = polflux[locmax_1[0]-size_box:locmax_1[0]+size_box, locmax_1[1]-size_box:locmax_1[1]+size_box]
                polflux1 = polflux1[locmax_2[0]-size_box:locmax_2[0]+size_box, locmax_2[1]-size_box:locmax_2[1]+size_box]
                polfluxsig = polfluxsig[locmax_1[0]-size_box:locmax_1[0]+size_box, locmax_1[1]-size_box:locmax_1[1]+size_box]
                polfluxsig1 = polfluxsig1[locmax_2[0]-size_box:locmax_2[0]+size_box, locmax_2[1]-size_box:locmax_2[1]+size_box]

                polflux_un = unumpy.uarray(polflux, polfluxsig)
                polflux1_un = unumpy.uarray(polflux1, polfluxsig1)
                polflux_avg_un = (polflux_un+polflux1_un)/2
                polflux = unumpy.nominal_values(polflux_avg_un)
                polfluxsig = unumpy.std_devs(polflux_avg_un)

            hdus.append(fits.ImageHDU(polflux, header=hduq[0].header, name="POLFLUX.IMAGE"))
            hdus.append(fits.ImageHDU(polfluxsig, name="POLFLUX.RMSMAP"))
            if sep_files:
                hdu_pf = fits.PrimaryHDU(polflux, header=hduplusq[0].header)
                hdu_pf.writeto(os.path.join(sep_dir, os.path.basename(name_out)+"_polflux.fits"), \
                overwrite=True)
                hdu_pfrms = fits.PrimaryHDU(polfluxsig, header=hduplusq[0].header)
                hdu_pfrms.writeto(os.path.join(sep_dir, os.path.basename(name_out)+\
                "_polfluxrms.fits"), overwrite=True)

            # =============================================================================
            #      DEGREE OF LINEAR POLARIZATION - Ref : Li et al. 2014, JQSRT
            #             https://doi.org/10.1016/j.jqsrt.2014.09.003
            # =============================================================================
            Qi_un = unumpy.uarray(hduq["IFRAME.IMAGE"].data, hduq["IFRAME.RMSMAP"].data)
            Ui_un = unumpy.uarray(hduu["IFRAME.IMAGE"].data, hduu["IFRAME.RMSMAP"].data)

            q = Qp_un / Qi_un
            u = Up_un / Ui_un
            poldeg_un = unumpy.sqrt(q*q + u*u)
            poldeg = unumpy.nominal_values(poldeg_un)
            tag_bad = poldeg > 1.0
            poldeg[tag_bad] = np.nan# value for bad pixels
            poldegsig = unumpy.std_devs(poldeg_un)
            poldegsig[tag_bad] = np.nan

            #If same filter on both cameras
            if hduq[0].header['ESO INS3 OPTI5 NAME'] == hduq[0].header['ESO INS3 OPTI6 NAME'] and cam == 1 and merge_common:
                poldeg = poldeg[locmax_1[0]-size_box:locmax_1[0]+size_box, locmax_1[1]-size_box:locmax_1[1]+size_box]
                poldeg1 = poldeg1[locmax_2[0]-size_box:locmax_2[0]+size_box, locmax_2[1]-size_box:locmax_2[1]+size_box]
                poldegsig = poldegsig[locmax_1[0]-size_box:locmax_1[0]+size_box, locmax_1[1]-size_box:locmax_1[1]+size_box]
                poldegsig1 = poldegsig1[locmax_2[0]-size_box:locmax_2[0]+size_box, locmax_2[1]-size_box:locmax_2[1]+size_box]
                poldeg_un = unumpy.uarray(poldeg, poldegsig)
                poldeg1_un = unumpy.uarray(poldeg1, poldegsig1)
                poldeg_avg_un = (poldeg_un+poldeg1_un)/2
                poldeg = unumpy.nominal_values(poldeg_avg_un)
                poldegsig = unumpy.std_devs(poldeg_avg_un)

            hdus.append(fits.ImageHDU(poldeg, name="POLDEG.IMAGE"))
            if sep_files:
                hdu_pd = fits.PrimaryHDU(poldeg, header=hduplusq[0].header)
                hdu_pd.writeto(os.path.join(sep_dir, os.path.basename(name_out)+"_poldeg.fits"), \
                overwrite=True)

            hdus.append(fits.ImageHDU(poldegsig, name="POLDEG.RMSMAP"))
            if sep_files:
                hdu_pdrms = fits.PrimaryHDU(poldegsig, header=hduplusq[0].header)
                hdu_pdrms.writeto(os.path.join(sep_dir, os.path.basename(name_out)+\
                "_poldegrms.fits"), overwrite=True)


            # =============================================================================
            #      POLARIZATION ANGLE - Ref : Li et al. 2014, JQSRT
            #             https://doi.org/10.1016/j.jqsrt.2014.09.003
            # =============================================================================
            q_nom = unumpy.nominal_values(q)
            u_nom = unumpy.nominal_values(u)
            polangle = np.arctan2(u_nom, q_nom)/2.0*(180./np.pi)

            #If same filter on both cameras
            if hduq[0].header['ESO INS3 OPTI5 NAME'] == hduq[0].header['ESO INS3 OPTI6 NAME'] and cam == 1 and merge_common:
                polangle = polangle[locmax_1[0]-size_box:locmax_1[0]+size_box, locmax_1[1]-size_box:locmax_1[1]+size_box]
                polangle1 = polangle1[locmax_2[0]-size_box:locmax_2[0]+size_box, locmax_2[1]-size_box:locmax_2[1]+size_box]
                polangle = np.angle((np.exp(1j*np.deg2rad(polangle))+np.exp(1j*np.deg2rad(polangle1)))/2., deg=True)

            hdus.append(fits.ImageHDU(polangle, name="POLANGLE"))
            if sep_files:
                hdu_pa = fits.PrimaryHDU(polangle, header=hduplusq[0].header)
                hdu_pa.writeto(os.path.join(sep_dir, os.path.basename(name_out)+"_polangle.fits"), \
                overwrite=True)


            hdu_list = fits.HDUList(hdus)
            hdu_list.writeto(name_out+".fits", overwrite=True)
            apply_WCS(name_out+'.fits')

            #Closing inputs
            hduq.close()
            hduplusq.close()
            hduminusq.close()
            hduu.close()
            hduplusu.close()
            hduminusu.close()

def computepolar_cube(reflex_product_dir, sep_files=False):
    """
    Computation of the polarization quantities : intensity, polarized flux, degree of linear
    polarization, polarization angle.

    :math:`I = \\frac{i(+Q) \\ + \\ i(-Q) \\ + \\ i(+U) \\ + \\ i(-U)}{4}`

    :math:`pI = \\sqrt{[p(+Q) - p(-Q)]^2 + [p(+U) - p(-U)]^2}`

    :math:`DoLP = \\sqrt{\\left(\\frac{p(Q)}{i(Q)}\\right)^2 +
    \\left(\\frac{p(U)}{i(U)}\\right)^2}`

    :math:`\\theta = 0.5*\\arctan\\left(\\frac{p(Q)}{i(Q)}; \\frac{p(U)}{i(U)}\\right)`

    With i the intensity frame and p the pframe of the different files.

    Parameters
    ----------
    **reflex_product_directory** : Directory containing the directories with the SPHERE reduced
    cubes

    Keywords
    ----------
    **(sep_files|False)** : If set to *True*, the different observables mentionned above are put in
    separated FITS file and in a multiextension FITS file. If set to False (default), only the
    multiextension FITS file is written.

    Output
    ----------
    The polarization observables are written in a FITS file(s) in the SPHER*tpl directories. Its
    name is *OB_name+filter_name+_POLAR_COMPUTED.fits*. The nature of each extension is provided in
    extname keyword.
    """

    if not locale.getlocale(locale.LC_TIME) == ('en_US', 'UTF-8'):
        locale.setlocale(locale.LC_TIME, "en_US.utf8")

    #Listting product files
    tpl_file = glob(os.path.join(reflex_product_dir, "coll_tmp_cal_Qplus_cam1.fits"))
    tpl_file = tpl_file[0][:-15]
    print(tpl_file)

    for cam in range(2):

        #Opening input files
        try:
            hduplusq = fits.open(tpl_file+"Qplus_cam"+str(cam+1)+".fits", mode='readonly')
            hduminusq = fits.open(tpl_file+"Qminus_cam"+str(cam+1)+".fits", mode='readonly')
            hduplusu = fits.open(tpl_file+"Uplus_cam"+str(cam+1)+".fits", mode='readonly')
            hduminusu = fits.open(tpl_file+"Uminus_cam"+str(cam+1)+".fits", mode='readonly')
        except FileNotFoundError:
            wa.warn(os.path.basename(reflex_product_dir)+" is missing some files")
            continue

        #Now
        now = datetime.datetime.now()
        now_history = now.strftime("%d %b %Y %H:%M:%S")


        #Writting output : 1 file per camera
        if cam == 0:
            filt = hduplusq[0].header['ESO INS3 OPTI5 NAME']
        else:
            filt = hduplusq[0].header['ESO INS3 OPTI6 NAME']
        name_out = tpl_file+filt+"_POLAR_COMPUTED"

        #Updating header
        upd_hdr = "Polarization computed on "+now_history+" by "+\
        os.path.basename(os.getenv("HOME"))
        hduplusq[0].header['HISTORY'] = upd_hdr
        hduminusq[0].header['HISTORY'] = upd_hdr
        hduplusu[0].header['HISTORY'] = upd_hdr
        hduminusu[0].header['HISTORY'] = upd_hdr

        #Intensity frame, rms and ncomb from ALL images
        iframe = (hduplusq[0].data[0, :, :] + hduminusq[0].data[0, :, :] + \
        hduplusu[0].data[0, :, :] + hduminusu[0].data[0, :, :])/4.0
        hdr_primary = hduplusq[0].header
        hdr_primary['EXTNAME'] = "INTENSITY.IMAGE"

        #If same filter on both cameras
        if hduplusq[0].header['ESO INS3 OPTI5 NAME'] == hduplusq[0].header['ESO INS3 OPTI6 NAME'] and cam == 1:

            #Opening computed CAM1
            with fits.open(name_out+'.fits') as hdul1:
                iframe1 = hdul1['INTENSITY.IMAGE'].data
#                iframesig1 = hdul1['INTENSITY.RMSMAP'].data
                polflux1 = hdul1['POLFLUX.IMAGE'].data
#                polfluxsig1 = hdul1['POLFLUX.RMSMAP'].data
                poldeg1 = hdul1['POLDEG.IMAGE'].data
#                poldegsig1 = hdul1['POLDEG.RMSMAP'].data
                polangle1 = hdul1['POLANGLE'].data

                hdr_primary['ESO DET SEQ1 EXPTIME'] += hdul1[0].header['ESO DET SEQ1 EXPTIME']

                #Find shift
                locmax_1 = np.unravel_index(np.argmax(iframe), iframe.shape)
                locmax_2 = np.unravel_index(np.argmax(iframe1), iframe1.shape)

                #Centring
                size_box = int((iframe.shape[0]-100)/2)
                iframe = iframe[locmax_1[0]-size_box:locmax_1[0]+size_box, locmax_1[1]-size_box:locmax_1[1]+size_box]
                iframe1 = iframe1[locmax_2[0]-size_box:locmax_2[0]+size_box, locmax_2[1]-size_box:locmax_2[1]+size_box]
#                iframesig = iframesig[locmax_1[0]-size_box:locmax_1[0]+size_box, locmax_1[1]-size_box:locmax_1[1]+size_box]
#                iframesig1 = iframesig1[locmax_2[0]-size_box:locmax_2[0]+size_box, locmax_2[1]-size_box:locmax_2[1]+size_box]

                #Summing
                iframe = (iframe+iframe1)/2
#                iframesig = np.sqrt(iframesig**2+iframesig1**2)


        hdus = [fits.PrimaryHDU(iframe, header=hdr_primary)]

        if sep_files:
            sep_dir = os.path.join(os.path.dirname(name_out), "sep_files")
            if not os.path.isdir(sep_dir):
                os.mkdir(sep_dir)
            hdu_i = fits.PrimaryHDU(iframe, header=hduplusq[0].header)
            hdu_i.writeto(os.path.join(sep_dir, os.path.basename(name_out)+"_i.fits"), \
            overwrite=True)

        # Polarized flux
        polflux = np.sqrt(np.square(hduplusu[0].data[1, :, :] - hduminusu[0].data[1, :, :])/4 \
        + np.square(hduplusq[0].data[1, :, :] - hduminusq[0].data[1, :, :])/4)

        #If same filter on both cameras
        if hduplusq[0].header['ESO INS3 OPTI5 NAME'] == hduplusq[0].header['ESO INS3 OPTI6 NAME'] and cam == 1:
            polflux = polflux[locmax_1[0]-size_box:locmax_1[0]+size_box, locmax_1[1]-size_box:locmax_1[1]+size_box]
            polflux1 = polflux1[locmax_2[0]-size_box:locmax_2[0]+size_box, locmax_2[1]-size_box:locmax_2[1]+size_box]
#            polfluxsig = polfluxsig[locmax_1[0]-size_box:locmax_1[0]+size_box, locmax_1[1]-size_box:locmax_1[1]+size_box]
#            polfluxsig1 = polfluxsig1[locmax_2[0]-size_box:locmax_2[0]+size_box, locmax_2[1]-size_box:locmax_2[1]+size_box]

            polflux = (polflux+polflux1)/2
#            polfluxsig = np.sqrt(polfluxsig**2 + polfluxsig1**2)

        hdus.append(fits.ImageHDU(polflux, header=hduplusq[0].header, name="POLFLUX.IMAGE"))
        if sep_files:
            hdu_pf = fits.PrimaryHDU(polflux, header=hduplusq[0].header)
            hdu_pf.writeto(os.path.join(sep_dir, os.path.basename(name_out)+"_polflux.fits"), \
            overwrite=True)

        # Degree of linear polarization - Ref : Li et al. 2014, JQSRT
        q = (hduplusq[0].data[1, :, :] - hduminusq[0].data[1, :, :]) / \
        (hduplusq[0].data[0, :, :] + hduminusq[0].data[0, :, :])
        u = (hduplusu[0].data[1, :, :] - hduminusu[0].data[1, :, :]) / \
        (hduplusu[0].data[0, :, :] + hduminusu[0].data[0, :, :])
        poldeg = np.sqrt(q*q+u*u)
        poldeg[np.where(poldeg > 1.0)] = -1E-10 # value for bad pixels

        #If same filter on both cameras
        if hduplusq[0].header['ESO INS3 OPTI5 NAME'] == hduplusq[0].header['ESO INS3 OPTI6 NAME'] and cam == 1:
            poldeg = poldeg[locmax_1[0]-size_box:locmax_1[0]+size_box, locmax_1[1]-size_box:locmax_1[1]+size_box]
            poldeg1 = poldeg1[locmax_2[0]-size_box:locmax_2[0]+size_box, locmax_2[1]-size_box:locmax_2[1]+size_box]
            poldeg = (poldeg+poldeg1)/2

        hdus.append(fits.ImageHDU(poldeg, name="POLDEG.IMAGE"))
        if sep_files:
            hdu_pd = fits.PrimaryHDU(poldeg, header=hduplusq[0].header)
            hdu_pd.writeto(os.path.join(sep_dir, os.path.basename(name_out)+"_poldeg.fits"), \
            overwrite=True)

        # Polarization angle - Ref : Li et al. 2014, JQSRT
        polangle = np.arctan2(u, q)/2.0*(180./np.pi)

        #If same filter on both cameras
        if hduplusq[0].header['ESO INS3 OPTI5 NAME'] == hduplusq[0].header['ESO INS3 OPTI6 NAME'] and cam == 1:
            polangle = polangle[locmax_1[0]-size_box:locmax_1[0]+size_box, locmax_1[1]-size_box:locmax_1[1]+size_box]
            polangle1 = polangle1[locmax_2[0]-size_box:locmax_2[0]+size_box, locmax_2[1]-size_box:locmax_2[1]+size_box]
            polangle = np.angle((np.exp(1j*np.deg2rad(polangle))+np.exp(1j*np.deg2rad(polangle1)))/2., deg=True)

        hdus.append(fits.ImageHDU(polangle, name="POLANGLE"))
        if sep_files:
            hdu_pa = fits.PrimaryHDU(polangle, header=hduplusq[0].header)
            hdu_pa.writeto(os.path.join(sep_dir, os.path.basename(name_out)+"_polangle.fits"), \
            overwrite=True)

        hdu_list = fits.HDUList(hdus)
        hdu_list.writeto(name_out+".fits", overwrite=True, output_verify='ignore')

        #Closing inputs
        hduplusq.close()
        hduminusq.close()
        hduplusu.close()
        hduminusu.close()

def plot_polar_products(input_file, norm="lin", window_size=40, step_polar=2, lim_poldeg=1):
    """
    Plots the products of the compute polar function. On the left panel, the intensity is shown. On
    the right pannel, the image represents the polarized flux. The vector norms correspond to the
    degree of linear polarization, their direction correspond to the polarization angle in position
    angle convention (0deg to the North, 90deg to the East, *orientation to be checked*).

    **WARNING** Must be run after computepolar.

    Parameters
    ----------
    **input_file** : Complete path to the FITS file to plot

    Keywords
    ----------
    **(norm|lin)** : Colorscale for the intensity plot (supported value are lin, sqrt and log)

    **(window_size|40)** : Size (in pixels) of the squared window showm around the image intensity
    maximum.

    **(step_polar|2)** : Step to plot the vectors of linear polarization (by default, one over two)

    **(lim_poldeg|1)** : Upper limit to consider for the degree of linear polarization (default
    is one)

    Returns
    ----------
    Return the figure instance of the plot.
    """

    #Getting data
    with (fits.open(input_file)) as hdu:
        hdr = hdu[0].header
        intens = hdu['INTENSITY.IMAGE'].data
        polflux = hdu['POLFLUX.IMAGE'].data
        poldeg = hdu['POLDEG.IMAGE'].data
        polangle = hdu['POLANGLE'].data

    #Figure
    fig, (ax_int, ax_pol) = plt.subplots(1, 2, sharex=True, sharey=True)

    #WCS
    limx = hdr['NAXIS1']*hdr['CD1_1']*3600*1e3/2.
    limy = hdr['NAXIS2']*hdr['CD2_2']*3600*1e3/2.

    #Plotting intensity
    if norm == "lin":
        plot_int = ax_int.imshow(intens, origin='lower', cmap='afmhot',\
        extent=[-limx, limx, -limy, limy])
    elif norm == "sqrt":
#        intens[(intens <= 0)] = 0.
        plot_int = ax_int.imshow(intens, origin='lower', cmap='afmhot',\
        extent=[-limx, limx, -limy, limy], norm=PowerNorm(0.5))
    elif norm == "log":
#        intens[(intens <= 0)] = 1e-10
        plot_int = ax_int.imshow(intens, origin='lower', cmap='afmhot',\
        extent=[-limx, limx, -limy, limy], norm=LogNorm())
    else:
        raise ValueError("norm should be lim, sqrt or log")
    fig.colorbar(plot_int, ax=ax_int)
    #Cosmetics
    ax_int.set_xlabel('Relative RA (mas)')
    ax_int.set_ylabel('Relative Dec (mas)')
    ax_int.grid()

    #Plotting polarization
    if norm == "lin":
        plot_pol = ax_pol.imshow(polflux, origin='lower', cmap='Reds_r',\
        extent=[-limx, limx, -limy, limy])
    elif norm == "sqrt":
        plot_pol = ax_pol.imshow(polflux, origin='lower', cmap='Reds_r',\
        extent=[-limx, limx, -limy, limy], norm=PowerNorm(0.5))
    elif norm == "log":
        plot_pol = ax_pol.imshow(polflux, origin='lower', cmap='Reds_r',\
        extent=[-limx, limx, -limy, limy], norm=LogNorm())
    fig.colorbar(plot_pol, ax=ax_pol)

    #Zooming
    loc_max = np.unravel_index(intens.argmax(), intens.shape)
    whs = int(window_size/2)
    minx = (loc_max[1]-hdr['NAXIS1']/2-whs)*hdr['CD1_1']*3600*1e3
    maxx = (loc_max[1]-hdr['NAXIS1']/2+whs)*hdr['CD1_1']*3600*1e3
    miny = (loc_max[0]-hdr['NAXIS2']/2-whs)*hdr['CD2_2']*3600*1e3
    maxy = (loc_max[0]-hdr['NAXIS2']/2+whs)*hdr['CD2_2']*3600*1e3
    ax_int.set_xlim(minx, maxx)
    ax_int.set_ylim(miny, maxy)

    #Vectors
    poldeg[(poldeg > lim_poldeg)] = 0.
    x = np.linspace(minx, maxx, abs((maxx-minx)/(hdr['CD1_1']*3600*1e3)))
    y = np.linspace(miny, maxy, abs((maxy-miny)/(hdr['CD2_2']*3600*1e3)))
    vec_x = poldeg*np.cos((polangle+90)*np.pi/180.)
    vec_y = poldeg*np.sin((polangle+90)*np.pi/180.)
    vec_x = vec_x[loc_max[0]-whs:loc_max[0]+whs, loc_max[1]-whs:loc_max[1]+whs]
    vec_y = vec_y[loc_max[0]-whs:loc_max[0]+whs, loc_max[1]-whs:loc_max[1]+whs]
    ax_pol.quiver(x[::step_polar], y[::step_polar], vec_x[::step_polar, ::step_polar], \
    vec_y[::step_polar, ::step_polar], pivot='mid', headaxislength=0, headwidth=0, headlength=0)
    #Cosmetics
    ax_pol.set_xlabel('Relative RA (mas)')

    #ID
    filename = os.path.basename(input_file)
    filt = filename[filename.find("REDUCED")+8:filename.find("_POLAR")]
    fig.suptitle(hdr['OBJECT']+" - "+filt)

    return fig

def plot_all_polar_products(input_dir, form, norm="lin", window_size=40, step_polar=2, \
    lim_poldeg=0.1):
    """
    This function plots the polarization products for an entire dataset reduced by ESOREFLEX
    (this mean all target/filters reduced by the pipeline).

    **WARNING** Must be run after computepolar.

    Parameters
    ----------
    **input_file** : Complete path to the product directory containing all the SPH*tpl directories
    for each target/filter.

    **form** : Format of the files containg the plots (pdf, jpg, png, ...)

    Keywords
    ----------
    **(norm|lin)** : Colorscale for the intensity plot (supported value are lin, sqrt and log)

    **(window_size|40)** : Size (in pixels) of the squared window showm around the image intensity
    maximum.

    **(step_polar|2)** : Step to plot the vectors of linear polarization (by default, one over two)

    **(lim_poldeg|1)** : Upper limit to consider for the degree of linear polarization (default
    is one)

    See also
    ----------
    *plot_polar_products*
    """

    #Listing products files
    list_files = glob(os.path.join(input_dir, "*/*POLAR_COMPUTED.fits"))


    #For each file
    if len(list_files) == 0:

        print("\nNo SPH*tpl directories containing *POLAR_COMPUTED.fits")
        print("in the current input directory.")
        print("\nCheck your path ! Aborting...\n")

    else:

        plt.ioff()

        for cur_file in list_files:

            #Plotting
            fig = plot_polar_products(cur_file, window_size=window_size, step_polar=step_polar,\
            lim_poldeg=lim_poldeg, norm=norm)
            fig.set_size_inches(16, 5)
            fig.savefig(cur_file[:cur_file.rfind("_POLAR")]+"."+form, bbox_inches='tight', dpi=300)
            plt.close(fig)

        plt.ion()

def angle_map(shape, centerx=None, centery=None, verbose=True, fullOutput=False, plot=False):
    """
    Creates a 2d array with the angle in rad (0 is North, then the array is positive going East
    and becomes negative after we pass South).

    Parameters
    ----------

    **shape**: A tuple indicating the desired shape of the output array, e.g. (100,100)
                The 1st dim refers to the y dimension and the 2nd to the x dimension

    Keywords
    --------

    **(centerx|None)**: the center of the frame from which to compute the
    distance from by default shape[1]/2 (integer division). Accepts numerical value

    **(centery|None)**: same for the y dimension

    **(verbose|True)**: to print a warning for even dimensions

    **(fullOutput|False)**: if True returns the angle array and in addition
    the 2d array of x values and y values in 2nd and 3rd ouptuts.

    **(plot|False)**: if set to True, plot the angle map
    """

    #Warnings about the center
    if len(shape) != 2 :
        raise ValueError('The shape must be a tuple of 2 elements for the y and x dimension!')
    if centerx == None:
        centerx = shape[1]//2
        if np.mod(shape[1],2) == 0 and verbose:
            print('The X dimension is even ({:d}), the center is assumed to be in {:d}'.format(shape[1],centerx))
    if centery == None:
        centery = shape[0]//2
        if np.mod(shape[0],2) == 0 and verbose:
            print('The Y dimension is even ({:d}), the center is assumed to be in {:d}'.format(shape[0],centery))

    #1D arrays
    x_array = np.arange(shape[1])-centerx
    y_array = np.arange(shape[0])-centery

    #Mesh grid
    xx_array,yy_array = np.meshgrid(x_array,y_array)

    #Angle
    theta = -np.arctan2(xx_array,yy_array)

    if plot:
        fig, ax = plt.subplots(1, 1)
        pl = ax.imshow(theta, origin='lower')
        fig.colorbar(pl, ax=ax, label='Angle (rad)')

    #Return
    if fullOutput:
        return theta,xx_array,yy_array
    else:
        return theta

def sphere_cut(hdul, new_FOV, nogauss=False, pix_size=3.6006):

    """
    A function to take a cut out of SPHERE data centred on the star

    Parameters
    ----------

    **hdul** : HDUlist instance

    **new_FOV** : FOV in mas

    Keywords
    --------

    **(nogauss|False)**: When set to False the star is identified by fitting a
    gaussian to the intensity image. If True, the max intensity defines the
    star center.

    **(pix_size|3.6006)** pixel size in mas/pix

    Returns
    ----------
    Clips of observations
    """

    #Getting header and intensity
    hdr = hdul[0].header
    intens = hdul[0].data

    #Calculate fov in pixels
#    radius_px_dec =  new_FOV / (hdr['PIXSCAL'])  #should be divided by 2 but pixscal seems to be twice as large
#    radius_px = round(radius_px_dec)
    lim_pix = int(np.abs(new_FOV/pix_size))


    # Fit gaussian to find centre
    if nogauss == True:
        loc_max = np.unravel_index(intens.argmax(), intens.shape)
        center_y = loc_max[0]
        center_x = loc_max[1]
    else:
        raise ValueError('nogauss=False not supported')
        # params_1 = fitgaussian(intens)
        # centre_y = int(round(params_1[1]))
        # centre_x = int(round(params_1[2]))

    # Cut out central clip
    for hdu in hdul:
        hdu.data = hdu.data[center_y-lim_pix:center_y+lim_pix, center_x-lim_pix:center_x+lim_pix]
        hdu.header['NAXIS1'] = lim_pix*2
        hdu.header['NAXIS2'] = lim_pix*2

    return hdul

def compute_uphi_qphi(reflex_product_dir, mode, merge_common=True):
    """
    Computing the U_PHI and Q_PHI maps

    :math:`Q_\\phi = Q\\cos(2\\phi) + U\\sin(2\\phi)`

    :math:`U_\\phi = -Q\\sin(2\\phi) + U\\cos(2\\phi)`

    with PHI the position angle.

    Parameters
    ----------
    **reflex_product_directory** : Directory containing the directories with the SPHERE reduced
    cubes

    **mode** : Should be "esoreflex" or "dc" depending on the source of the data

    Keywords
    --------
    **(merge_common|True)** If True, merge common filters on the 2 arms

    See also
    --------
    U_phi/Q_phi : Engler et al. 2017, A&A, 607, A90
    Pol deg/angle : Li et al. 2014, JQSRT, https://doi.org/10.1016/j.jqsrt.2014.09.003
    """

    if not locale.getlocale(locale.LC_TIME) == ('en_US', 'UTF-8'):
      locale.setlocale(locale.LC_TIME, "en_US.utf8")

    #Listing templates
    if mode == 'esoreflex':
        list_prod = glob(os.path.join(reflex_product_dir, "SPH*tpl"))
    elif mode == 'dc':
        list_prod = glob(os.path.join(reflex_product_dir, '*zpl_science_p23*'))
    else:
        raise ValueError('mode should be esoreflex or dc')
    list_prod.sort()

    #For each template
    for cur_tpl in list_prod:

        #Listting product files
        if mode == 'esoreflex':
            tpl_file = glob(os.path.join(cur_tpl, "*ZPL_SCIENCE_P23_REDUCED_Q_CAM1.fits"))
            tpl_file = tpl_file[0][:-11]
        else:
            tpl_file = os.path.join(cur_tpl, 'zpl_science_p23-ZPL_SCIENCE_P23_REDUCED-sci')
        if len(tpl_file) == 0:
            continue

        for cam in range(2):

            if mode == 'esoreflex':

                #Opening input files
                try:
                    hduq = fits.open(tpl_file+"Q_CAM"+str(cam+1)+".fits", mode='readonly')
                    # hduplusq = fits.open(tpl_file+"QPLUS_CAM"+str(cam+1)+".fits", mode='readonly')
                    # hduminusq = fits.open(tpl_file+"QMINUS_CAM"+str(cam+1)+".fits", mode='readonly')
                    hduu = fits.open(tpl_file+"U_CAM"+str(cam+1)+".fits", mode='readonly')
                    # hduplusu = fits.open(tpl_file+"UPLUS_CAM"+str(cam+1)+".fits", mode='readonly')
                    # hduminusu = fits.open(tpl_file+"UMINUS_CAM"+str(cam+1)+".fits", mode='readonly')
                except FileNotFoundError:
                    wa.warn(os.path.basename(cur_tpl)+" is missing some files")
                    continue

            else:

                with fits.open(tpl_file+str(cam+1)+'.fits') as sci_hdu:
                    img_sci = sci_hdu[0].data
                    hdr_sci = sci_hdu[0].header
                    hdr_sci_prim = sci_hdu[0].header

                hdr_sci_prim['EXTNAME'] = 'IFRAME.IMAGE'
                hduq = fits.HDUList([
                    fits.PrimaryHDU(img_sci[0, :, :], header=hdr_sci_prim),
                    fits.ImageHDU(img_sci[1, :, :], name='PFRAME.IMAGE')
                    ])
                hduu = fits.HDUList([
                    fits.PrimaryHDU(img_sci[2, :, :], header=hdr_sci_prim),
                    fits.ImageHDU(img_sci[3, :, :], name='PFRAME.IMAGE')
                    ])

            #Now
            now = datetime.datetime.now()
            now_history = now.strftime("%d %b %Y %H:%M:%S")

            #Cropping and centering
            hduq = sphere_cut(hduq, 500, nogauss=True)
            hduu = sphere_cut(hduu, 500, nogauss=True)

            #Writting output : 1 file per camera
            if cam == 0:
                filt = hduq[0].header['ESO INS3 OPTI5 NAME']
            else:
                filt = hduq[0].header['ESO INS3 OPTI6 NAME']
            if mode == 'esoreflex':
                name_out = tpl_file+filt+"_UPHI_QPHI"
            else:
                name_out = tpl_file+'_'+filt+"_UPHI_QPHI"

            #Updating header
            upd_hdr = "UPHI-QPHI computed on "+now_history+" by "+\
            os.path.basename(os.getenv("HOME"))
            hduq[0].header['HISTORY'] = upd_hdr
            hduu[0].header['HISTORY'] = upd_hdr

            # =============================================================================
            #                        INTENSITY
            # =============================================================================
            if mode == 'esoreflex':
                hduq_un = unumpy.uarray(hduq['IFRAME.IMAGE'].data, hduq['IFRAME.RMSMAP'].data)
                hduu_un = unumpy.uarray(hduu['IFRAME.IMAGE'].data, hduu['IFRAME.RMSMAP'].data)
            else:
                hduq_un = hduq['IFRAME.IMAGE'].data
                hduu_un = hduu['IFRAME.IMAGE'].data

            #Combining
            iframe_un = (hduq_un + hduu_un)/2
            if mode == 'esoreflex':
                iframe = unumpy.nominal_values(iframe_un)
                iframesig = unumpy.std_devs(iframe_un)
            else:
                iframe = iframe_un

            hdr_primary = hduq[0].header
            hdr_primary['EXTNAME'] = "INTENSITY.IMAGE"

            #If same filter on both cameras
            if hduq[0].header['ESO INS3 OPTI5 NAME'] == hduq[0].header['ESO INS3 OPTI6 NAME'] and cam == 1 and merge_common:

                #Opening computed CAM1
                with fits.open(name_out+'.fits') as hdul1:
                    iframe1 = hdul1['INTENSITY.IMAGE'].data
                    poldeg1 = hdul1['POLDEG.IMAGE'].data
                    poldegsig1 = hdul1['POLDEG.RMSMAP'].data
                    polangle1 = hdul1['POLANGLE.IMAGE'].data
                    if mode == 'esoreflex':
                        iframesig1 = hdul1['INTENSITY.RMSMAP'].data

                    # No need to change DIT or EXPTIME as we average
                    # hdr_primary['ESO DET SEQ1 EXPTIME'] += hdul1[0].header['ESO DET SEQ1 EXPTIME']
                    # hdr_primary['ESO DET DIT1'] += hdul1[0].header['ESO DET DIT1']
                    hdr_primary['COMMENT'] = 'This file is the result of the average of the two arms of ZIMPOL'

                if 0 in iframe.shape or 0 in iframe1.shape:
                    print("Error with "+name_out)
                    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
                    axes[0].imshow(iframe1, cmap='cubehelix', origin='lower')
                    axes[1].imshow(iframe, cmap='cubehelix', origin='lower')
                    sys.exit(10)

                #Summing
                if mode == 'esoreflex':
                    iframe_un = unumpy.uarray(iframe, iframesig)
                    iframe1_un = unumpy.uarray(iframe1, iframesig1)
                    iframe_avg_un = (iframe_un+iframe1_un)/2
                    iframe = unumpy.nominal_values(iframe_avg_un)
                    iframesig = unumpy.std_devs(iframe_avg_un)
                    iframe_un = iframe_avg_un
                else:
                    iframe = (iframe+iframe1)/2

            hdu_I = fits.PrimaryHDU(iframe, header=hdr_primary)
            if mode == 'esoreflex':
                hdu_Irms = fits.ImageHDU(iframesig, name="INTENSITY.RMSMAP")

            # =============================================================================
            #              Q_PHI/U_PHI / Engler et al. 2017, A&A, 607, A90
            # =============================================================================

            # Q/U
            Q = hduq['PFRAME.IMAGE'].data
            U = hduu['PFRAME.IMAGE'].data

            # Angle map
            phi = angle_map(Q.shape)

            #U_phi/Q_phi
            Q_phi = (Q*np.cos(2*phi) + U*np.sin(2*phi))
            U_phi = np.abs(-Q*np.sin(2*phi) + U*np.cos(2*phi))

            #If same filter on both cameras
            if hduq[0].header['ESO INS3 OPTI5 NAME'] == hduq[0].header['ESO INS3 OPTI6 NAME'] and cam == 1:

                #Opening computed CAM1
                with fits.open(name_out+'.fits') as hdul1:
                    uphi1 = hdul1['UPHI.IMAGE'].data
                    qphi1 = hdul1['QPHI.IMAGE'].data

                    hdr_primary['ESO DET SEQ1 EXPTIME'] += hdul1[0].header['ESO DET SEQ1 EXPTIME']

                    #Averaging
                    Q_phi = (Q_phi+qphi1)/2
                    U_phi = (U_phi+uphi1)/2

            #Uarray
            polflux_un = unumpy.uarray(np.abs(Q_phi), U_phi)
            #Polarization HDUs
            hdu_Qphi = fits.ImageHDU(Q_phi, name='QPHI.IMAGE')
            hdu_Uphi = fits.ImageHDU(U_phi, name='UPHI.IMAGE')


            # =============================================================================
            #      DEGREE OF LINEAR POLARIZATION - Ref : Li et al. 2014, JQSRT
            #             https://doi.org/10.1016/j.jqsrt.2014.09.003
            # =============================================================================
            if mode == 'esoreflex':
                poldeg_un = polflux_un/iframe_un
            else:
                poldeg_un = polflux_un/iframe
            poldeg = unumpy.nominal_values(poldeg_un)
            tag_bad = poldeg > 1.0
            poldeg[tag_bad] = np.nan# value for bad pixels
            poldegsig = unumpy.std_devs(poldeg_un)
            poldegsig[tag_bad] = np.nan

            #If same filter on both cameras
            if hduq[0].header['ESO INS3 OPTI5 NAME'] == hduq[0].header['ESO INS3 OPTI6 NAME'] and cam == 1 and merge_common:
                poldeg_un = unumpy.uarray(poldeg, poldegsig)
                poldeg1_un = unumpy.uarray(poldeg1, poldegsig1)
                poldeg_avg_un = (poldeg_un+poldeg1_un)/2
                poldeg = unumpy.nominal_values(poldeg_avg_un)
                poldegsig = unumpy.std_devs(poldeg_avg_un)

            hdu_DOLP = fits.ImageHDU(poldeg, name="POLDEG.IMAGE")
            hdu_DOLPsig = fits.ImageHDU(poldegsig, name="POLDEG.RMSMAP")


            # =============================================================================
            #      POLARIZATION ANGLE - Ref : Li et al. 2014, JQSRT
            #             https://doi.org/10.1016/j.jqsrt.2014.09.003
            # =============================================================================
            polangle = np.arctan2(U, Q)/2.0*(180./np.pi)

            #If same filter on both cameras
            if hduq[0].header['ESO INS3 OPTI5 NAME'] == hduq[0].header['ESO INS3 OPTI6 NAME'] and cam == 1 and merge_common:
                polangle = np.angle((np.exp(1j*np.deg2rad(polangle))+np.exp(1j*np.deg2rad(polangle1)))/2., deg=True)
            hduAngle = fits.ImageHDU(polangle, name='POLANGLE.IMAGE')

            # =============================================================================
            #             WRITTING OUTPUT AND CLOSING INPUTS
            # =============================================================================
            if mode == 'esoreflex':
                hdu_list = fits.HDUList([hdu_I, hdu_Irms, hdu_Qphi, hdu_Uphi, hdu_DOLP, hdu_DOLPsig, hduAngle])
            else:
                hdu_list = fits.HDUList([hdu_I, hdu_Qphi, hdu_Uphi, hdu_DOLP, hdu_DOLPsig, hduAngle])
            hdu_list.writeto(name_out+".fits", overwrite=True, output_verify='ignore')
            apply_WCS(name_out+'.fits')

            #Closing inputs
            hduq.close()
            hduu.close()
