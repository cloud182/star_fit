#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:00:15 2018

@author: congiu
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization as vis

from astropy.io import fits,ascii 
from astropy.wcs import WCS, utils
from astropy.nddata.utils import Cutout2D
from astropy.coordinates import SkyCoord,FK5
from astropy.modeling import models, fitting
from scipy.interpolate import interp1d

from astropy.stats import gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma
import sys

########################################################################

@models.custom_model
def moffat_2D(x, y, I0 = 1, x0 = 0, y0 = 0, fwhm = 1, e = 1, theta = 0, 
               c = 0, beta = 1):
    """
    Two dimensional elliptical Moffat model function
    FWHM measured in pixels
    """
    fwhm = fwhm
    alpha = fwhm / (2 * np.sqrt(2 ** (1.0 / beta) - 1.0))
    theta = theta * np.pi/180
    cost = np.cos(theta)
    sint = np.sin(theta)
    xdiff = x - x0
    ydiff = y - y0
    rr_gg = (((ydiff * cost - xdiff * sint) / alpha) ** 2 +
             ((ydiff * sint + xdiff * cost) / alpha / e) ** 2)
    # the function includes the continuum
    moffat = c + I0 * (1 + rr_gg) ** (-beta)
    return moffat

########################################################################

@models.custom_model
def gaussian_2D( x, y, I0=10., x0=0, y0=0, fwhm=1, e=1., theta=0., c = 0.):
    """ 2D elliptical Gaussian model"""
    #converting from arcsec to pixel using 0.2"/px scale
    fwhm = fwhm 
    xdiff = x - x0
    ydiff = y - y0
    theta = theta * np.pi/180
    cost = np.cos(theta)
    sint = np.sin(theta)
    sigma_y = (fwhm*gaussian_fwhm_to_sigma)**2
    sigma_x = (fwhm*gaussian_fwhm_to_sigma*e)**2
    gauss = c+I0*np.exp(-(ydiff * cost - xdiff * sint)**2/(2*sigma_y))\
            * np.exp(-(ydiff * sint + xdiff * cost)**2/(2*sigma_x))
    return gauss    

########################################################################
    
def fit_error(fitter, model):
    
    cov = fitter.fit_info['param_cov']
    error = dict(zip(model.param_names, np.diag(cov)**0.5))
    return error
        
########################################################################

def open_image(hdu, extension, err = True, debug=False):
    """
    Function to open fits file.
    Input:
    - filename
    - extension name or number
    - err
    Output:
    - image
    - header
    - wcs
    - image_err (only if the keyword err = True)
    """
    if type(extension) != int:
        if extension.lower() =='av':
            extension = -1
        else:
            extension = extension_locator(extension) 
    if debug:
        print(extension)       
    image = hdu[extension].data
    if len(np.shape(image)) == 3:
        image = image[0,:,:]
    header = hdu[extension].header
    header['NAXIS'] = 2
    try:
        del header['NAXIS3']
    except:
        pass
    wcs = WCS(header)
    if err:
        image_err = hdu[extension+1].data
        image_err = image_err[0,:,:]
        return image, header, wcs, image_err
    else:    
        return image, header, wcs
########################################################################   
def read_table(filename):
    """
    Open the catalogue of the region and it saves the information 
    needed for the plots.
    Input
    -filename
    Output
    -table['PHANGS_INDEX']: region number
    -table['RA']: region RA
    -table['DEC']: region DEC
    -table['SIZE_OBS']: dimension of the region
    """
    hdu = fits.open(filename)
    table = hdu[1].data
    return table['PHANGS_INDEX'], table['RA'], table['DEC'], table['SIZE_OBS']

######################################################################## 
    
def select_cutout(image, wcs):
    
    #I'm looking for how many pointings are in the mosaic 
    #I don't know if it is always accurate
    nrow = image.shape[0]//300.
    ncol = image.shape[1]//300.
    #measuring the exact width of a row and of a column
    drow = image.shape[0]/nrow
    dcol = image.shape[1]/ncol
    #I'm showing the image to select the correct section
    #I'm picking the center with a mouse click (maybe)
    
    fig, ax = plt.subplots(1,1)
    interval = vis.PercentileInterval(99.9)
    vmin,vmax = interval.get_limits(image)
    norm = vis.ImageNormalize(vmin=vmin, vmax=vmax, 
                              stretch=vis.LogStretch(1000))
    ax.imshow(image, cmap =plt.cm.Greys, norm = norm, origin = 'lower')
    for x in np.arange(0,image.shape[1],dcol):
        ax.axvline(x)
    for y in np.arange(0,image.shape[0],drow):
        ax.axhline(y)

    def onclick(event):
        
        ix, iy = event.xdata, event.ydata
        col = ix//300.
        row = iy//300.
        print(col, row)
        global x_cen, y_cen
        x_cen = 150+300*(col) #x of the center of the quadrans
        y_cen = 150+300*(row) #y of the center of thw quadrans
        print('x: {:3.0f}, y: {:3.0f}' .format(x_cen, y_cen))
        if event.key == 'q':
            fig.canvas.mpl_disconnect(cid)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    nrow = image.shape[0]//300.
    ncol = image.shape[1]//300.
    print(image.shape[0]/nrow)
    x = int(x_cen)
    y = int(y_cen)
    print(x, y)
    cutout = Cutout2D(image, (x, y), size = (image.shape[0]/nrow -20)* u.pixel, 
                      wcs = wcs)
    return cutout

########################################################################
    
def measure_PSF(cutout, catalogue, function):
    
    regions, ra, dec, size = read_table(catalogue) 
    fitter = fitting.LevMarLSQFitter()
    
    fwhm_x = []
    fwhm_y = []
    e = []
    theta = []
    for (region,ra_r,dec_r,size_r) in zip(regions, ra, dec, size):
        coords = SkyCoord(ra_r, dec_r, frame= FK5, unit=(u.deg,u.deg))
        fwhm = size_r * 5 #fwhm in pixels
        if cutout.wcs.footprint_contains(coords):
            #converting coords to pixels
            x0, y0 = utils.skycoord_to_pixel(coords, cutout.wcs)
            #selecting the model
            if function == 'gauss':
                #adding a continuum to the gaussian fit
                model = gaussian_2D(40, x0, y0, fwhm, 1, 0, 10)
            elif function == 'moff':
                model = moffat_2D(40, x0, y0, fwhm = fwhm, beta = 1, e = 1, 
                                   theta = 0, c = 10 )
            X, Y = np.indices(np.shape(cutout.data))
            fit_result = fitter(model, X, Y, cutout.data) 
#            error = fit_error(fitter, model)
            if function == 'gauss':
                if fit_result.e < 1:
                    fwhm_x.append(fit_result.fwhm[0]* fit_result.e[0])
                    fwhm_y.append(fit_result.fwhm[0])
                else:
                    fwhm_x.append(fit_result.fwhm[0])
                    fwhm_y.append(fit_result.fwhm[0]* fit_result.e[0])
            elif function == 'moff':
                if fit_result.e < 1:
                    fwhm_x.append(fit_result.fwhm[0] * fit_result.e[0])
                    fwhm_y.append(fit_result.fwhm[0])
                else:
                    fwhm_x.append(fit_result.fwhm[0])
                    fwhm_y.append(fit_result.fwhm[0] * fit_result.e[0])
            e.append(fit_result.e[0])
            theta.append(fit_result.theta[0])

    return np.array(fwhm_x)*0.2, np.array(fwhm_y)*0.2, np.array(e), np.array(theta)
      
########################################################################

def fwhm_distribution(fwhm_x, fwhm_y, psf_file, wave):
    
    data = ascii.read(psf_file)
    fwhm_interp_x = interp1d(data['wavelength'], data['fwhm_x'])
    fwhm_interp_y = interp1d(data['wavelength'], data['fwhm_y'])
    fwhm_line_x = fwhm_interp_x(wave)
    fwhm_line_y = fwhm_interp_y(wave)
    
    bins = np.arange(np.min(fwhm_y),np.max(fwhm_y)+0.2,0.2)
    plt.hist([fwhm_x,fwhm_y], bins=bins,label=['FWHM_x', 'FWHM_y'])
    plt.axvline(fwhm_line_x)
    plt.axvline(fwhm_line_y)
    plt.legend(loc = 'best')
    plt.show()
    

    print('data_median x: {:1.3f}, data_mean x: {:1.3f}, sigma x: {:1.3f} '
          .format(np.median(fwhm_x), np.mean(fwhm_x), np.std(fwhm_x))
          +'PSF_fwhm x: {:1.3f}' .format(fwhm_line_x))
    print('data_median y: {:1.3f}, data_mean y: {:1.3f}, sigma y: {:1.3f} '
          .format(np.median(fwhm_y), np.mean(fwhm_y), np.std(fwhm_y))
          +'PSF_fwhm y: {:1.3f}' .format(fwhm_line_y))
    
if __name__ == '__main__':
    
    filename = '../Mosaics/DATACUBE_FINAL_AVC_NGC1087_1_comp_mosaic.fits'
    hdu = fits.open(filename)
    ha, header, wcs = open_image(hdu, 20, err = False)
    catalogue = '../catalogues_and_masks/ngc1087_hiicat_v1.fits'
    cutout = select_cutout(ha, wcs)
    fwhm_x, fwhm_y, e, theta = measure_PSF(cutout, catalogue, function = 'gauss')
    psf_file = 'Moffat_NGC1087_P01-003.txt'
    fwhm_distribution(fwhm_x,fwhm_y, psf_file, 6562.)
    
    
    
    
    