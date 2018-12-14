#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 16:00:15 2018

@author: congiu
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits 
from astropy.wcs import WCS
from astropy.nddata.utils import Cutout2D

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
    
    

if __name__ == '__main__':
    
    filename = '../Mosaics/DATACUBE_FINAL_AVC_NGC1087_1_comp_mosaic.fits'
    hdu = fits.open(filename)
    ha, header, wcs = open_image(hdu, 20, err = False)
    nrow = int(input('pointings on y axis: '))
    print('The size of the cutout is '+str(ha.shape[0]/nrow))
    x = int(input('x0: '))
    y = int(input('y0: '))
    cutout = Cutout2D(ha, (x, y), size = ha.shape[0]/nrow * u.pixel, wcs = wcs)
    
    