#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 02:58:47 2019

@author: congiu
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii


def fwhm_atm(s,x,lam):
    """
    Modeling theorethical fwhm as a function of airmass
    reference: ESO ETC
    """
    
    r0 = 0.1*s**(-1)*(lam/5000)**1.2*x**(-0.6)
    a = np.sqrt(1-0.981644*2.183*(r0/46)**0.356)
    fwhm = s*x**0.6*(lam/5000)**(-0.2)*a
    return fwhm

def fwhm_tel(lam, D = 8.2):
    """
    FWHM of the telescope.
    Diameter in meters
    """
    D = D*1e10
    fwhm = 0.000212*lam/D
    return fwhm

def fwhm(s, x, lam):
    
    fwhm1 = fwhm_atm(s,x,lam)
    fwhm2 = fwhm_tel(lam)
    fwhm = np.sqrt(fwhm1**2+fwhm2**2)    

    return fwhm

    
if __name__ == '__main__':
    
    galaxy = 'IC5332'
    pointing = 'P04'
    x = '78'
    y = '256'
    
    info_name = '/media/congiu/Data/Dati/PHANGS/star_fit/info_header'
    info_table = ascii.read(info_name)
    
    directory = '/media/congiu/Data/Dati/PHANGS/star_fit/plots/'
    file = directory+'Moffat_'+galaxy.upper()+'_'+pointing+'_x'+x+'_y'+y+'.txt'
    psf_lam = ascii.read(file)
    
    for i, name in enumerate(info_table['file']):
        if galaxy.upper() in name and pointing in name:
            seeing = info_table['seeing'][i]
            airmass = info_table['airmass'][i]
    
    fwhm = fwhm(seeing, airmass, psf_lam['wavelength'])
    plt.scatter(psf_lam['wavelength'],psf_lam['fwhm_y'], label = 'measured')
    plt.scatter(psf_lam['wavelength'],fwhm, label = 'model') 
    plt.legend(loc='best')
    plt.show()       
    
    
    
    
    
    
    