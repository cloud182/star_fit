#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 16:37:57 2018

@author: congiu
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import os





if __name__ == '__main__':
    
    input_dir = './PSF_tests/Stellar_PSF/'
    
    plt.figure()
    plt.ylim(0.5,1.5)
    
    for file in os.listdir(input_dir):
        data = ascii.read(input_dir+file, delimiter = ' ')
        
        plt.scatter(data['wavelength'], 
                    data['fwhm_y']/np.median(data['fwhm_y']), 
                    label = file[7]+file[10:-4])
    plt.legend(loc=9, bbox_to_anchor=(1.3, 1), ncol=1)
    plt.xlabel('Wavelength ($\\AA$)')
    plt.ylabel('normalized FWHM')
    plt.savefig('all_fwhm.png',additional_artists=[],
    bbox_inches="tight")
    plt.close()
    