# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from mpdaf.obj import Cube

import os

f = open('/media/congiu/Data/Dati/PHANGS/star_fit/info_header', 'w')
f2 =  open('/media/congiu/Data/Dati/PHANGS/star_fit/pointing_coords', 'w')
print('file', 'MJD','airmass','seeing1', 'seeing2','seeing3','seeing_zen', file = f)
print('file', 'RA', 'DEC', file = f2)

directory ='/media/congiu/Data/Dati/PHANGS/Cubes'

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)
    if filename.startswith('DATACUBE'):
        cube = Cube(os.path.join(directory,filename))
        mjd = cube.primary_header['MJD-OBS']
        air_start = cube.primary_header['HIERARCH ESO TEL AIRM START']
        air_end = cube.primary_header['HIERARCH ESO TEL AIRM END']  
        airmass = (air_start+air_end)/2
        fwhm_start = cube.primary_header['HIERARCH ESO TEL AMBI FWHM START']  
        fwhm_end = cube.primary_header['HIERARCH ESO TEL AMBI FWHM END']  
        seeing_zen = (fwhm_start+fwhm_end)/2
        seeing1 = cube.primary_header['HIERARCH ESO TEL IA FWHM']
        seeing2 = cube.primary_header['HIERARCH ESO TEL IA FWHMLIN']
        seeing3 = cube.primary_header['HIERARCH ESO TEL IA FWHMLINOBS']
        print(filename[15:-5], mjd, airmass, seeing1, seeing2, seeing3, seeing_zen, file = f)
        ra = cube.primary_header['RA']
        dec = cube.primary_header['DEC']
        print(filename, ra, dec, file = f2)
f.close()