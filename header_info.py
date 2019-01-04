# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from mpdaf.obj import Cube

import os

f = open('/media/congiu/Data/Dati/PHANGS/star_fit/info_header', 'w')
print('file', 'MJD','airmass','seeing','rot', file = f)
directory ='/media/congiu/Data/Dati/PHANGS/Cubes'

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.startswith('DATACUBE'):
        cube = Cube(os.path.join(directory,filename))
        mjd = cube.primary_header['MJD-OBS']
        air_start = cube.primary_header['HIERARCH ESO TEL AIRM START']
        air_end = cube.primary_header['HIERARCH ESO TEL AIRM END']  
        airmass = (air_start+air_end)/2
        fwhm_start = cube.primary_header['HIERARCH ESO TEL AMBI FWHM START']  
        fwhm_end = cube.primary_header['HIERARCH ESO TEL AMBI FWHM END']  
        seeing = (fwhm_start+fwhm_end)/2
        rot = cube.primary_header['HIERARCH ESO TEL AMBI FWHM END']
        print(filename[15:-9], mjd, airmass, seeing, rot, file = f)
f.close()