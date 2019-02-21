#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 11:19:46 2019

@author: congiu
"""
import numpy as np
import matplotlib.pyplot as plt
import functions_v1 as fu
import astropy.units as u

from mpdaf.obj import Cube


class PSF(object):
    
    def __init__(self, fwhm, err_fw, n, err_n):
        self.fwhm = fwhm
        self.err_fwhm = err_fw
        self.n = n
        self.err_n = err_n

def white_image(cube):
    
    start, end = fu.wavelength_range(cube)
    
    ima = cube.get_image((start,end))

    return ima

def select_star(image):

    fig = plt.figure(figsize = (12,10))
    gs = fig.add_gridspec(3,4)
    ax1 = fig.add_subplot(gs[:,0:3])
    ax2 = fig.add_subplot(gs[0,3])
    ax3 = fig.add_subplot(gs[1,3])
    ax4 = fig.add_subplot(gs[2,3])
    image.plot(ax = ax1, zscale = True)
    fwhm = []
    err_fwhm=[]
    n = []
    err_n = []
    t1 = ax1.text(20,20, 'x: , y:')
    def onclick(event):
        if event.button == 1:
            global fwhm_fit , e_fwhm, n_fit, e_n
            x, y = event.xdata, event.ydata
            #first I perform a fit in order to find the center of the region
            newimage = image.subimage((y,x), size = 3, unit_center = None)
            fit, fitima, chi2 = fu.fit_function(newimage, circular = True)
            #then with the correct coordinates of the center I create 
            #a new image and perform the fit once again          
            newimage = image.subimage(fit.center, size = 3)
            fit, fitima, chi2 = fu.fit_function(newimage, circular = True)
            # to evaluate the fit I'm plotting the selected object,
            # the fit and the ratio between the two
            vmin = np.nanmin(newimage.data)
            vmax = np.nanmax(newimage.data)
            newimage.plot(ax = ax2, zscale = False, 
                          vmin = vmin, 
                          vmax = vmax, colorbar = 'v')
            fitima.plot(ax = ax3, zscale = False, 
                          vmin = vmin, 
                          vmax = vmax, colorbar = 'v'  )
            newimage.unit = u.dimensionless_unscaled
            ratio = newimage/fitima
            ratio.plot(ax=ax4, vmin = 0.8, vmax = 1.2, colorbar = 'v')
            x, y = image.wcs.wcs.wcs_world2pix(fit.center[1],fit.center[0],0)
            t1.set_text('x: {:0.0f}, y: {:0.0f}' 
                     .format(x, y))
            plt.draw()
            
            #this should be the position of the center of the star in the 
            #datacube

            fwhm_fit = fit.fwhm[0]
            e_fwhm = fit.err_fwhm[0]
            n_fit = fit.n
            e_n = fit.err_n
            #I'd like to plot an x to see which regions I've already measured
            print('x: {:1.2f}, y: {:1.2f}, fwhm: {:1.3f}+-{:1.3f}, n: {:1.3f}+-{:1.3f}' 
                  .format(x, y, fwhm_fit,e_fwhm, n_fit,e_n))

        
    def press(event):
        if event.key == 'a':
            print('Saved')
            fwhm.append(fwhm_fit)
            err_fwhm.append(e_fwhm)
            n.append(n_fit)
            err_n.append(e_n)

        
    cid0 = fig.canvas.mpl_connect('button_press_event', onclick)
    cid1 = fig.canvas.mpl_connect('key_press_event', press)


    plt.show()

    fwhm = np.array(fwhm)
    err_fwhm=np.array(err_fwhm)
    n = np.array(n)
    err_n = np.array(err_n)

    # removing nan and values that cannot be real, to clean the data a little
    # everytime I found something wrong I'll remove the corrispending value
    # in both arrays
    idx = [i for i, arr in enumerate(fwhm) if np.isnan(arr) or arr>4]
    fwhm = np.delete(fwhm, idx)
    err_fwhm = np.delete(err_fwhm, idx)
    n = np.delete(n, idx)
    err_n = np.delete(err_n, idx)
    
    idy = [i for i, arr in enumerate(n) if np.isnan(arr) or np.isinf(arr)] 
    fwhm = np.delete(fwhm, idy)
    err_fwhm = np.delete(err_fwhm, idy) 
    n = np.delete(n, idy)
    err_n = np.delete(err_n, idy)       
    
    if len(fwhm) > 1:
        fwhm_mean = np.average(fwhm, weights = 1/err_fwhm)
        fwhm_std = np.sqrt(np.average((fwhm-fwhm_mean)**2, 
                                      weights = 1/err_fwhm))
    elif len(fwhm) == 1:
        fwhm_mean = fwhm[0]
        fwhm_std = err_fwhm[0]
    else:
        return


    if len(n)>1:
        n_mean = np.average(n, weights = 1/err_n)
        n_std = np.sqrt(np.average((n-n_mean)**2, weights = 1/err_n))
    elif len(n)==1:
        n_mean = n[0]
        n_std = err_n[0]
    else:
        return
    
    print('weighted average fwhm: {:1.3f}+-{:1.3f}' 
          .format(fwhm_mean,fwhm_std))
    print('weighted average n: {:1.3f}+-{:1.3f}' .format(n_mean,n_std))
    
    psf = PSF(fwhm_mean, fwhm_std, n_mean, n_std)
    
    return psf
    

if __name__ == '__main__':
    galaxy = 'NGC1087'
    point   = '03'
    
    filename = '../Cubes/DATACUBE_FINAL_'+galaxy+'_P'+point+'.fits'
    cube = Cube(filename)
    image = white_image(cube)
    psf = select_star(image)
    