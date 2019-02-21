#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 21:38:59 2018

@author: congiu

This is the code to measure the PSF as a function of the wavelength.
"""



import numpy as np
from mpdaf.obj import Cube
import matplotlib.pyplot as plt
import astropy.units as u    
from scipy.interpolate import interp1d
from astropy.convolution import Gaussian2DKernel, convolve
import functions_v1 as fu
import pandas as pd
import sys
import warnings



###############################################################

def return_wave(debug=False): #cambia nome
    d = {}
    with open("wavelength.txt") as f:
        for line in f:
           (key, val) = line.split()
           d[key.strip()] = float(val)
    return d

###############################################################

def extension_locator(line_name, debug = False):
    """
    Converts the name of the extension from string to number
    Input:
    -name of the extension
    output:
    -extension number
    """
    if debug:
        print(line_name, type(line_name))
    data = [line.rstrip('\n') for line in open('hdu_index_short.txt')]
    if debug:
        print(data)
    for i in range(len(data)):
        if data[i].strip() == line_name.upper().strip():  
            return i

###############################################################

def select_star(cube,p,q,dp,dq):
    """
    This function extract a cube with height 2*dp and width 2*dq
    around the coordinates (p,q).
    In MUSE datacubes p = y and q = x.
    """
    
    star = cube[:,p-dp:p+dp,q-dq:q+dq]
    return star

    
###############################################################

def analysis(star_cube, wave_range, function = 'moff', circular = False, 
             n_0 = 2., plot = False):
    """
    the function divides the datacube in single images summing the lambda between wave_range wavelengths.
    Input:
    -star_cube: the datacube centered on the star
    -wave_range: the number of wavelengths to be summed per each image
    -function: it is possible to choose between 'gauss' (2D gaussian) or 'moff' (2D Moffat)
    -circular: if True the fitting function will be circular. Else is elliptical
    -plot: if True it plots the image, the fit and the relative residuals of the fit
    Output:
    -fits: is a dictionary with all the information on the fit of each image.
           -- x0 an y0 are in degrees
           -- FWHM are in arcsec
           -- rot is in degrees and it should be measured from North westward
    """

    


    start, end = fu.wavelength_range(star_cube) #measuring the initial and final wavelength of the datacube
    #setting the list to contain the results of the fits
    lam = []
    y0 = []
    err_y0 = []
    x0 = []
    err_x0 = []
    I0 = []
    err_I0 = []
    fwhm_x = []
    err_fwhm_x = []
    fwhm_y = []
    err_fwhm_y = []    
    rot = []
    err_rot = []
    chi2 = []
    if function == 'moff':
        n = []
        err_n = []
    

        
    delta = 0
    #I'm fitting a first image summing 100 angstrom to have always a reference at the beginning of the wavelength range
    # This is needed because Hb is quite close to the beginning of the range
    if wave_range > 100:
        delta = 100
        lam.append(np.mean([start,start+delta]))  
        ima = star_cube.get_image((start,start+delta))
        x, y = np.indices(ima.data.shape)
        mask = (x-15)**2+(y-15)**2 > 8 **2
        ima.data.mask[mask] = True
    
        fit, fitim, chi2_test = fu.fit_function(ima,function = function, 
                                                circular = circular, n = n_0, 
                                                verbose = True) 
        chi2.append(chi2_test)
        
        if plot:
            plots_2d(ima.data, fitim.data)   #plotting the 2D images
            radial_plot(ima.data, fitim.data) # plotting radial profiles
#            ima.data.mask[mask] = False
        #converting the center in pixel coordinates
        x, y = ima.wcs.wcs.wcs_world2pix(fit.center[1],fit.center[0],0)
        x0.append(x)
        y0.append(y)
        err_y0.append(fit.err_center[0])
        err_x0.append(fit.err_center[1])
        I0.append(fit.peak)
        err_I0.append(fit.err_peak)
        fwhm_x.append(fit.fwhm[1])                 
        err_fwhm_x.append(fit.err_fwhm[1])
        fwhm_y.append(fit.fwhm[0])
        err_fwhm_y.append(fit.err_fwhm[0])    
        rot.append(fit.rot)
        err_rot.append(fit.err_rot)
        if function == 'moff':
            n.append(fit.n)
            err_n.append(fit.err_n)


    #for the fits I'm looping between the initial and final wavelength of the datacube with a
    # wave_range step. Probably I'm going to change it and to divide the interval in a certain number o fixed steps.
    for wave in np.arange(start+delta,end, wave_range):
        # I'm saving the intermediate wavelength of each image
        lam.append(np.mean([wave, wave+wave_range]))
        # from the datacube I'm producing one image for each wavelength interval
        ima = star_cube.get_image((wave,wave+wave_range),)
        x, y = np.indices(ima.data.shape)
        mask = (x-15)**2+(y-15)**2 > 8 **2   
        fit, fitim, chi2_test = fu.fit_function(ima, function = function, 
                                                circular = circular, n = n_0) 
        chi2.append(chi2_test) # saving the reduced chi2
        if plot:
#            ima.data.mask[mask] = False
            plots_2d(ima.data, fitim.data)
            radial_plot(ima.data, fitim.data)

        #I'm saving the results of the fit in the previously prepared lists
        #I'm leaving everything in arcsec, except the center is in pixel
        x, y = ima.wcs.wcs.wcs_world2pix(fit.center[1],fit.center[0],0)
        x0.append(x)
        y0.append(y)
        err_y0.append(fit.err_center[0])
        err_x0.append(fit.err_center[1])
        I0.append(fit.peak)
        err_I0.append(fit.err_peak)
        fwhm_x.append(fit.fwhm[1])                 
        err_fwhm_x.append(fit.err_fwhm[1])
        fwhm_y.append(fit.fwhm[0])
        err_fwhm_y.append(fit.err_fwhm[0])    
        rot.append(fit.rot)
        err_rot.append(fit.err_rot)
        if function == 'moff':
            n.append(fit.n)
            err_n.append(fit.err_n)
        
    #saving everything in a dictionary for simplicity
    measurements = {} 
    measurements['chi2'] = np.array(chi2)
    measurements['wavelength'] = np.array(lam)
    measurements['x0'] = np.array(x0)
    measurements['err_x0'] = np.array(err_x0)
    measurements['y0'] = np.array(y0)
    measurements['err_y0'] = np.array(err_y0)
    measurements['I0'] = np.array(I0)
    measurements['err_I0'] = np.array(err_I0)
    measurements['fwhm_x'] = np.array(fwhm_x)
    measurements['err_fwhm_x'] = np.array(err_fwhm_x)
    measurements['fwhm_y'] = np.array(fwhm_y)
    measurements['err_fwhm_y'] = np.array(err_fwhm_y)
    measurements['rot'] = np.array(rot)
    measurements['err_rot'] = np.array(err_rot)
    if function == 'moff':
        measurements['n'] = np.array(n)
        measurements['err_n'] = np.array(err_n)
    return measurements    


###############################################################
    
def make_plots(fits, save = False, outname = './'):   

    #plot peak intensity
    fig,ax = plt.subplots(1,1)
    ax.errorbar(fits['wavelength'], fits['I0'], yerr = fits['err_I0'], marker = 'o', ls ='', capsize = 3)  
    ax.set_title('peak intensity')
    ax.set_xlabel('Wavelength ($\\AA$)')
    ax.set_ylabel('Flux (10$^{-20}$ erg cm$^{-2}$ s$^{-1}$)')
    if save:
        plt.savefig(outname+'I0_plot.png')
        plt.close()
    else:   
        plt.show()

    #plot center
    fig,ax = plt.subplots(1,1)
    x0 = fits['x0']
    y0 = fits['y0'] 

    # measuring the shift in pixel with respect to the median and converting  
    # in arcsec       
    x0 = (x0 - x0[1])*0.2                 
    y0 = (y0 - y0[1])*0.2
    ax.errorbar(fits['wavelength'], x0, fits['err_x0']*3600, marker = 'o', 
                ls ='', capsize = 3, label = 'x0') 
    ax.errorbar(fits['wavelength'], y0, fits['err_y0']*3600, marker = 'o', 
                ls ='', capsize = 3, label = 'y0') 
    ax.set_title('Center')
    ax.set_xlabel('Wavelength ($\\AA$)')
    ax.set_ylabel('$\\Delta$ x, $\\Delta$ y (arcsec) ')
    ax.legend(loc='best')
    if save:
        plt.tight_layout()
        plt.savefig(outname+'center_plot.png')
        plt.close()
    else:   
        plt.show()
        
    #polar coordinates
    r = np.sqrt(x0**2+y0**2)
    phi = np.arctan2(y0,x0)*180/np.pi  #measuring the angle and converting in deg
    phi = phi%360.
    fig,ax = plt.subplots(1,1)
    ax.scatter(fits['wavelength'], r, marker = 'o') 
    ax.set_title('Polar shift: R')
    ax.set_xlabel('Wavelength ($\\AA$)')
    ax.set_ylabel('R (arcsec)')
    if save:
        plt.tight_layout()
        plt.savefig(outname+'polar_r_plot.png')
        plt.close()
    else:   
        plt.show()
        
    fig,ax = plt.subplots(1,1)
    ax.scatter(fits['wavelength'], phi, marker = 'o',) 
    ax.set_title('Polar shift: Phi')
    ax.set_xlabel('Wavelength ($\\AA$)')
    ax.set_ylabel('Phi (deg)')
    if save:
        plt.tight_layout()
        plt.savefig(outname+'polar_phi_plot.png')
        plt.close()
    else:   
        plt.show()

    #plot FWHM
    fig,ax = plt.subplots(1,1)
    ax.errorbar(fits['wavelength'], fits['fwhm_x'], 0, marker = 'o', ls ='', 
                capsize = 3, label = 'fwhm_x') 
    ax.errorbar(fits['wavelength'], fits['fwhm_y'], fits['err_fwhm_y'], marker = 'o', ls ='', 
                capsize = 3, label = 'fwhm_y') 
    ax.set_title('FWHM')
    ax.set_xlabel('Wavelength ($\\AA$)')
    ax.set_ylabel('FWHM (arcsec)')
    ax.legend(loc='best')
    if save:
        plt.tight_layout()
        plt.savefig(outname+'fwhm_plot.png')
        plt.close()
    else:   
        plt.show()

    #plot PA
    fig,ax = plt.subplots(1,1)
    ax.errorbar(fits['wavelength'], fits['rot'], yerr = fits['err_rot'], 
                marker = 'o', ls ='', capsize = 3)  
    ax.set_title('PA')
    ax.set_xlabel('Wavelength ($\\AA$)')
    ax.set_ylabel('PA (deg)')
    if save:
        plt.tight_layout()
        plt.savefig(outname+'rot_plot.png')
        plt.close()
    else:   
        plt.show()

    #plot chi2
    fig,ax = plt.subplots(1,1)
    ax.errorbar(fits['wavelength'], fits['chi2'], yerr = 0, marker = 'o', 
                ls ='', capsize = 3)  
    ax.set_title('Reduced Chi2')
    ax.set_xlabel('Wavelength ($\\AA$)')
    ax.set_ylabel('Reduced Chi2')
    if save:
        plt.tight_layout()
        plt.savefig(outname+'chi2_plot.png')
        plt.close()
    else:   
        plt.show()
        
    #plot ellipticity
    
    err = 1/fits['fwhm_y']*fits['err_fwhm_y']
    fig,ax = plt.subplots(1,1)
    ax.errorbar(fits['wavelength'], fits['fwhm_x']/fits['fwhm_y'], yerr = 0, 
                marker = 'o', ls ='', capsize = 3)  
    ax.set_title('Ellipticity')
    ax.set_xlabel('Wavelength ($\\AA$)')
    ax.set_ylabel('Ellipticity')
    if save:
        plt.tight_layout()
        plt.savefig(outname+'ellipticity_plot.png')
        plt.close()
    else:   
        plt.show()
        
    if 'n' in fits.keys():
        # n parameter    
        fig,ax = plt.subplots(1,1)
        ax.errorbar(fits['wavelength'], fits['n'], yerr = fits['err_n'], 
                    marker = 'o', ls ='', capsize = 3)  
        ax.set_title('n')
        ax.set_xlabel('Wavelength ($\\AA$)')
        ax.set_ylabel('n parameter')
        if save:
            plt.tight_layout()
            plt.savefig(outname+'n_plot.png')
            plt.close()
        else:   
            plt.show()


###############################################################

def plots_2d(ima, fitim, save= False):
    residuals = (ima - fitim)/ima
    print('max: {:1.4f}, min: {:1.4f}, median: {:1.4f}' .format(np.max(residuals), np.min(residuals), np.median(residuals)))
    # plotting the original image, the fit and the residuals
    fig, ax = plt.subplots(1,3, figsize=(15,4))
    im0 = ax[0].imshow(ima, vmin = np.min(ima), vmax = np.max(ima), 
            origin = 'lower')
    ax[0].set_title('data')
    fig.colorbar(im0, ax = ax[0])
    im1 = ax[1].imshow(fitim, vmin = np.min(ima), vmax = np.max(ima), 
            origin = 'lower')
    ax[1].set_title('fit')
    fig.colorbar(im1, ax = ax[1])
    im2 = ax[2].imshow(residuals, vmin = -0.2, vmax = 0.2, origin = 'lower')
    ax[2].set_title('relative residuals')
    fig.colorbar(im2, ax = ax[2])
    if save:
        plt.savefig('2d_PSF_plots.png')
        plt.close()
    else:
        plt.show()

###############################################################

def radial_plot(ima, fit, save= False):
    max_index = np.unravel_index(ima.argmax(),np.shape(ima))
#    radial_ima = np.sum(ima, axis = 0)
#    radial_fit = np.sum(fit, axis = 0)
    print(max_index)
    radial_ima_x = ima[max_index[0],:]
    radial_fit_x = fit[max_index[0],:]
    radial_ima_y = ima[:,max_index[0]]
    radial_fit_y = fit[:,max_index[0]]
    pixel = np.arange(len(radial_ima_x))
    fig = plt.figure()
    frame1 = fig.add_axes((0.1,0.3,0.8,0.6))
    frame1.scatter(pixel, radial_ima_x, c= 'b', label = 'x axis')
    frame1.plot(pixel, radial_fit_x, c='b')
    frame1.scatter(pixel, radial_ima_y, c= 'r', label = 'y axis')
    frame1.plot(pixel, radial_fit_y, c='r')
    frame1.set_xticklabels([])
    frame1.set_ylabel('flux')
    frame1.legend(loc = 'best')
    frame2 = fig.add_axes((0.1,0.1,0.8,0.2))
    residual_x = (radial_ima_x-radial_fit_x)/radial_fit_x
    residual_y = (radial_ima_y-radial_fit_y)/radial_fit_y
    # setting the range of the y axis looking for the max (in absolute value)
    # of the residual, and incrasing it of a 50 %
    lim = np.max([np.max(residual_x), np.abs(np.min(residual_x)),\
                  np.max(residual_y), np.abs(np.min(residual_y))])
    lim = lim+0.5*lim
    frame2.set_ylim(lim, -lim)
    frame2.scatter(pixel, residual_x, c = 'b')
    frame2.scatter(pixel, residual_y, c = 'r')
    frame2.plot(pixel, 0*pixel, ls = '--', c= 'k')
    frame2.set_xlabel('pixels')
    frame2.set_ylabel('rel. res.')
    if save:
        plt.savefig('radial_PSF_plots.png')
        plt.close()
    else:
        plt.show()


###############################################################

def measure_sigma(fits):
    
    line_wave = return_wave()
    
    #Converting FWHM to STD
    sigma_x = fits['fwhm_x']*gaussian_fwhm_to_sigma
    sigma_y = fits['fwhm_y']*gaussian_fwhm_to_sigma

    #interpolating the FWHM to have an extimate of the FWHM at the line position
    interp_x = interp1d(fits['wavelength'],sigma_x)
    interp_y = interp1d(fits['wavelength'],sigma_y)

    #I have to bring all the FWHM to the largest values, so I'll loo for them
    x_max = np.max(sigma_x)
    y_max = np.max(sigma_y)
    print('max FWHM x: {}, max FWHM y: {}'.format(x_max, y_max))
    new_x, new_y = [], []

    for line in line_wave.values():
        fwhm_x_l = interp_x(line)
        fwhm_y_l = interp_y(line)
    
        new_x.append(np.sqrt(x_max**2 - fwhm_x_l**2))
        new_y.append(np.sqrt(y_max**2 - fwhm_y_l**2))
    
    data_for_conv = {'line_wave' :line_wave, 'sigma_x' : new_x , 
                     'sigma_y' : new_y, 'PA': fits['rot']}

    return data_for_conv

###############################################################
   
def convolution(hdu, fits, output):
    
    parameters = measure_sigma(fits)
    
    for (i,line) in enumerate(parameters['line_wave']):
        kernel = Gaussian2DKernel(x_stddev=parameters['sigma_x'][i], \
                                  y_stddev=parameters['sigma_y'][i], \
                                  theta = parameters['PA'][i]*np.pi/180)
        print('FWHM x: {:2.4f}, FWHM y: {:2.4f}, PA: {:3.4f}, line: {}' \
             .format(parameters['sigma_x'][i], parameters['sigma_y'][i], 
                     parameters['PA'][i], line))
        index = extension_locator(line) 
        image = convolve(hdu[index].data[0,:,:], kernel)
        err = convolve(hdu[index+1].data[0,:,:], kernel)
        hdu[index].data = image
        hdu[index].header['X_CONV'] = '{:2.4f}'.format(parameters['sigma_x'][i])
        hdu[index].header['Y_CONV'] = '{:2.4f}'.format(parameters['sigma_y'][i])
        hdu[index].header['PA_CONV'] = '{:2.4f}'.format(parameters['PA'][i])
        hdu[index+1].data = err
        hdu[index+1].header['X_CONV'] = '{:2.4f}'.format(parameters['sigma_x'][i])
        hdu[index+1].header['Y_CONV'] = '{:2.4f}'.format(parameters['sigma_y'][i])
        hdu[index+1].header['PA_CONV'] = '{:2.4f}'.format(parameters['PA'][i])

    hdu.writeto(output, overwrite = True)
    return hdu
    
###############################################################
    
def print_result(fits, output):
    
    data = pd.DataFrame.from_dict(fits)
    data.to_csv(output, sep = ' ', index = False)
    

if __name__ == '__main__':

    warnings.simplefilter('ignore')
    galaxy = 'NGC1672'
    point   = '02'
    x = 258
    y = 285
    
    filename = '../Cubes/DATACUBE_FINAL_'+galaxy+'_P'+point+'.fits'
    cube = Cube(filename)
    outname = './plots/'+galaxy+'_P'+point+'_x'+str(x)+'_y'+str(y)+'_'
    star1 = select_star(cube,y,x,15,15)
    
#    fits = analysis(star1, 300, function ='moff', circular = False, 
#                    plot = True)
#    make_plots(fits, save = True, outname = outname)
#    output = './plots/Moffat_'+galaxy+'_P'+point+'_x'+str(x)+'_y'+str(y)+'.txt'
#    print_result(fits,output)
    
    #circular fit
    outname = './plots/'+galaxy+'_P'+point+'_x'+str(x)+'_y'+str(y)+'_circ_'
    star1 = select_star(cube,y,x,15,15)
    fits = analysis(star1, 300, function ='moff', circular = True, 
                    plot = False, n_0 = 2.0)
    make_plots(fits, save = False, outname = outname)
#    output = './plots/Moffat_'+galaxy+'_P'+point+'_x'+str(x)+'_y'+str(y)+'circ.txt'
#    print_result(fits,output)
    
    n_mean = np.average(fits['n'], weights = 1/fits['err_n'])
    outname = './plots/'+galaxy+'_P'+point+'_x'+str(x)+'_y'+str(y)+'_nmean_'
    fits = analysis(star1, 300, function ='moff', circular = True, 
                    plot = True, n_0 = n_mean)
    make_plots(fits, save = False, outname = outname)
#    output = './plots/Moffat_'+galaxy+'_P'+point+'_x'+str(x)+'_y'+str(y)+'nmean.txt'
#    print_result(fits,output)
    
    print(galaxy+'_P'+point+'_x'+str(x)+'_y'+str(y), 'n: {:0.3f}' 
          .format(n_mean))

    



