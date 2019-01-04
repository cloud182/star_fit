#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 21:38:59 2018

@author: congiu
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import functions_v1 as fu
import astropy.visualization as vis

from astropy.io import ascii 
from astropy.coordinates import SkyCoord,FK5
from scipy.interpolate import interp1d
from astropy.nddata.utils import Cutout2D
from astropy.stats import sigma_clip
from mpdaf.obj import Image, WCS
from mpdaf.obj import gauss_image, moffat_image

#from astropy.stats import gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma
import sys
import time




########################################################################
  
def circular(model):
    """
    This is a function to define a circular gaussian
    """
    fwhm_y = model.fwhm_x
    return fwhm_y

########################################################################
    
def plots_2d(f1, f2, filename=None, save= False):
    """
    Produces a plot of 2 2D images and of their ratio.
    Input:
    - f1: first image (usually the data)
    - f2: second image (usually the fit)
    - filename: name of the eventual output plot
    - save: if True the plot is saved
    """
    
    residuals = f1/f2
    print('max: {:1.4f}, min: {:1.4f}, median: {:1.4f}' 
          .format(np.max(residuals), np.min(residuals), np.median(residuals)))
    # plotting the original image, the fit and the residuals
    fig, ax = plt.subplots(1,3, figsize=(15,4))
    im0 = ax[0].imshow(f1, vmin = np.min(f1), vmax = np.max(f1),
                       origin = 'lower')
    ax[0].set_title('Data')
    fig.colorbar(im0, ax = ax[0])
    im1 = ax[1].imshow(f2, vmin = np.min(f1), vmax = np.max(f1), 
                       origin = 'lower')
    ax[1].set_title('Fit')
    fig.colorbar(im1, ax = ax[1])
    im2 = ax[2].imshow(residuals, vmin = 0., 
                       vmax = 1.5, origin = 'lower')
    ax[2].set_title('Ratio')
    fig.colorbar(im2, ax = ax[2])
    if save:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

########################################################################
        
def fwhm_distribution(parameters, psf_file, wave, filename = None,
                      save = False, clip = False):
    """
    Produces an istogram of the FWHM and compares it to the values obtained 
    for the selected star.
    Input:
    - parameters: file with the results of the fit
    - psf_file: file with the stellar PSF as a function of wavelength
    - wave: wavelength of the line map considered
    - filename: name of the eventual output plot
    - save: if True the plot is saved
    - clip: if True apply a 3sigma clipping on the FWHM
    Output:
    - histogram of the FWHM
    """
    
    #reading the file with the PSF information
    data = ascii.read(psf_file)
    #interpolating the PSF FWHM to obtain the predicted value at the linemap
    #wavelength
    fwhm_interp_x = interp1d(data['wavelength'], data['fwhm_x'])
    fwhm_interp_y = interp1d(data['wavelength'], data['fwhm_y'])
    fwhm_line_x = fwhm_interp_x(wave)
    fwhm_line_y = fwhm_interp_y(wave)
    
    #loading the parameters of the fit
    fwhm_x = parameters['fwhm_x']
    fwhm_y = parameters['fwhm_y']
    #removing the values that are too large, hopefully it means that the fit 
    #did not work. Maybe I can do something with chi2.
    #I think this s better if I have to save the final results in a file
    #while the clipping I can do next is good for the plot.
    idx = [i for i, arr in enumerate(parameters['chi2']) if arr > 3 or arr<0.50]
    fwhm_x = np.delete(fwhm_x, idx)
    fwhm_y = np.delete(fwhm_y, idx)
    
    #checking if there are Nan
    idx = [i for i, arr in enumerate(fwhm_x) if np.isnan(arr) or np.isinf(arr)]
    idy = [i for i, arr in enumerate(fwhm_y) if np.isnan(arr) or np.isinf(arr)]
    fwhm_x = np.delete(fwhm_x, idx)
    fwhm_y = np.delete(fwhm_y, idy)    
    #cleaning the measured PSF removing the outliers with a sigma clipping
    #method
    if clip:
        fwhm_x = sigma_clip(fwhm_x, 3, masked = False)
        fwhm_y = sigma_clip(fwhm_y, 3, masked = False)
        print(len(fwhm_y))      #just to check how many regions are there

    #making the histogram. The histtype='step' allows the histogram to be 
    #produced almost istantaneously
    
#    print(fwhm_y)
    max_hist = np.min([10,np.max(fwhm_y)])
    bins = np.arange(np.min(fwhm_y),max_hist,0.05)
    plt.hist([fwhm_x,fwhm_y], bins=bins,label=['FWHM_x', 'FWHM_y'],
             histtype='step')
    plt.axvline(fwhm_line_x, c = 'red', label = 'Star_x', ls = '--')
    plt.axvline(fwhm_line_y, c = 'black', label = 'Star_y', ls = ':')
    plt.legend(loc = 'best')
    if save:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
    
    #printing the results of the plot
    print('data_median x: {:1.3f}, data_mean x: {:1.3f}, sigma x: {:1.3f} '
          .format(np.median(fwhm_x), np.mean(fwhm_x), np.std(fwhm_x))
          +'PSF_fwhm x: {:1.3f}' .format(fwhm_line_x))
    print('data_median y: {:1.3f}, data_mean y: {:1.3f}, sigma y: {:1.3f} '
          .format(np.median(fwhm_y), np.mean(fwhm_y), np.std(fwhm_y))
          +'PSF_fwhm y: {:1.3f}' .format(fwhm_line_y))
    result = {'fwhm_star': fwhm_line_y, 'min': np.min(fwhm_y), 
              'median': np.median(fwhm_y), 'mean' : np.mean(fwhm_y), 
              'std': np.std(fwhm_y)}
    return result

########################################################################

def region_selection(regions, ra, dec, size, radius):
    """
    This plot performs a selection of the regions in the catalogue, selecting
    only the ones without a nearby region in a radius specified by the user.
    Input:
    - regions: array with the number of the regions
    - ra: array with the RA of the regions
    - dec: array with the DEC of the regions
    - size: array with the size of the region
    - radius: this is a parameter used to define the minimum distance 
              between two regions in order for the examined one to be isolated.
              The distance is defined as:
                  limit = np.sqrt(2)*radius*size
                  where:
                  - size is the size of the region in arcsec
                  - np.sqrt(2) is just because I like it.
    output:
    - new_reg: array with the region number only of the selected regions
    - new_ra: array with the RA only of the selected regions
    - new_dec: array with the DEC only of the selected regions
    - new_size: array with the size only of the selected regions
    """
    #defining the coordinates for all the regions
    all_coords = SkyCoord(ra,dec, frame = FK5, unit=(u.deg,u.deg))
    #defining the list containing the selected region
    new_reg = []
    new_ra = []
    new_dec = []
    new_size = []
    #looping on the regions
    for (region, coords, size_r) in zip(regions, all_coords, size):
        #measuring the distance between the selected region and all the 
        #other and checking how many regions are within a defined distance
        limit = np.sqrt(2)*radius*size_r*u.arcsec #maybe there is a constant already for sqrt(2)
        distance = coords.separation(all_coords)    
        close = [dist for dist in distance if dist.arcsec*u.arcsec < limit]
        #selecting only the isolated regions
        #len = 1 the region itself is always present in close.
        if len(close) == 1:
            new_reg.append(region)
            new_ra.append(coords.ra.degree)
            new_dec.append(coords.dec.degree)
            new_size.append(size_r)
            
    #returning the new arrays
    return np.array(new_reg), np.array(new_ra), np.array(new_dec), \
           np.array(new_size)
           
########################################################################
           
def HII_fit(region, ra_r, dec_r, size_r, image, mask = False,
            function = 'gauss', plot = False):
        """
        This code performa the fit of the single H II region.
        Input:
        - region, ra_r, dec_r, size_r: information on the single region
        - image: an MPDAF Image object built from the emission line map
        - mask: if True it applies a mask on the image, based on the distance 
                of the pixels from the center of the image 
        - function: it selects the type of function to use for the fit
                - gauss: 2D Gaussian
                - moff: 2D Moffat
        - plot: if True the plot of the region, the fit and the ratio are 
                produced.
        Output:
        - FWHM_x: in arcsec
        - FWHM_y: in arcsec
        - theta: PA in degrees
        - ellipt: ellipticity of the region
        """
        #defining the coordinates of the region and selecting a cutout
        coords = SkyCoord(ra_r,dec_r, frame = FK5, unit=(u.deg,u.deg))
        region_cut = image.subimage(center = (coords.dec.deg, coords.ra.deg),
                                    size = 3*size_r)
        
        #converting the coordinates from FK5 to pixels
        #creating a mask, putting at zero all the pixel outide 1.5*size_r 
        # in pixels
        if mask:
#            I don't understand why it is not working
#            y0, x0= image.wcs.sky2pix((coords.dec.deg,coords.ra.deg))
            y0, x0 = region_cut.data.shape
            X, Y = np.indices(region_cut.data.shape)
            mask_array = (X-x0/2)**2+(Y-y0/2)**2 > (1.41*size_r/0.2)**2
            region_cut.data.mask[mask_array] = True
            
        
        # defining the model to be used during the fit and eventual boundaries
        # and other details on the parameter for the fit and performing the fit
        # Maybe I should put them as inputs of the function

        fit, fit_ima, chi2_test = fit_function(region_cut, 
                                               function = function,
                                               circular = False,
                                               verbose = False)


            
        if plot:
            print('fwhm_x: {:1.3f}, fwhm_y: {:1.3f}, theta: {:1.3f}, \
                  chi2: {:1.3f}' 
                  .format(fit.fwhm[1], fit.fwhm[0], fit.rot,chi2_test))
            plots_2d(region_cut.data, fit_ima.data)

        if mask:
            #resetting the mask
            region_cut.data.mask[mask_array] = False
            
            #print(fit.theta[0])
        #measuring the ellipticity
        return fit, chi2_test

########################################################################

def select_cutout(image, wcs):
    """
    This function allows to select a cutout of the line map, choosing it 
    interactivly from a plot.
    Input:
    - image: the line image
    - wcs: the wcs of the image
    Output:
    - cutout: the desired cutout.
    """
    #I'm looking for how many pointings are in the mosaic 
    #I don't know if it is always accurate
    print(image.shape)
    nrow = image.shape[0]//300. 
    ncol = image.shape[1]//300.
    if image.shape[0]%300 > 200:
        nrow += 1
    if image.shape[1]%300 > 200:
        ncol += 1
    #measuring the exact width of a row and of a column
    drow = image.shape[0]/nrow
    dcol = image.shape[1]/ncol
    
    #I'm showing the image to select the correct section
 
    fig, ax = plt.subplots(1,1)
    interval = vis.PercentileInterval(99.9)
    vmin,vmax = interval.get_limits(image)
    norm = vis.ImageNormalize(vmin=vmin, vmax=vmax, 
                              stretch=vis.LogStretch(1000))
    ax.imshow(image, cmap =plt.cm.Greys, norm = norm, origin = 'lower')
    # plotting lines that can help identifying the different pointings
    for x in np.arange(0,image.shape[1],dcol):
        ax.axvline(x)
    for y in np.arange(0,image.shape[0],drow):
        ax.axhline(y)
        
        
    #this function allows to select the cutout from a mouse click
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
    cutout = Cutout2D(image, (x, y), size = (image.shape[0]/nrow -20)* u.pixel, 
                      wcs = wcs)
    return cutout, x, y

########################################################################

def region_in_point(regions, ra, dec, size, pointing):
    """
    This function allows to select only the region inside a pointing.
    Input:
    - region, ra_r, dec_r, size_r: information on the single region
    - pointing: cutout of the pointing region
    Output:
    - new_reg: array with the region number only of the selected regions
    - new_ra: array with the RA only of the selected regions
    - new_dec: array with the DEC only of the selected regions
    - new_size: array with the size only of the selected regions
    """    
    #defining the new list which will contain the output
    new_reg = []
    new_ra = []
    new_dec = []
    new_size = []
    for (region, ra_r, dec_r, size_r) in zip(regions, ra, dec, size):
        #defining the coordinate of the region and checking if it is inside the
        #selected region.
        coords = SkyCoord(ra_r, dec_r, frame = FK5, unit=(u.deg,u.deg))
        if pointing.wcs.footprint_contains(coords):
             new_reg.append(region)
             new_ra.append(ra_r)
             new_dec.append(dec_r)
             new_size.append(size_r) 
    #converting the lists in arrays before returning them
    return np.array(new_reg), np.array(new_ra), np.array(new_dec), \
           np.array(new_size) 
    
########################################################################
 
def fit_function(ima, function='moff', circular = False, 
                 verbose = False):
    """
    This is the function to perform the fits. It is possible to choose between 
    'gauss' (Gaussian) or 'moff' (Moffat) as fitting functions.
    The fit is performed with MPDAF built in functions.
    Input:
    - ima: the image over which the fit must be performed.
    - center: coordinates of the object to be fit
    - circular: if true the function is circular, otherwise it is elliptical
    - function: function to use for the fit. 
                -- 'gauss': 2D Gaussian
                -- 'moff': 2D Moffat
    - verbose: print the results of the fit
    Output:
    - fit: the object containing all the info about the fit
    - fitim: the image of the fit
    - chi2_test: the reduced chi2 value for the fit.
    """
    # I'm setting everything up so I can choose to fit with a gaussian or with 
    # a moffat I'm doing the fit with MPDAF built in fitting functions
    if function == 'gauss':
        fit = ima.gauss_fit(verbose = verbose,fit_back = True,
                            flux = np.max(ima.data), peak = True, 
                            circular = circular, maxiter = 5000)
        fitim = gauss_image(wcs=ima.wcs, gauss=fit)
        if circular:
            chi2_test = np.sum((ima.data-fitim.data)**2/fitim.data)\
            /(ima.data.count()-5)
        else:
            chi2_test = np.sum((ima.data-fitim.data)**2/fitim.data)\
            /(ima.data.count()-7)
    elif function == 'moff':
        fit = ima.moffat_fit(verbose = verbose, fit_back = True, 
                             flux = np.max(ima.data), peak = True, 
                             circular = circular, maxiter = 5000)
        fitim = moffat_image(wcs=ima.wcs, moffat=fit)
        if circular:
            chi2_test = np.sum((ima.data-fitim.data)**2/fitim.data)\
            /(ima.data.count()-6)
        else:
            chi2_test = np.sum((ima.data-fitim.data)**2/fitim.data)\
            /(ima.data.count()-8)
    else:
        print('Function not supported')
        sys.exit()
    return fit, fitim, chi2_test  
    
########################################################################

def measure_PSF(cube, catalogue, line, x0 = None, y0 = None, 
                function = 'moff',  plot = False, mask = False):
    
    hdu = fu.open_hdu(cube)
    image, header, wcs, err = fu.open_image(hdu, line.upper(), err = True)
#    image = fu.create_masked(image, err, 3)
    regions, ra, dec, size = fu.read_table(catalogue)  
    if x0 == None:
        pointing_cut, x0, y0 = select_cutout(image, wcs)
        regions, ra, dec, size = region_in_point(regions, ra, dec, size, 
                                             pointing_cut)
    
    new_wcs = WCS(hdr=header)
    new_image = Image(data = image, var = err**2, wcs = new_wcs)
    fwhm_x = []
    fwhm_y = []
    ellipt = []
    theta = []
    chi2 = []
    for (region, ra_r, dec_r, size_r) in zip(regions, ra, dec, size):
        fit_result, fit_chi2 = HII_fit(region, ra_r, dec_r, size_r, new_image,
                             function = function, plot = plot, mask = mask)
        fwhm_x.append(fit_result.fwhm[1])
        fwhm_y.append(fit_result.fwhm[0])
        theta.append(fit_result.rot)
        ellipt.append(fit_result.fwhm[1]/fit_result.fwhm[0])
        chi2.append(fit_chi2)
        
    
    parameters = {'region': regions, 'ra':ra, 'dec': dec, 
                  'fwhm_x': np.array(fwhm_x), 'fwhm_y':np.array(fwhm_y),
                  'e':np.array(ellipt), 'theta': np.array(theta), 
                  'chi2': np.array(chi2)}
    return parameters, x0, y0
    
if __name__ == '__main__':
    
    start = time.time()
    
    #defining input quantities
    cube = '../Mosaics/DATACUBE_FINAL_NGC1087_1_comp_mosaic.fits'
    table = '../catalogues_and_masks/ngc1087_hiicat_v1.fits'
    lines = ['HALPHA']
    wavelength = [6562]

    galaxy = 'NGC1087'
    point = '06'
    x = '035'
    y = '249'
    psf_name = './plots/Moffat_'+galaxy+'_P'+point+'_x'+x+'_y'+y+'.txt'

    #output file
    out_name = './plots/hii_fit_'+galaxy+'_P'+point+'_x'+x+'_y'+y+'.txt'
    f = open(out_name, 'w')    
    print('line','wave','fwhm_star', 'median', 'mean', 'std', file = f)
    x0 = None
    y0 = None
    #performing the fit for each line map
    for (line,wave) in zip(lines,wavelength):
        print('line: '+line)
        #fitting the PSF
        parameters, x0, y0 = measure_PSF(cube, table, line, function = 'moff',  
                                         plot = False, x0 = x0, y0 = y0, 
                                         mask = True) 
        #plotting the histogram
        result = fwhm_distribution(parameters, psf_name, wave, clip = True)
        print(line, wave, result['fwhm_star'], result['min'], result['median'], 
              result['mean'], result['std'], file = f)
        
    print('Completed in {:1.2f} seconds' .format(time.time()-start))
        
    
    