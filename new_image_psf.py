#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 21:38:59 2018

@author: congiu

This is the code to fit automatically all the HII in a pointing to try
to measure the width of the PSF from them. (not very successful)
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import functions_v1 as fu
import astropy.visualization as vis
import emcee
import seaborn as sns
import pandas as pd

from astropy.io import ascii 
from astropy.coordinates import SkyCoord,FK5
from scipy.interpolate import interp1d
from astropy.nddata.utils import Cutout2D
from astropy.stats import sigma_clip
from mpdaf.obj import Image, WCS
from mpdaf.obj import gauss_image, moffat_image
from scipy.stats import skewnorm, exponnorm


#from astropy.stats import gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma
import sys
import time
import warnings

########################################################################

def lnprob_exp(param, x):
    
    k, location, scale = param[0], param[1], param[2]
    P = exponnorm.logpdf(x, k, location, scale)
    return P

########################################################################

def lnprobtot_exp(param, x):
    """
    Logarithmic likelihood of the entire sample.
    Input:
    - param: Parameters of the PDF (probability distribution function)
    - x: data
    Output:
    - Logarithmic likelihood of the entire sample.
    """
    
    k, location, scale = param[0], param[1], param[2]
    lam = 1/(scale*k)
    if lam < 0 or scale**2 < 0 or k < 0: return -np.inf
    return np.sum(lnprob_exp(param,x))

########################################################################

def lnprob(param, x):
    """
    Logarithmic likelihood per point point.
    Input:
    - param: Parameters of the PDF (probability distribution function)
    - x: data
    Output:
    - P: logarithmic likelihood calculated at each point.
    """
         
    shape, location, scale = param[0], param[1], param[2]
    P = skewnorm.logpdf(x, shape, loc = location, scale = scale)
    return P

########################################################################

def lnprobtot(param, x):
    """
    Logarithmic likelihood of the entire sample.
    Input:
    - param: Parameters of the PDF (probability distribution function)
    - x: data
    Output:
    - Logarithmic likelihood of the entire sample.
    """
    
    shape, location, scale = param[0], param[1], param[2]
    if scale < 0: return -np.inf
    return np.sum(lnprob(param,x))

########################################################################
    
def MLE_fitting(data, sigma = 0.01, niter = 100):
    
    MLE = []
    i = 0
    while i < niter:
        a = np.random.normal(0,sigma,len(data))
        newdata = data+a
        MLE.append(skewnorm.fit(newdata, 1))
        i +=1
    MLE = np.array(MLE)
    newpar = MLE.mean(axis=0)
    newpar_err = MLE.std(axis=0)
    print('\n')
    print('###################### Skewed Norm #######################')
    print('\n')
    print("Built-in fit")
    print("shape of skewed gaussian:  {:0.3f} +- {:0.3f}" 
          .format(newpar[0], newpar_err[0]))
    print("location of skewed gaussian: {:0.3f} +- {:0.3f}" 
          .format(newpar[1], newpar_err[1]))
    print("scale of skewed gaussian: {:0.3f} +- {:0.3f}" 
          .format(newpar[2], newpar_err[2]))
    print("########################################################")
          
    MLE = []
    i = 0
    while i < niter:
        a = np.random.normal(0,sigma,len(data))
        newdata = data+a
        MLE.append(exponnorm.fit(newdata, 1))
        i +=1
    MLE = np.array(MLE)
    exppar = MLE.mean(axis=0)
    exppar_err = MLE.std(axis=0)
    print('\n')
    print('###################### Exp Norm #######################')
    print('\n')
    print("Built-in")
    print("shape of exponential gaussian:  {:0.3f} +- {:0.3f}" 
          .format(exppar[0], exppar_err[0] ))
    print("location of exponential gaussian: {:0.3f} +- {:0.3f}" 
          .format(exppar[1], exppar_err[1]))
    print("scale of exponential gaussian: {:0.3f} +- {:0.3f}" 
          .format(exppar[2], exppar_err[2]))
    
    return newpar, newpar_err, exppar, exppar_err

def MCMC_fitting(data, ndim= 3, nwalkers= 100, nsteps= 1000,
                 bounds = [0,2,2,3,2,4]):
    """
    The function perform an MCMC fitting of a data distribution using 
    a skewed gaussian as a fitting function. To double check the results
    the function returns also the results of the built-in fitting function
    defined in scipy. It should be a maximum likelihood fitting.
    Inputs:
    - data: the data to be fit
    - ndim: number of parameters of the curves (for the skewed gausian it's 3)
    - nwalkers: number of starting points for the fit
    - nsteps: number of steps for each walkers
    - bounds: starting interval for the walkers
    Output:
    - meanpar: parameters obtained with the MCMC fitting algorithm
    - stdpar: errors on the parameters obtained via MCMC
    - newpar: parameters from the built-in fitting function, averaging the 
              results of nsteps fit
    - newstd: error on parameters from the built-in fitting function.
              the errors are obtained performing the fitting nsteps
              times and measuring the std deviation

    """
    
    pos = [[np.random.uniform(low = bounds[0], high = bounds[1]), 
            np.random.uniform(low = bounds[2], high = bounds[3]),
            np.random.uniform(low = bounds[4], high = bounds[5])] 
            for i in range(nwalkers)] 
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobtot, args = [data])
    sampler.run_mcmc(pos, nsteps)
    
    # take second half of each chain, discard "burn-in"
    #The first half seems to not be good for the purpose
    samples = sampler.chain[:, int(nsteps/2):, :].reshape((-1, ndim)) 
    
    #results
    meanpar = np.mean(samples, axis=0)
    stdpar = np.std(samples, axis=0)
    
    print('\n')
    print('###################### Skewed Norm #######################')
    print('\n')

    print("Dedicate fit")
    print("shape of skewed gaussian:  {:0.3f} +- {:0.3f}" 
          .format(meanpar[0], stdpar[0]))
    print("location of skewed gaussian: {:0.3f} +- {:0.3f}" 
          .format(meanpar[1], stdpar[1]))
    print("scale of skewed gaussian: {:0.3f} +- {:0.3f}" 
          .format(meanpar[2], stdpar[2]))
    print("########################################################")
    
          

    pos2 = [[np.random.uniform(low = bounds[0], high = bounds[1]), 
            np.random.uniform(low = bounds[2], high = bounds[3]),
            np.random.uniform(low = bounds[4], high = bounds[5])] 
            for i in range(nwalkers)] 

    sampler2 = emcee.EnsembleSampler(nwalkers, ndim, lnprobtot_exp, 
                                     args = [data])
    sampler2.run_mcmc(pos2, nsteps)
    samples2 = sampler2.chain[:, int(nsteps/2):, :].reshape((-1, ndim)) 

    meanexp = np.mean(samples2, axis=0)
    stdexp = np.std(samples2, axis=0)
    
    print('\n')
    print('###################### Exp Norm #######################')
    print('\n')
    print("Dedicate fit")
    print("shape of skewed gaussian:  {:0.3f} +- {:0.3f}" 
          .format(meanexp[0], stdexp[0]))
    print("location of skewed gaussian: {:0.3f} +- {:0.3f}" 
          .format(meanexp[1], stdexp[1]))
    print("scale of skewed gaussian: {:0.3f} +- {:0.3f}" 
          .format(meanexp[2], stdexp[2]))
    print("########################################################")
    
          

    
    return meanpar, stdpar, meanexp, stdexp  


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
          .format(np.max(residuals), np.min(residuals), np.ma.median(residuals)))
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
        
def fwhm_distribution(parameters, wave = None, filename = None, 
                      psf_file = False, save = False, clip = False):
    """
    Produces an istogram of the FWHM and compares it to the values obtained 
    for the selected star. I'm considering only the FWHM_y that should always be
    the major axis.
    Input:
    - parameters: file with the results of the fit
    - wave: wavelength of the line map considered
    - filename: name of the eventual output plot
    - psf_file: file with the stellar PSF as a function of wavelength.
                If False the star won't be considered
    - save: if True the plot is saved
    - clip: if True apply a 3sigma clipping on the FWHM
    Output:
    - histogram of the FWHM
    """
    if psf_file:
        
        #reading the file with the PSF information
        data = ascii.read(psf_file)
        #interpolating the PSF FWHM to obtain the predicted value at the linemap
        #wavelength
        fwhm_interp_y = interp1d(data['wavelength'], data['fwhm_y'])
        fwhm_line_y = fwhm_interp_y(wave)
    
    #loading the parameters of the fit
    fwhm_y = parameters['fwhm_y']

    #selecting results with reasonable chi2
    idy = [i for i, arr in enumerate(parameters['chi2']) if arr > 30 or arr<0.50]
    fwhm_y = np.delete(fwhm_y, idy)
    
    #checking if there are Nan and removing the obviously wrong fits
    #(too large or too small)
    idy = [i for i, arr in enumerate(fwhm_y) if np.isnan(arr) or arr>10]
    fwhm_y = np.delete(fwhm_y, idy)
    idy = [i for i, arr in enumerate(fwhm_y) if  arr<0.3]
    fwhm_y = np.delete(fwhm_y, idy)    
    
    
    #cleaning the measured PSF removing the outliers with a sigma clipping
    #method
    if clip:
        fwhm_y = sigma_clip(fwhm_y, 3, masked = False)
        print(len(fwhm_y))      #just to check how many regions are there

    #making the histogram. I'm using seaborn to plot it, it's easier
    
    meanpar, stdpar, meanexp, stdexp = MCMC_fitting(fwhm_y)
    newpar, newpar_err, exppar, exppar_err = MLE_fitting(fwhm_y, sigma = 0.01)
    
    
    max_hist = np.min([10,np.max(fwhm_y)])
    bins = np.arange(np.min(fwhm_y),max_hist,0.05)
    
    x = np.linspace(0,10,1000)
#    model1 = skewnorm.pdf(x,*newpar)*(bins[1]-bins[0])*fwhm_y.size
#    model2 = skewnorm.pdf(x,*meanpar)*(bins[1]-bins[0])*fwhm_y.size
#    model3 = exponnorm.pdf(x, *exppar)*(bins[1]-bins[0])*fwhm_y.size
    model4 = exponnorm.pdf(x, *meanexp)*(bins[1]-bins[0])*fwhm_y.size

    fig, ax = plt.subplots(1,1, figsize=(8,6))
    sns.distplot(fwhm_y, bins = bins, kde = False, label = 'FWHM', ax = ax)    
#    ax.plot(x, model1, label = 'Built in (sk)')
#    ax.plot(x, model2, label = 'Dedicated fit (sk)')
#    ax.plot(x, model3, label = 'Built in (ex)')
    ax.plot(x, model4, label = 'Dedicated fit (ex)')
    if psf_file:
        plt.axvline(fwhm_line_y, c = 'black', label = 'Star_y', ls = ':')
    plt.legend(loc = 'best')
    if save:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
    
    #printing the results of the plot


    result = {'shape': [meanpar[0]], 'shape_err': [stdpar[0]], 
              'location': [meanpar[1]], 'location_err': [stdpar[1]],
              'scale' : [meanpar[2]], 'scale_err' : [stdpar[2]], 
              'shape_b': [newpar[0]], 'shape_b_err': [newpar_err[0]],
              'location_b': [newpar[1]], 'location_b_err': [newpar_err[1]],
              'scale_b' : [newpar[2]],'scale_b_err': [newpar_err[2]],
              'e_shape_b': [exppar[0]], 'e_shape_b_err': [exppar_err[0]],
              'e_location_b': [exppar[1]], 'e_location_b_err': [exppar_err[1]],
              'e_scale_b' : [exppar[2]],'e_scale_b_err': [exppar_err[2]],
              'e_shape': [meanexp[0]], 'e_shape_err': [stdexp[0]], 
              'e_location': [meanexp[1]], 'e_location_err': [stdexp[1]],
              'e_scale' : [meanexp[2]], 'e_scale_err' : [stdexp[2]]}
    
    if psf_file:
        result['fwhm_star'] = fwhm_line_y
        
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
             circular = False, function = 'gauss', plot = False):
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
        print('region:', region)
        #defining the coordinates of the region and selecting a cutout
        size = 3*size_r
        if size < 2.0:
            size = 2.0
        coords = SkyCoord(ra_r,dec_r, frame = FK5, unit=(u.deg,u.deg))
        region_cut = image.subimage(center = (coords.dec.deg, coords.ra.deg),
                                    size = size)

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

        fit, fit_ima, chi2_test = fu.fit_function(region_cut, 
                                               function = function,
                                               circular = circular,
                                               verbose = False)


            
        if plot:
            print('fwhm_x: {:1.3f}, fwhm_y: {:1.3f}, theta: {:1.3f}' 
                  .format(fit.fwhm[1], fit.fwhm[0], fit.rot))
            plots_2d(region_cut.data, fit_ima.data)

        if mask:
            #resetting the mask
            region_cut.data.mask[mask_array] = False
            
            #print(fit.theta[0])
        print('chi2: {:1.3f}' .format(chi2_test))
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

def measure_PSF(cube, catalogue, line, galaxy, point, x0 = None, y0 = None, 
                function = 'moff',  plot = False, mask = False, 
                circular = False, interactive = False):
    
    hdu = fu.open_hdu(cube)
    image, header, wcs, err = fu.open_image(hdu, line.upper(), err = True)

#    image = fu.create_masked(image, err, 3)
    #carico le regioni a mano per ora
    regions, ra, dec, size = fu.read_table(catalogue)  
    
#    planetary = ascii.read(table)
#    regions, ra, dec, size = planetary['reg'],planetary['ra'],planetary['dec'],planetary['size']
    #identifying the pointing thanks to the coordinate of the center
    #of the field
    pointing_coords = ascii.read('/media/congiu/Data/Dati/PHANGS/star_fit/pointing_coords')
    for i, file in enumerate(pointing_coords['file']):
        if galaxy.upper() in file and point in file:
                
            x0, y0 = wcs.wcs_world2pix(pointing_coords['RA'][i],
                                       pointing_coords['DEC'][i], 0) 
            print(x0,y0)
            plt.imshow(image, origin = 'lower', vmin = 0, vmax = 3000)
            plt.scatter(x0,y0)
            plt.show()
            pointing_cut = Cutout2D(image, (x0, y0), 
                                    size = 280* u.pixel, 
                                    wcs = wcs)    
    if interactive:
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
#        if region == 461 or region == 546:
#            continue #this region gives some problem

        fit_result, fit_chi2 = HII_fit(region, ra_r, dec_r, size_r, new_image,
                             function = function, circular = circular,
                             plot = plot, mask = mask)
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
    
    warnings.simplefilter("ignore")
    start = time.time()
    galaxy = 'NGC1672'
    point = 'P02'
#    x = '152'
#    y = '220'

#    psf_file = './plots/Moffat_'+galaxy+'_'+point+'_x'+x+'_y'+y+'circ.txt'
    #defining input quantities
    cube = '../Mosaics/DATACUBE_FINAL_'+galaxy.upper()+'_1_comp_mosaic.fits'
    table = '../catalogues_and_masks/'+galaxy.lower()+'_hiicat_v1.fits'
    lines = ['HALPHA']
    wavelength = [6562]

    #questa è una prova per vedere se le PNE funzionano
    
#    table = '../Mosaics/planetary'

    #output file
    out_name = './plots/hii_fit_'+galaxy+'_'+point+'.txt'
    x0 = None
    y0 = None
    #performing the fit for each line map
    for (line,wave) in zip(lines,wavelength):
        print('line: '+line)
        #fitting the PSF
        parameters, x0, y0 = measure_PSF(cube, table, line, galaxy, point,
                                         function = 'moff', circular = True,
                                         interactive = False, 
                                         plot = False, x0 = x0, y0 = y0, 
                                         mask = True) 
        #plotting the histogram
        result = fwhm_distribution(parameters, clip = False)
        result = pd.DataFrame.from_dict(result)
        #saving the results of the fit
        result.to_csv(out_name, sep = ' ', index = False)
    
        
    print('Completed in {:1.2f} seconds' .format(time.time()-start))
        
    
    