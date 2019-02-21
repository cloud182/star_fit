#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 12:36:39 2019

@author: congiu
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter
from astropy.modeling import models,fitting
from astropy.io import ascii
import os
import sys

def fit_error(fitter, model):
    
    cov = fitter.fit_info['param_cov']
    try:
        error = dict(zip(model.param_names, np.diag(cov)**0.5))
    except ValueError:
        error = dict(zip(model.param_names, 
                         np.arange(len(model.param_names))*0))
    return error

def fit_parameter(table, field, plot= False, name=None , save = False):
    fitter = fitting.LevMarLSQFitter()
    model = models.Linear1D(0.5,0.8)
    x = table['wavelength']
    y = table[field.lower()]
    yerr = table['err_'+field.lower()]
    fit = fitter(model, x,y, weights = 1/yerr)
    error = fit_error(fitter, model)
    if plot:
        fig, ax = plt.subplots(1,1)
        ax.errorbar(x,y, yerr=yerr, ls = '', marker = 'o')
        ax.plot(x,fit(x))
        if save:
            plt.savefig(name)
            plt.close()
        else:
            plt.show()
    results = {'slope':fit.slope[0], 'slope_err': error['slope'],
               'intercept':fit.intercept[0], 'intercept_err': error['intercept']}
    return results
    
def measure_slope(directory, outname, field):
    f = open(outname, 'w')
    print('pointing','slope' , 'slope_err', 'intercept', 'intercept_err',
          file = f)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.startswith('Moffat') and filename.endswith('.txt'):
            table = ascii.read(os.path.join(directory,filename))
            results = fit_parameter(table,field,plot = True)
            #priting on screen
            print('slope: {:1.2e}+-{:1.2e}' 
                  .format( results['slope'], results['slope_err']))
            #printing on file
            print(filename[7:-13].strip('_'), results['slope'], 
                  results['slope_err'], results['intercept'], 
                  results['intercept_err'], file = f)
    f.close()
    
    
if __name__ == '__main__':
    directory = '/media/congiu/Data/Dati/PHANGS/star_fit/plots/'
    outname = '/media/congiu/Data/Dati/PHANGS/star_fit/linear_fit'
    headname = '/media/congiu/Data/Dati/PHANGS/star_fit/info_header'
    measure = True
    if measure:
        measure_slope(directory,outname, 'fwhm_y')
        
    fit = ascii.read(outname)
    header = ascii.read(headname)
    fit['MJD'] = np.zeros(len(fit['pointing']))
    fit['airmass'] = np.zeros(len(fit['pointing']))
    fit['seeing'] = np.zeros(len(fit['pointing']))
    fit['rot'] = np.zeros(len(fit['pointing']))
    
    for i, item in enumerate(fit['pointing']):
        if item in header['file']:
            idx = np.where(header['file']==item)
            fit['MJD'][i] = header['MJD'][idx]
            fit['airmass'][i] = header['airmass'][idx]
            fit['seeing'][i] = header['seeing'][idx]
            fit['rot'][i] = header['rot'][idx]

    fig, ax = plt.subplots(2,2, sharey = True)
    
    ax[0,0].errorbar(fit['airmass'], fit['slope'], yerr = fit['slope_err'],
                      ls = '', marker = 'o')
    ax[0,0].set_xlabel('airmass')
    ax[0,0].set_ylabel('slope')
    ax[0,0].xaxis.set_label_position('top')
    ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

    
    ax[1,0].errorbar(fit['MJD'], fit['slope'], yerr = fit['slope_err'],
                      ls = '', marker = 'o')
    ax[1,0].set_xlabel('MJD')
    ax[1,0].set_ylabel('slope')
    ax[1,0].yaxis.set_major_formatter(FormatStrFormatter('%.2e'))
                
    ax[0,1].errorbar(fit['rot'], fit['slope'], yerr = fit['slope_err'],
                      ls = '', marker = 'o')
    ax[0,1].set_xlabel('rot')
    ax[0,1].xaxis.set_label_position('top')

                
    ax[1,1].errorbar(fit['seeing'], fit['slope'], yerr = fit['slope_err'],
                      ls = '', marker = 'o')
    ax[1,1].set_xlabel('seeing (FWHM)')
#    plt.tight_layout()
    fig.suptitle('FWHM_y')
    
    plt.show()
            
            
            
            
            
    
    
    
    
    
    
            
            

