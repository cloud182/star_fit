#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 11:28:07 2019

@author: congiu
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.modeling import models, fitting
from scipy.stats.stats import pearsonr  
from scipy.stats import exponnorm, chisquare
from scipy.optimize import minimize 
from astropy.table import join
import os
import sys


def function1():
    """
    confronta la fwhm a 5000A con una delle misure di seeing salvate 
    nell'header
    """
    
    info = ascii.read('./info_header')
    
    seeing = ascii.read('/media/congiu/Data/Dati/PHANGS/star_fit/reliable_pointing_ha.txt')
    
    new_table = join(info,seeing, keys='pointing')

    fig, ax = plt.subplots(1,1, figsize = (10,10))

    for i, value in enumerate(new_table['pointing']):
        ax.errorbar(new_table['seeing3'][i],new_table['fwhm'][i],
                    new_table['fwhm_err'][i], ls = '', marker = 'o', label = value)
    
    fitter = fitting.LinearLSQFitter()
    model = models.Linear1D(1,0)
    fit = fitter(model, new_table['seeing2'],new_table['fwhm'])#, weights = 1/new_table['fwhm_err'])
    chi2 = np.sum((new_table['fwhm'] - fit(new_table['seeing3']))**2/(fit(new_table['seeing3'])))\
                    /(len(new_table['seeing2'])-2)
    
    ax.set_xlim(0.6,1.3)
    ax.set_ylim(0.6,1.3)
    ax.plot(new_table['seeing3'], fit(new_table['seeing3']), label = 'fit')
    ax.plot([],[], ls = '',label = 'm = {:1.2f}' .format(fit.slope[0]))
    ax.plot([],[], ls = '',label = 'q = {:1.2f}' .format(fit.intercept[0]))
    ax.plot([],[], ls = '',label = 'chi2 = {:1.3f}' .format(chi2))
    plt.plot([0,2],[0,2], c = 'k', ls = '--')
    ax.set_xlabel('Seeing from header (arcsec)')
    ax.set_ylabel('FWHM at 6500A')
    ax.set_title('FWHMLISOBS')
    plt.legend(loc='best')
#    plt.savefig('fwhm_seeing.png', dpi = 150)
#    plt.close()
    plt.show()

def function2():
    """
    plotta i valori di n in funzione della lunghezza d'onda
    """

    directory = '/media/congiu/Data/Dati/PHANGS/star_fit/plots/'
    
    fig, ax = plt.subplots(1,1)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('circ.txt'):
            table = ascii.read(directory+filename)
            print(filename, np.nanmean(table['n']))
            ax.errorbar(table['wavelength'], table['n']/np.nanmean(table['n']), 
                        yerr = table['err_n']/np.nanmean(table['n']),
                        label = filename)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

#    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Wavelength $\\AA$')
    ax.set_ylabel('Power index')
    plt.tight_layout()
    plt.show()
#    plt.savefig('skewness.png', dpi = 300 )
#    plt.close()
      
    
def function3():
    """
    Come funzione 2 ma per la fwhm e le curve sono normalizzate
    """
    
    directory = '/media/congiu/Data/Dati/PHANGS/star_fit/plots/'
    
    fig, ax = plt.subplots(1,1)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('nmean.txt'):
            table = ascii.read(directory+filename)
            ax.errorbar(table['wavelength'], 
                        table['fwhm_y']/np.nanmean(table['fwhm_y']),
                        yerr = table['err_fwhm_y'],
                        label = filename)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

#    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Wavelength $\\AA$')
    ax.set_ylabel('FWHM')
    plt.tight_layout()       
    plt.show()
        
def function4():
    """
    Plotta l'andamento di n con la lunghezza d'onda, fa il fit per ogni stella
    e calcola la pendenza media
    """
    
    directory = '/media/congiu/Data/Dati/PHANGS/star_fit/plots/'
    
    slope = []
    err = []
    
    fitter = fitting.LevMarLSQFitter()
    model = models.Linear1D(1,1)
    
    fig, ax = plt.subplots(1,1)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('circ.txt'):
            table = ascii.read(directory+filename)
            fit = fitter(model, table['wavelength'], table['n'], 
                         weights = 1/table['err_n'])
            error = np.diag(fitter.fit_info['param_cov'])**2
            slope.append(fit.slope[0])
            err.append(error[0])
            
            ax.errorbar(table['wavelength'], table['n']/np.mean(table['n']),
                        yerr = 0)
    
    slope = np.array(slope)
    err = np.array(err)
    slope_mean = np.average(slope, weights = 1/err)
    slope_std = np.sqrt(np.average((slope-slope_mean)**2,weights = 1/err))
    
    print('slope: {:1.6f}+-{:1.6f}' .format(slope_mean, slope_std))
    
    
def function5():
    """
    confronta il seeing misurato manualmente nelle immagini in luce bianca
    con il seeing ad una certa lunghezza d'onda ottenuto con un fit
    """
    
    directory = '/media/congiu/Data/Dati/PHANGS/star_fit/plots/'
    
    seeing = ascii.read('unresolved_seeing.txt')
    
    fwhm = []
    n = []
    white_seeing = []
    white_n = []
    
    fitter = fitting.LevMarLSQFitter()
    model = models.Linear1D(1,1)
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.startswith('Moffat') and filename.endswith('circ.txt'):
            table = ascii.read(directory+filename)
            fit1 = fitter(model, table['wavelength'], table['n'], 
                         weights = 1/table['err_n'])
            fit2 = fitter(model, table['wavelength'], table['fwhm_y'],
                         weights = 1/table['err_fwhm_y'])
            fwhm.append(fit2(6500))
            n.append(fit1(6500))
            for i, arg in enumerate(seeing['pointing']):
                if arg in filename:
                    print(arg, seeing['n'][i])
                    white_seeing.append(seeing['fwhm'][i])
                    white_n.append(seeing['n'][i])
     
    fit = fitter(model, fwhm, white_seeing)   
    print(fit.slope, fit.intercept)      

    fig, ax = plt.subplots(1,1)
    ax.set_xlim(0.5,1.1)
    ax.set_ylim(0.5,1.1)
    ax.scatter(fwhm,white_seeing)
    ax.plot([0,1.5], [0,1.50], c = 'k')
    ax.plot(fwhm, fit(fwhm))
    ax.set_ylabel('white light fwhm (arcsec)')
    ax.set_xlabel('6500A fwhm (arcsec)')
    ax.set_title('Unresolved')
#    plt.show()
    plt.savefig('unresolved_seeing.png')
    plt.close()

    fit_n = fitter(model, n, white_n)
    print(fit_n.slope,fit_n.intercept)
    
    fig, ax = plt.subplots(1,1)
    ax.set_xlim(2.25,3.75)
    ax.set_ylim(2.25,3.75)
    ax.scatter(n, white_n)
    ax.plot([2.25,3.75], [2.25,3.75], c = 'k')
    ax.plot(n, fit_n(n))
    ax.set_xlabel('white light n (arcsec)')
    ax.set_ylabel('6500A n (arcsec)')
#    plt.show()
    ax.set_title('Unresolved')
    plt.savefig('unresolved_n.png')
    plt.close()

def function6():
    """
    fa il fit per ogni singola stella del parametro n e plotta singolarmente
    """

    directory = '/media/congiu/Data/Dati/PHANGS/star_fit/plots/'
    
    fitter = fitting.LevMarLSQFitter()
    model = models.Linear1D(1,1)
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('circ.txt'):
            table = ascii.read(directory+filename)
            fit = fitter(model, table['wavelength'], table['n'], 
                         weights = 1/table['err_n'])
            fig, ax = plt.subplots(1,1)
            ax.errorbar(table['wavelength'], table['n'], yerr = table['err_n'],
                        label = filename)
            ax.plot(table['wavelength'], fit(table['wavelength']), 
                    label ='fit')
            ax.plot([],[], ls = '',label = 'm = {:1.5f}' .format(fit.slope[0]))
            ax.plot([],[], ls = '',label = 'q = {:1.5f}' .format(fit.intercept[0]))
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.show()
#    plt.savefig('skewness.png', dpi = 300 )
#    plt.close()
            
            
def function7():
    """ 
    Fa il plot del seeing a 6562 A con il i parametri location ottenuti 
    dal fit della distribuzione della FWHM delle regioni H II
    """
        
    directory = '/media/congiu/Data/Dati/PHANGS/star_fit/plots/'

    ha_table = ascii.read('/media/congiu/Data/Dati/PHANGS/star_fit/reliable_pointing_ha.txt')
    seeing_ha = []
    location = []
    location_err = []
    fitter = fitting.LinearLSQFitter()
    model = models.Linear1D(1,1)
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.startswith('hii'):
            table = ascii.read(directory+filename)
            for i,value in enumerate(ha_table['pointing']):
                if value in filename:
                    seeing_ha.append(ha_table['fwhm'][i])
                    
                    location.append(table['e_location'][0])
                    location_err.append(table['e_location_err'][0])
                    ax.errorbar(ha_table['fwhm'][i], table['e_location'], 
                                yerr = table['e_location_err'], marker = 'o',
                                label = value)
    print(location)
    location_err = np.array(location_err)
    fit = fitter(model, seeing_ha, location, weights = 1/location_err)
    plt.plot(seeing_ha, fit(seeing_ha), label = 'fit')
    chi2 = np.sum((location-fit(seeing_ha))**2/fit(seeing_ha))/(len(location)-2)
    ax.set_xlabel('FWHM at 6562A')
    ax.set_ylabel('loc')   
    ax.set_xlim([0.4,1.1])
    ax.set_ylim([0.4,1.1])
    ax.plot([0.4,1.1],[0.4,1.1], label = 'bisector')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.plot([],[], ls = '', label = 'chi2 = {:0.2f}' .format(chi2))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('location (exponnorm)')
    plt.savefig('location_exponnorm.png')
    plt.close()
#    plt.show()
                        
def function8():
    directory = '/media/congiu/Data/Dati/PHANGS/star_fit/plots/'                  
          
    fitter = fitting.LinearLSQFitter()
    model = models.Linear1D(1,1)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('circ.txt'):
            table = ascii.read(directory+filename)
            fit = fitter(model,table['wavelength'], table['fwhm_y'],
                         weights = 1/table['err_fwhm_y'])
            print(filename, fit(6562))
            
def function9():
    def to_minimize(x, param):
        return - exponnorm.pdf(x, *param)
    
    def mode(param):
        x = np.linspace(0,10, 10000)
        model = exponnorm.pdf(x, *param)
        index = np.argmax(model)
        mode = x[index]
        return mode
    
    """ fa il plot di funzione 7 ma con la moda"""
    directory = '/media/congiu/Data/Dati/PHANGS/star_fit/plots/'

    ha_table = ascii.read('/media/congiu/Data/Dati/PHANGS/star_fit/reliable_pointing_ha.txt')
    seeing_ha = []
    mode1 = []
    region = []
    fitter = fitting.LinearLSQFitter()
    model = models.Linear1D(1,1)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.startswith('hii'):
            table = ascii.read(directory+filename)
            for i,value in enumerate(ha_table['pointing']):
                if value in filename:
                    seeing_ha.append(ha_table['fwhm'][i])
                    param = table['e_shape'][0],table['e_location'][0],table['e_scale'][0]
                    mode_val = mode(param)
                    mode1.append(mode_val)
                    region.append(value)

    mode1 = np.array(mode1)
    print(mode1)
    
    ratio = seeing_ha/mode1
    
    median_ratio = np.median(ratio)
    print(median_ratio)
    fig, ax = plt.subplots(1,1, figsize=(8,6))

    
    for seeing, mode, region in zip(seeing_ha,mode1,region):
        ax.scatter(seeing, mode*median_ratio, label = region)
    

    fit = fitter(model, seeing_ha, mode1*median_ratio)
    rms = np.sqrt(np.sum((mode1*median_ratio-fit(seeing_ha))**2)/len(mode1))
    ax.plot(seeing_ha, fit(seeing_ha), label = 'fit')
    ax.plot([0.4,1.3], [0.4,1.3])

    ax.plot([],[], ls ='', label = 'slope = {:0.2f}' .format(fit.slope[0]))
    ax.plot([],[], ls ='', label = 'int. = {:0.2f}' .format(fit.intercept[0]))
    ax.plot([],[], ls ='', label = 'ratio = {:0.2f}' .format(median_ratio))
    ax.plot([],[], ls ='', label = 'sigma = {:0.2f}' .format(rms))

    ax.set_xlim(0.4,1.2)
    ax.set_ylim(0.4,1.2)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('FWHM 6562A')
    ax.set_ylabel('Scaled Mode')
    ax.set_title('Scaled Mode (exponnorm)')
    plt.savefig('scaled_mode_exponnorm.png')
    plt.close()


def function10():
    """
    faccio il plot del seeing misurato con le nebulose e di quello dalle stelle
    """
    

    pn_table = ascii.read('/media/congiu/Data/Dati/PHANGS/PN_selection/'\
                          +'psf_planetary.txt')
    o3_table = ascii.read('/media/congiu/Data/Dati/PHANGS/star_fit/reliable_pointing_o3.txt')

    new_table = join(pn_table,o3_table, keys='pointing')
    

    fitter = fitting.LinearLSQFitter()
    model = models.Linear1D(1,1)
    
    
    
    
    
    fig, ax = plt.subplots(1,1, figsize=(8,6))

    for o3, pn, err, region in zip(new_table['fwhm_2'],
                                   new_table['fwhm_1'],
                                   new_table['fwhm_err_1'],
                                   new_table['pointing']):
        
        ax.errorbar(o3, pn, yerr = err, label = region, ls = '', marker = 'o')

    fit = fitter(model, new_table['fwhm_2'], new_table['fwhm_1'],
                 weights = 1/new_table['fwhm_err_1'])
    print(len(new_table['fwhm_2'].pformat()))

    chi2 = np.sum((new_table['fwhm_1']-fit(new_table['fwhm_2']))**2\
                  / fit(new_table['fwhm_2']))/(len(new_table['fwhm_2'].pformat()))
    

    plt.plot(new_table['fwhm_2'], fit(new_table['fwhm_2']), label = 'fit')
    ax.set_xlabel('FWHM at 5007A')
    ax.set_ylabel('PN FWHM')   
    ax.set_xlim([0.5,1.2])
    ax.set_ylim([0.5,1.2])
    ax.plot([0.4,1.2],[0.4,1.2], label = 'bisector')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.plot([],[], ls = '', label = 'chi2 = {:0.2f}' .format(chi2))
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('PN FWHM (exponnorm)')
#    plt.savefig('location_exponnorm.png')
#    plt.close()
    plt.show()    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    function3()
    
    