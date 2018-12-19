import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astropy.stats import gaussian_sigma_to_fwhm, gaussian_fwhm_to_sigma
import sys
import time
import timeit

###############################################################

def plots_2d(f1, f2, filename=None, save= False):
    residuals = f1/f2
    print('max: {:1.4f}, min: {:1.4f}, median: {:1.4f}' 
          .format(np.max(residuals), np.min(residuals), np.median(residuals)))
    # plotting the original image, the fit and the residuals
    fig, ax = plt.subplots(1,3, figsize=(15,4))
    im0 = ax[0].imshow(f1, vmin = np.min(f1), vmax = np.max(f1),
                       origin = 'lower')
    ax[0].set_title('PSF1')
    fig.colorbar(im0, ax = ax[0])
    im1 = ax[1].imshow(f2, vmin = np.min(f1), vmax = np.max(f1), 
                       origin = 'lower')
    ax[1].set_title('PSF2')
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
###############################################################
        
def gaussian_2D(I0, x0, y0, fwhm, x, y):
    #converting from arcsec to pixel using 0.2"/px scale
    fwhm = fwhm*5 
    x0 = x0*5
    y0 = y0*5
    sigma2 = (fwhm*gaussian_fwhm_to_sigma)**2
    gauss = I0*np.exp(-((x-x0)**2/(2*sigma2)+(y-y0)**2/(2*sigma2)))
    return gauss

def moffat_2D(I0, x0, y0, fwhm, alpha, x, y):
    #converting from arcsec to pixel using 0.2"/px scale
    fwhm = fwhm * 5
    x0 = x0*5
    y0 = y0*5
    gamma = np.sqrt(0.5*fwhm/np.sqrt(2**(1/alpha)-1))
    print(gamma)
    moffat = models.Moffat2D.evaluate(x,y,I0,x0,y0,gamma, alpha)
    return moffat
    

###############################################################

def radial_plot(f1, f2, filename = None, save= False):
#    radial_ima = np.sum(ima, axis = 0)
#    radial_fit = np.sum(fit, axis = 0)
    radial_f1_x = f1[20,:]
    radial_f2_x = f2[20,:]
    pixel = np.arange(len(radial_f1_x))
    fig = plt.figure()
    frame1 = fig.add_axes((0.1,0.3,0.8,0.6))
    frame1.scatter(pixel, radial_f1_x, c= 'b', label = 'x axis')
    frame1.plot(pixel, radial_f2_x, c='b')
    frame1.set_xticklabels([])
    frame1.set_ylabel('flux')
    frame1.legend(loc = 'best')
    frame2 = fig.add_axes((0.1,0.1,0.8,0.2))
    residual_x = radial_f1_x/radial_f2_x
    # setting the range of the y axis looking for the max (in absolute value)
    # of the residual, and incrasing it of a 50 %
    frame2.set_ylim(0.8, 1.2)
    frame2.scatter(pixel, residual_x, c = 'b')
    frame2.axhline(y=1.)
    frame2.set_xlabel('pixels')
    frame2.set_ylabel('Ratio')
    if save:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

###############################################################
        
def create_image( x, y, x1 = 0, fwhm = 1, fwhm_0 = None, function='gauss'):
    if fwhm_0 == None:
        fwhm_0 = fwhm
    if function == 'gauss':
        f1 = gaussian_2D(1,0,0,fwhm_0,x,y)
        f2 = gaussian_2D(1,x1,0,fwhm,x,y)
    elif function == 'moff':
        #to be changed into a moffat
        f1 = moffat_2D(1, 0, 0, fwhm_0, 2, x,y)
        f2 = moffat_2D(1, x1, 0, fwhm, 2, x,y)
    else:
        print('Function not supported')
        sys.exit()
    return f1, f2
        
###############################################################
    
def psf_test(dx, nstep_x, dfwhm, nstep_fwhm, fwhm_0 = None, function = 'gauss', save = False):
    """
    Function to thest the shape of the residuals of PSF fit.10
    Input:
    - dx: x step for the center of the second PSF
    - nstep_x: number of steps in dx
    - dfwhm: step in fwhm for both PSFs
    - nstep_fwhm: number of steps in fwhm
    - function: type of function to be used 
                - 'gauss': gaussian
                - 'moff': moffat
    -save: if True the plots a re saved
    Output:
    - 1D and 2D plots of the fits and the residuals
    """
    
    #I define the grid where I'll build the images and perform the fit
    x, y = np.meshgrid(np.arange(-20,21,1), np.arange(-20,21,1))

    #I start with looping on the fwhm
    step_fwhm = 0
    fwhm = 0.5          #I'm starting from a FWHM of 0.5    
    while step_fwhm < nstep_fwhm:
        #for each fwhm I loop on the dx
        step_x = 0
        x1 = 0
        while step_x < nstep_x:
            #creating the simulated image
            f1, f2 = create_image(x, y, x1=x1, fwhm = fwhm, fwhm_0 = fwhm_0,
                             function = function)
            #defining fit names
            filename_2d = function\
                          + '_2d_dx_{:1.2f}_fwhm_{:1.2f}.png' .format(x1, fwhm)
            filename_1d = function \
                          + '_1d_dx_{:1.2f}_fwhm_{:1.2f}.png' .format(x1, fwhm)
            #plotting the 2d maps and the 1d fits
            plots_2d(f1, f2, filename = filename_2d, save = save)
            radial_plot(f1, f2, filename = filename_1d, save = save)
            x1 = x1+dx
            step_x += 1
        step_fwhm += 1
        fwhm += dfwhm

###############################################################   
def psf_test2(dfwhm, nstep_fwhm, fwhm_0 = 1.0, function = 'moff', save = False):
    """
    Function to thest the shape of the residuals of PSF fit.10
    Input:
    - dx: x step for the center of the second PSF
    - nstep_x: number of steps in dx
    - dfwhm: step in fwhm for both PSFs
    - nstep_fwhm: number of steps in fwhm
    - function: type of function to be used 
                - 'gauss': gaussian
                - 'moff': moffat
    -save: if True the plots a re saved
    Output:
    - 1D and 2D plots of the fits and the residuals
    """
    
    #I define the grid where I'll build the images and perform the fit
    x, y = np.meshgrid(np.arange(-20,21,1), np.arange(-20,21,1))

    #I start with looping on the fwhm
    step_fwhm = 0
    fwhm = 0.5          #I'm starting from a FWHM of 0.5    
    while step_fwhm < nstep_fwhm:
        #for each fwhm I loop on the dx
        #creating the simulated image
        f1, f2 = create_image(x, y, x1=0, fwhm = fwhm, fwhm_0 = fwhm_0,
                             function = function)
        #defining fit names
        filename_2d = function\
                      + '_2d_fwhm1_{:1.2f}_fwhm2_{:1.2f}.png' .format(fwhm_0, fwhm)
        filename_1d = function \
                      + '_1d_fwhm1_{:1.2f}_fwhm2_{:1.2f}.png' .format(fwhm_0, fwhm)
        #plotting the 2d maps and the 1d fits
        plots_2d(f1, f2, filename = filename_2d, save = save)
        radial_plot(f1, f2, filename = filename_1d, save = save)
        step_fwhm += 1
        fwhm += dfwhm
  
###############################################################

if __name__ == '__main__':
    start = time.time()
#   psf_test(0.05,20,0.5,10, function = 'moff', save = True)
    psf_test2(0.1,40, fwhm_0 = 1., function = 'moff', save = True)
    print('Completed in {:4.4f} s' .format(time.time() -start))
