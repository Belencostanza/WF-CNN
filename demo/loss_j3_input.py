#funcion lossj3 para planck, no la puedo programar con quicklens ya que no podria usar tensorflow gpu

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
import scipy

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

import camb
from camb import model, initialpower
#import healpy as hp
from read_input import load_config
#import utilities

mask = np.load('mask_planck.npy')

################################################################
#Warning message if you have more than one argument:
if len(sys.argv) != 2:
    print("Usage: python your_script.py input_file")
    sys.exit(1)

input_file = sys.argv[1]

#Dictionary with the parameters of the input file:
input_data = load_config(input_file)

use_variance = input_data["use_variance"]

use_variance = int(use_variance)

lmax=7000
Lsize = int(input_data["Lsize"])
nx = int(input_data["npixels"])
d2r = np.pi/180.
dx = Lsize*d2r / float(nx) # size of pixels in radians
fwhm_arcmin = 5.
deviation = float(input_data["deviation"])
map_rescale_factor = 1./deviation


def bl(fwhm_arcmin, lmax):
    ls = np.arange(0,lmax+1)
    return np.exp( -(fwhm_arcmin * np.pi/180./60.)**2 / (16.*np.log(2.)) * ls*(ls+1.) )

#power spectra 

def get_power_spectrum():

    #Set up a set of parameters for CAMB (with r=0.1)
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0.1)
    pars.set_for_lmax(lmax, lens_potential_accuracy=1)
    pars.WantTensors = True

    #calculate results for these parameters
    results = camb.get_results(pars)

    #get dictionary of CAMB power spectra
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')

    unlensedCL=powers['unlensed_scalar']
    unlensed_Tot=powers['unlensed_total']
    lensed_SC=powers['lensed_scalar']

    ls = np.arange(unlensedCL.shape[0])
    factor = ls*(ls+1)/2./np.pi
    cl_TT_for_map = np.copy(unlensedCL[:,0])  #copio en un array nuevo el Dl_TT unlensed (como sale del CAMB)
    cl_TT_for_map[2:] = cl_TT_for_map[2:]/factor[2:]
    # cl_TT_for_map[0:2]=0.0
    
    beam = bl(fwhm_arcmin, lmax)
    cl_tt = cl_TT_for_map[:lmax+1]
    cl_tt = cl_tt * (map_rescale_factor**2.)
    
    lx,ly=np.meshgrid( np.fft.fftfreq( nx, dx )[0:int(nx/2+1)]*2.*np.pi,np.fft.fftfreq( nx, dx )*2.*np.pi )
    l = np.sqrt(lx**2 + ly**2)
    ell = l.flatten()
    ellmask_nonzero = np.logical_and(ell >= 2, ell <= lmax)
    ell_ql = np.arange(0,cl_tt.shape[0])
    
    cl_tt[0:2] = cl_tt[2]
    interp = scipy.interpolate.interp1d(np.log(ell_ql[1:]),np.log(cl_tt[1:]), kind='linear',fill_value=0,bounds_error=False)
    cl_tt = np.zeros(ell.shape[0])
    cl_tt[1:] = np.exp(interp(np.log(ell[1:])))
    cl_tt[0] = cl_tt[1]

    return cl_tt

def variance_inho():
    #nlev_t = 35.
    #rad2arcmin = 180.*60./np.pi
    #noise_pix = nlev_t**2. / (dx**2 * rad2arcmin**2)
    npix=512
    sigma0=600
    delta0=1
    sigma2_noise_map=np.fromfunction(lambda i, j: sigma0*(1+delta0*i/npix), 
                                 (npix, npix), dtype=float)
    return sigma2_noise_map
    
def variance_cuad():

    npix=512
    sigma0=600
    delta0=0.01
    a= delta0
    b= delta0*(1-npix)
    c= sigma0 + delta0*((1-npix)**2)/4.
    
    sigma2_cuad_map=np.fromfunction(lambda i, j: a*i**2 + b*i + c,
                                 (npix, npix), dtype=float)

    return sigma2_cuad_map


def planck_variances():
    #nlev_t = 35.
    #rad2arcmin = 180.*60./np.pi
    #noise_pix = nlev_t**2. / (dx**2 * rad2arcmin**2)
    muestras = np.load('variance_map2.npy')
    muestra1 = muestras[:,:,0]
    muestra2 = muestras[:,:,1]
    muestra3 = muestras[:,:,2]
    muestra1_uk = 1e12*muestra1
    muestra2_uk = 1e12*muestra2
    muestra3_uk = 1e12*muestra3

    return muestra1_uk, muestra2_uk, muestra3_uk


def inverse_cl(data):
    cl = get_power_spectrum()
    data = tf.math.divide_no_nan(data,cl)
    return data

def realspace_loss(y_true, y_pred):
    #square map in real space and multiply by mask to remove masked pixels. weight the other by pixel noise variance.
    y_true = y_true[:,:,:,0]
    y_pred = y_pred[:,:,:,0]
    
    if use_variance == 0:
        noise_pix = variance_inho()
    elif use_variance == 1:
        noise_pix = variance_cuad()
    elif use_variance == 2:
        noise_pix,_,_ = planck_variances()
    elif use_variance == 3:
        _,noise_pix,_ = planck_variances()
    elif use_variance == 4:
        _,noise_pix,_ = planck_variances()
        noise_pix = 4*noise_pix
    elif use_variance == 5:
        noise_pix,_,_ = planck_variances()
        noise_pix = noise_pix/4.
        
    loss = (y_pred - y_true)*(y_pred - y_true) / (noise_pix*map_rescale_factor**2.)

    loss = loss*mask

    loss = tf.reduce_mean(loss)

    return loss

def fourier_loss(y_pred):
    rmap = y_pred[:,:,:,0]

    # tengo que tomar transformada de fourier real al mapa
    # 2D real valued fast fourier transform con tensorflow
    # since DFT of a real signal is Hermitian-symmetric, rfft2 only returns the FFT_lenght/2 + 1: the zero frequency term, followed by the fft_lenght/2 positive-frequency terms
    #(siguiendo quicklens)
    tfac = np.sqrt((dx*dx)/(nx*nx))
    rfft = tf.compat.v1.spectral.rfft2d(rmap)*tfac
    rfft_shape = rfft.get_shape().as_list()
    power = tf.math.real((rfft * tf.math.conj(rfft)))
    power = tf.reshape(power,[-1,rfft_shape[1]*rfft_shape[2]])

    #weight by signal power spectrum
    power = inverse_cl(power)

    loss = tf.reduce_mean(power)

    return loss



def loss_wiener_j3(y_true, y_pred):
    #real space noise weighted difference
    term1 = realspace_loss(y_true, y_pred)
    #fourier space on the input map
    term2 = fourier_loss(y_pred)

    loss = term1 + term2
    return loss
