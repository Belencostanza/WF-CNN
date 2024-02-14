import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os
import scipy 
from scipy import interpolate
import cmath

import camb
from camb import model, initialpower
from read_input import load_config

import time

#parametros a elegir

if len(sys.argv) != 2:
    print("Usage: python your_script.py input_file")
    sys.exit(1)

input_file = sys.argv[1]

#Dictionary with the parameters of the input file:
input_data = load_config(input_file)


#-----------------------------------------------
nx = int(input_data["npixels"])
Lsize = int(input_data["Lsize"])
npad = int(input_data["npad"])
deviation = float(input_data["deviation"])
map_rescale_factor = 1./deviation

d2r = np.pi/180.
dx = Lsize*d2r / float(nx)

lmax=7000
resolution = Lsize*60/float(nx) #arcmin
tfac = np.sqrt((dx*dx)/(nx*nx))


#-----------------------------------------------

def signal_spectrum():

    #CAMB para simular espectro
    
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0.1)
    pars.set_for_lmax(7000, lens_potential_accuracy=1)
    pars.WantTensors = True

    results = camb.get_results(pars)
    
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')

    totCL          = powers['total']
    unlensedCL     = powers['unlensed_scalar']
    unlensed_Tot   = powers['unlensed_total']
    lensed_SC      = powers['lensed_scalar']
    tensorCL       = powers['tensor']
    lens_potential = powers['lens_potential']  

    ls = np.arange(unlensedCL.shape[0])
    factor = ls*(ls+1)/2./np.pi

    cl_TT_unlensed = np.copy(unlensedCL[:,0])
#no quitamos el monopolo y dipolo (lo hacemos despues)
    cl_TT_unlensed[2:] = cl_TT_unlensed[2:] / factor[2:]


    cl_TT_for_map = np.copy(cl_TT_unlensed)

    #cl_TT_for_map[lmax+1:len(cl_TT_unlensed)] = 0.
    
    return cl_TT_for_map



def flat_spectrum(cl_angular): 
    
    lx,ly=np.meshgrid( np.fft.fftfreq( nx, 1/float(nx) ),np.fft.fftfreq( nx, 1/float(nx) ) )
    l = np.sqrt(lx**2 + ly**2) #shape (128,65)
    ell_flat = l.flatten()
    
    ell_flat = ell_flat*18.
   
    ls = np.arange(len(cl_angular))
    inter = scipy.interpolate.interp1d(ls[2:], cl_angular[2:],bounds_error=False,fill_value=np.min(cl_angular[2:]))
    cl_plano = inter(ell_flat)
   
    return ell_flat, cl_plano
    
def flat_spectrum_real(cl_angular): 
    
    lx,ly=np.meshgrid( np.fft.rfftfreq( nx, 1/float(nx) ),np.fft.fftfreq( nx, 1/float(nx) ) )
    l = np.sqrt(lx**2 + ly**2) #shape (128,65)
    ell_flat = l.flatten()
    
    ell_flat = ell_flat*18.
   
    ls = np.arange(len(cl_angular))
    inter = scipy.interpolate.interp1d(ls[2:], cl_angular[2:],bounds_error=False,fill_value=np.min(cl_angular[2:]))
    cl_plano = inter(ell_flat)
   
    return ell_flat, cl_plano    


def spectrum_unique(ell_flat, cl_plane):
    ell_flat_unique, rep = np.unique(ell_flat, return_counts=True)

    cl_flat_unique = np.zeros(ell_flat_unique.shape)

    idx = 0
    for ell_value in ell_flat_unique:
        cl_flat_unique[idx] = cl_plane[np.where(ell_flat == ell_value)][0]
        idx+=1
    return ell_flat_unique, cl_flat_unique


#-------------------------------------------------------------------------------
#bineado
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx   #,array[idx]

def compute_bins(lmin, lmax, Nbins):

    ls = np.arange(lmax+1)

    num_modes = np.zeros(lmax+1)
    cumulative_num_modes = np.zeros(lmax+1)

    bin_edges = np.zeros(Nbins+1)
    bin_edges[0] = lmin
    
    cumulative = 0
    for i in range(lmin,lmax+1):
        num_modes[i] = 2*i +1
        
        cumulative += num_modes[i]
        cumulative_num_modes[i] = cumulative

            
    Num_modes_total = num_modes.sum()
    print("Total number of modes in (l_min,l_max) = ", Num_modes_total)   
    Num_modes_per_bin = Num_modes_total / Nbins
    print("Number of modes in each bin = ", Num_modes_per_bin)

    for i in range(1,Nbins+1):
        
        #Num_modes_per_bin*i #cumulative modes up to bin "i"
        
        bin_edges[i] = find_nearest(cumulative_num_modes, Num_modes_per_bin*i)
    
    bin_edges = np.asarray(bin_edges,int)

    
    return Num_modes_per_bin, cumulative_num_modes, bin_edges


def binning(ell, cl, bins): 
 
    # Start timing
    start_time = time.time()

    bin_indices = np.digitize(ell, bins, right=False)

    # Initialize arrays to store results
    count = np.zeros(len(bins) - 1)
    cl_bin_sum = np.zeros(len(bins) - 1)
    el_med_sum = np.zeros(len(bins) - 1)


    # Calculate sum of ell values and cl values for each bin
    for i in range(1, len(bins)):
        mask = (bin_indices == i)
        count[i - 1] = np.sum(mask)
        cl_bin_sum[i - 1] = np.sum(cl[mask])
        el_med_sum[i - 1] = np.sum(ell[mask] * cl[mask])

    # Calculate the binned results
    el_med = (el_med_sum / cl_bin_sum).astype(int)
    cl_bin = cl_bin_sum / count

    # End timing
    end_time = time.time()


    return el_med, cl_bin, count


def power(mapa):
    tfac = np.sqrt((dx*dx)/(nx*nx))
    fft = np.fft.fft2(mapa[:,:])*tfac
    fft_shape = fft.shape 
    power = np.real(fft*np.conj(fft))
    #power_flat = power.flatten()
    power_reshape = np.reshape(power, [fft_shape[0]*fft_shape[1]]) #coef de fourier mapa
    return power_reshape

def power_real(mapa):
    tfac = np.sqrt((dx*dx)/(nx*nx))
    rfft = np.fft.rfft2(mapa[:,:])*tfac
    rfft_shape = rfft.shape 
    power = np.real(rfft*np.conj(rfft))
    #power_flat = power.flatten()
    power_reshape = np.reshape(power, [rfft_shape[0]*rfft_shape[1]]) #coef de fourier mapa
    return power_reshape, rfft
    

def El_fid(ell_flat, cl_plane, power, bins, cl_bin): 
    #ell_flat, cl_plane = flat_spectrum(cl)

    start_time = time.time()

    bin_indices = np.digitize(ell_flat, bins) - 1

    El_sum = np.zeros(len(bins) - 1)
    El_count = np.zeros(len(bins) - 1)

    for bin_index in range(len(bins) - 1):
        mask = (bin_indices == bin_index)
        El_sum[bin_index] = np.sum(power[mask] / cl_plane[mask])
        El_count[bin_index] = np.sum(mask)

    El = 0.5 * El_sum / cl_bin

    # End timing
    end_time = time.time()

 
    return El   





