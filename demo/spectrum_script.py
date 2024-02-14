import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import os

import functions_spectrum

from read_input import load_config

path = os.path.dirname(__file__)

if len(sys.argv) != 2:
    print("Usage: python your_script.py input_file")
    sys.exit(1)

input_file = sys.argv[1]

#Dictionary with the parameters of the input file:
input_data = load_config(input_file)

#---------------------------------------------------------    
filtered_map = np.load(path + "/result/" + input_data["name_result_cnn"])
fisher = np.load(path + "/fisher/" + input_data["name_file_fisher"])
bl = np.load(path + "/fisher/" + input_data["name_file_bl"])

lmin = int(input_data["lmin"])
lmax =  int(input_data["lmax"])
Nbins = int(input_data["nbins"])

#---------------------------------------------------------

cl_ang = functions_spectrum.signal_spectrum()
cl_true = np.load(path + "/data/" +'true_spectrum.npy')
ell_flat, cl_plane_fid = functions_spectrum.flat_spectrum(cl_ang)
ell_flat, cl_plane_true = functions_spectrum.flat_spectrum(cl_true)

_,_, bin_edges = functions_spectrum.compute_bins(lmin,lmax,Nbins)
el_bin_fid, cl_bin_fid, count = functions_spectrum.binning(ell_flat, cl_plane_fid, bin_edges)
el_bin_true, cl_bin_true, count = functions_spectrum.binning(ell_flat, cl_plane_true, bin_edges)

prediction = filtered_map[0,:,:,0]
power_true = functions_spectrum.power(prediction)
El_true = functions_spectrum.El_fid(ell_flat, cl_plane_fid, power_true, bin_edges, cl_bin_fid) 

fisher_inverse = np.linalg.inv(fisher)
nuevo = El_true-bl
nuevo = np.reshape(nuevo, (25,1))
correction_shape = np.dot(fisher_inverse, nuevo)
correction=np.reshape(correction_shape, (25))

estimation = cl_bin_fid + correction

plt.plot(el_bin_true, cl_bin_fid, lw=2, alpha=1, color='purple', label='Fiducial power spectrum')
plt.plot(el_bin_true, cl_bin_true, lw=2, alpha=1, color='gray', label='True power spectrum')
plt.plot(el_bin_true, estimation, color='crimson', marker='o', ms=4, label='Estimation')

plt.yscale('log')
plt.legend(fontsize=12, markerscale=2)
plt.xlabel(r'$\ell$', fontsize=18)
plt.ylabel(r'$C_\ell$', fontsize=18)
plt.xlim(100,3900)
plt.ylim(1e-6, 0.2)

plt.savefig(path + "/result/" +"power_estimation.png")
plt.show()
