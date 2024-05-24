import sys 
sys.path.append('/data/stage_merry/functions') 
import numpy as np
from math import radians, pi
from simu import *
from fisher_matrix import *
from plot_functions import *
from noise_functions import *

LMIN = 500
LMAX = 1500
FSKY = 0.25
print("LMAX = " + str(LMAX))

spectra = "total"

# Planck resulsts
cosmo_params = {
    "ombh2": 0.022,
    "omch2": 0.122,
    "H0": 65,
    "tau": 0.0544,
    "As": 2e-9,
    "ns": 0.965,
    "alpha": 0 / 180 * 3.1415,
}

# From arxiv:2304.05203
planck_var = {
    "ombh2": 0.00015,
    "omch2": 0.0012,
    "H0": 3.2,  # ACT
    "tau": 0.012,  # ACT+Planck
    "As": (np.exp(3.058) - np.exp(3.044)) * 1e-10 / 2,
    "ns": 0.0042,
    "alpha": 0.06 / 180 * 3.1415,
}

data_cls = mk_ini_spectra(
    cosmo_params,
    spectra,
    LMAX,
    cut_2_l=True,
    wanted_keys=["TT", "EE", "BB", "TE", "TB", "EB"],
)

filenames = ["pa6_f090", "pa5_f090", "pa6_f150", "pa5_f150"]
nls_tot = get_nls_from_dat(file_names=filenames, LMAX=LMAX, eff_bool=True)
for key in data_cls.keys():
    data_cls[key] += nls_tot

fisher_matrix = get_fisher_matrix(
    data_cls,
    cosmo_params,
    LMAX,
    data_cls_bool=True,
    cut_2_l=True,
    step=0.01,
    plot_noise=None,
    fsky=FSKY,
    dont_sum=True,
)

fisher_matrix = np.sum(fisher_matrix[LMIN:], axis=0)

sigma_params = {}
for i, key in enumerate(cosmo_params):
    sigma_params[key] = fisher_to_sigmas(fisher_matrix)[i]
    if key in ["alpha", "beta"]:
        print(key, cosmo_params[key] * 180 / 3.1415, sigma_params[key] * 180 / 3.1415)
    else:
        print(key, cosmo_params[key], sigma_params[key])

plot_params_var(
    fisher_matrix,
    planck_var,
    cosmo_params,
    filename="Figures/params_results_biref_ACT",
    lmax=LMAX,
    detect_name="ACT",
)
