import os
import camb
import numpy as np
from matplotlib import pyplot as plt
import healpy as hp
from tqdm import tqdm
import pandas as pd
from Simu.simu import mk_spectra, mk_ini_spectra
from functions.fisher_matrix import *
from Simu.simu import mk_ini_spectra
from math import radians

LMAX = 2500
print("LMAX = " + str(LMAX))

names = ["TT", "EE", "BB", "TE"]

spectra = "total"

cosmo_params = {
    "H0": 67.5,
    "ombh2": 0.022,
    "omch2": 0.122,
    "mnu": 0.06,
    "omk": 0,
    "tau": 0.06,
}

pars = camb.CAMBparams()
pars.set_cosmology(**cosmo_params)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(LMAX, lens_potential_accuracy=0)
data_cls = mk_ini_spectra(pars, spectra, LMAX, cut_2_l=True)
# Characteristic of Planck noise [theta beam (arcmin), Temperature noise (muK.deg), Polarisation noise (muK.deg)] from Planck18 p8
charact_Planck = {
    "100": [9.66, 1.29, 1.96],
    "143": [7.22, 0.55, 1.17],
    "217": [4.90, 0.78, 1.75],
}

theta_FWHM_arcmin, s_T_deg, s_pol_deg = charact_Planck["143"]

s_T = radians(s_T_deg)
s_pol = radians(s_pol_deg)
theta_FWHM = radians(theta_FWHM_arcmin / 60)

s = {"TT": s_T, "EE": s_pol, "BB": s_pol, "TE": 0}

nls = get_nls(s, theta_FWHM, lmax=LMAX)

fisher_matrix = get_fisher_matrix(
    data_cls,
    cosmo_params,
    LMAX,
    data_cls_bool=True,
    cut_2_l=True,
    nls=nls,
    step=0.01,
    plot_noise=None,
)

fisher_matrix_inv = np.linalg.inv(fisher_matrix)


sigma_params = {}
for i, key in enumerate(cosmo_params):
    sigma_params[key] = fisher_to_sigmas(fisher_matrix)[i]
    print(key, cosmo_params[key], round(sigma_params[key], 5))
