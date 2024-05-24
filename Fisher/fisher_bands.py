import os
import camb
import numpy as np
from matplotlib import pyplot as plt
import healpy as hp
from tqdm import tqdm
import pandas as pd
from Simu.simu import mk_spectra, mk_ini_spectra
from functions.fisher_matrix import *
from functions.noise_functions import *
from Simu.simu import mk_ini_spectra
from math import radians, pi

LMAX = 2000
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
    "100": [9.66 / 60 * pi / 180, 1.29 * pi / 180, 1.96 * pi / 180],
    "143": [7.22 / 60 * pi / 180, 0.55 * pi / 180, 1.17 * pi / 180],
    "217": [4.90 / 60 * pi / 180, 0.78 * pi / 180, 1.75 * pi / 180],
}

params_sigmas = {}

for band in charact_Planck.keys():
    theta_FWHM, s_T, s_pol = charact_Planck[band]

    s = {"TT": s_T, "EE": s_pol, "BB": s_pol, "TE": 0}

    nls = get_nls(s, theta_FWHM, lmax=LMAX)
    fisher_matrix = get_fisher_matrix(
        data_cls,
        cosmo_params,
        LMAX,
        data_cls_bool=True,
        cut_2_l=True,
        nls=nls,
        step=0.02,
        plot_noise=None,
    )

    params_sigmas[band] = fisher_to_sigmas(fisher_matrix)


plt.savefig("Figures/cls_nls_all.png")
plt.clf()
for i, key in enumerate(cosmo_params.keys()):
    print(
        key,
        round(cosmo_params[key], 4),
        round(params_sigmas["100"][i], 5),
        round(params_sigmas["143"][i], 5),
        round(params_sigmas["217"][i], 5),
    )
