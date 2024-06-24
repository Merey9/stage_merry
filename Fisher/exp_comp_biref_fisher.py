import sys

sys.path.append("/data/stage_merry/functions")
import numpy as np
from math import radians, pi
from simu import *
from fisher_matrix import *
from plot_functions import *
from noise_functions import *
from copy import deepcopy

LMIN = 50
LMAX = 3500
FSKY = 1
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
    wanted_keys=["TT", "EE", "BB", "EB"],
)

# filenames = ["pa6_f090", "pa5_f090", "pa6_f150", "pa5_f150"]
# nls_tot = get_nls_from_dat(file_names=filenames, LMAX=LMAX, eff_bool=True)

# Characteristic of Planck noise [theta beam (arcmin), Temperature noise (muK.deg), Polarisation noise (muK.deg)]
noises = {}
# Planck18
charact_Planck = {
    "100": [9.66 / 60 * pi / 180, 1.29 * pi / 180, 1.96 * pi / 180],
    "143": [7.22 / 60 * pi / 180, 0.55 * pi / 180, 1.17 * pi / 180],
    "217": [4.90 / 60 * pi / 180, 0.78 * pi / 180, 1.75 * pi / 180],
}
noises["Planck"] = get_total_noise(charact_Planck, LMAX)
# Arxiv 2307.01258
charact_ACT = {
    "150": [1.4 / 60 * pi / 180, 20 / 60 * pi / 180, 28 / 60 * pi / 180],
    "220": [1.0 / 60 * pi / 180, 65 / 60 * pi / 180, 65 / 60 * pi / 180],
}
noises["ACT"] = get_total_noise(charact_ACT, LMAX)

charact_SO = {
    "145": [1.4 / 60 * pi / 180, 10 / 60 * pi / 180, 14.1 / 60 * pi / 180],
    "225": [1.0 / 60 * pi / 180, 22 / 60 * pi / 180, 22 * 1.41 / 60 * pi / 180],
    "280": [0.9 / 60 * pi / 180, 54 / 60 * pi / 180, 54 * 1.41 / 60 * pi / 180],
}
noises["SO"] = get_total_noise(charact_SO, LMAX)

charact_LITEBird = {
    "oue": [20 / 60 * pi / 180, 6.56 / 1.41 / 60 * pi / 180, 6.56 / 60 * pi / 180]
}
noises["LITEBird"] = get_total_noise(charact_LITEBird, LMAX)
# print(data_cls)
# print(noises['Planck'])

fskys = {"Planck": 0.7, "ACT": 0.4, "SO": 0.4, "LITEBird": 0.7}

for survey in ["Planck", "ACT", "SO", "LITEBird"]:
    noised_data_cls = deepcopy(data_cls)
    for key in noises[survey].keys():
        noised_data_cls[key] += noises[survey][key]

    fisher_matrix = get_fisher_matrix(
        noised_data_cls,
        cosmo_params,
        LMAX,
        data_cls_bool=True,
        cut_2_l=True,
        step=0.01,
        plot_noise=None,
        fsky=fskys[survey],
        dont_sum=True,
    )

    fisher_matrix = np.sum(fisher_matrix[LMIN:], axis=0)
    # print(fisher_matrix)
    sigma_params = {}
    for i, key in enumerate(cosmo_params):
        sigma_params[key] = fisher_to_sigmas(fisher_matrix)[i]
        # if key in ["alpha", "beta"]:
        #     print(key, cosmo_params[key] * 180 / 3.1415, sigma_params[key] * 180 / 3.1415)
        # else:
        #     print(key, cosmo_params[key], sigma_params[key])
    print("%s : +- %s" % (survey, sigma_params["alpha"] * 180 / 3.1415))

# plot_params_var(
#     fisher_matrix,
#     planck_var,
#     cosmo_params,
#     filename="Fisher/Figures/params_results_biref_ACT",
#     lmax=LMAX,
#     detect_name="ACT",
# )
