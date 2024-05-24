from functions.plot_functions import *
from math import pi

param_to_vary = "alpha"

charact_Planck = {
    "100": [9.66 / 60 * pi / 180, 1.29 * pi / 180, 1.96 * pi / 180],
    "143": [7.22 / 60 * pi / 180, 0.55 * pi / 180, 1.17 * pi / 180],
    "217": [4.90 / 60 * pi / 180, 0.78 * pi / 180, 1.75 * pi / 180],
}

cosmo_params = {
    "ombh2": 0.022,
    "omch2": 0.122,
    "cosmomc_theta": 1.04 / 100,
    "tau": 0.0544,
    "As": 2e-9,
    "ns": 0.965,
    "alpha": 0,
}

plot_spectra_params(
    cosmo_params=cosmo_params,
    spectra="total",
    lmax=3000,
    keys_to_plot=["TT", "EE", "BB", "EB"],
    param_to_vary=param_to_vary,
    param_facs=np.array([-0.3, -0.15, 0.15, 0.3]) / 180 * 3.1415,
    filename="Figures/plot_spetrum_param_" + param_to_vary + ".png",
    plot_var_bool=True,
    charact_detec=charact_Planck,
)
