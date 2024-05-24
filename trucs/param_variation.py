from functions.plot_functions import *

param_to_vary = "omch2"

plot_spectra_params(
    cosmo_params={
        "ombh2": 0.022,
        "omch2": 0.122,
        "cosmomc_theta": 1.04 / 100,
        "tau": 0.0544,
        "As": 2e-9,
        "ns": 0.965,
    },
    spectra="total",
    lmax=2000,
    keys_to_plot=["TT", "EE", "BB", "TE"],
    param_to_vary=param_to_vary,
    param_facs=[0.9, 0.95, 1, 1.05, 1.1],
    filename="Figures/plot_spetrum_param_"+param_to_vary+".png",
)
