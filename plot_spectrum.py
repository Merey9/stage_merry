from functions.fisher_matrix import *
from functions.plot_functions import *
from math import pi

charact_Planck = {
    "100": [9.66 / 60 * pi / 180, 1.29 * pi / 180, 1.96 * pi / 180],
    "143": [7.22 / 60 * pi / 180, 0.55 * pi / 180, 1.17 * pi / 180],
    "217": [4.90 / 60 * pi / 180, 0.78 * pi / 180, 1.75 * pi / 180],
}

keys_to_plot = ["TT", "EE", "BB", "TE"]

plot_spectrum(
    lmax=2500,
    charact_detec=charact_Planck,
    keys_to_plot=keys_to_plot,
    var_fac=3,
    filename="Figures/spectrum_variance.png"
)
