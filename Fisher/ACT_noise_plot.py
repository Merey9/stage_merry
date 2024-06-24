import sys

sys.path.append("/data/stage_merry/functions")
import numpy as np
from math import radians, pi
from simu import *
from fisher_matrix import *
from plot_functions import *
from noise_functions import *

LMAX = 2000
# print("LMAX = " + str(LMAX))

spectra = "total"

cosmo_params = {
    "ombh2": 0.022,
    "omch2": 0.122,
    "cosmomc_theta": 1.04 / 100,
    "tau": 0.0544,
    "As": 2e-9,
    "ns": 0.965,
    "alpha": 0.31 / 180 * 3.1415,
}

data_cls = mk_ini_spectra(
    cosmo_params,
    spectra,
    LMAX,
    cut_2_l=True,
    wanted_keys=["TT", "EE", "BB", "TE", "TB", "EB"],
)

ls = np.arange(2, LMAX)
fac = ls * (ls + 1)
plt.semilogy(ls, data_cls["EE"] * fac, marker="", label="EE", alpha=0.9)


filenames = ["pa6_f090", "pa5_f090", "pa6_f150", "pa5_f150"]

nls_tot = get_nls_from_dat(file_names=filenames, LMAX=LMAX, eff_bool=True)

plt.semilogy(ls, nls_tot * fac, marker="", label="eff", alpha=0.9, color="black")
plt.legend()
plt.savefig("Figures/ACT_noise_test.png")
