import numpy as np
from math import radians, pi
from functions.simu import *
from functions.fisher_matrix import *
from functions.plot_functions import *
from functions.noise_functions import *

LMAX = 1700
FSKY_ACT = 0.25
FSKY_PLANCK = 0.7
print("LMAX = " + str(LMAX))

spectra = "total"

# Planck resulsts
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
# Characteristic of Planck noise [theta beam (arcmin), Temperature noise (muK.deg), Polarisation noise (muK.deg)] from Planck18 p8
charact_Planck = {
    "100": [9.66 / 60 * pi / 180, 1.29 * pi / 180, 1.96 * pi / 180],
    "143": [7.22 / 60 * pi / 180, 0.55 * pi / 180, 1.17 * pi / 180],
    "217": [4.90 / 60 * pi / 180, 0.78 * pi / 180, 1.75 * pi / 180],
}

nls_Planck = get_total_noise(charact_Planck, lmax=LMAX)

Planck_cls = {}
for key in ["TT", "EE", "BB"]:
    Planck_cls[key] = data_cls[key] + nls_Planck[key]

for key in ["TE", "EB", "TB"]:
    Planck_cls[key] = data_cls[key]

filenames = ["pa6_f090", "pa5_f090", "pa6_f150", "pa5_f150"]
nls_ACT = get_nls_from_dat(file_names=filenames, LMAX=LMAX, eff_bool=True)
ACT_cls = {}
for key in ["TT", "EE", "BB"]:
    ACT_cls[key] = data_cls[key] + nls_ACT

for key in ["TE", "EB", "TB"]:
    ACT_cls[key] = data_cls[key]

cls_list = [Planck_cls, ACT_cls, data_cls]
fsky_list = [FSKY_PLANCK, FSKY_ACT, 1]
color_list = ["C0", "red", "black"]
label_list = ["Planck", "ACT", "Without noise"]

fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)  # Share x-axis
for cls, fsky, color, label in zip(cls_list, fsky_list, color_list, label_list):
    fisher_matrix = get_fisher_matrix(
        cls,
        cosmo_params,
        LMAX,
        data_cls_bool=True,
        cut_2_l=True,
        step=0.01,
        plot_noise=None,
        fsky=fsky,
        dont_sum=True,
    )

    # fisher_matrix = np.sum(fisher_matrix[20:], axis=0)
    # fisher_matrix = fisher_matrix.diagonal()
    first_bin = 200
    bin_size = 50
    bins = np.arange(first_bin, (LMAX - first_bin) // bin_size * bin_size, bin_size)
    fisher_l = np.zeros_like(bins, dtype=float)
    fisher_l_diff = np.zeros_like(bins, dtype=float)
    sigma_l = np.zeros_like(bins, dtype=float)
    # if label=="ACT":
    #    full_sigma = fisher_to_sigmas(np.sum(fisher_matrix[int(first_bin - bin_size / 2):], axis=0))[6]
    #    axs[0].set_ylim(0, 10 * full_sigma)

    for i, b in enumerate(bins):
        fisher_l[i] = float(
            np.sum(
                fisher_matrix[int(first_bin - bin_size / 2) : int(b + bin_size / 2)],
                axis=0,
            )[6, 6]
        )
        fisher_l_diff[i] = float(
            np.sum(
                fisher_matrix[int(b - bin_size / 2) : int(b + bin_size / 2)], axis=0
            )[6, 6]
        )
        sigma_l[i] = fisher_to_sigmas(
            np.sum(
                fisher_matrix[int(first_bin - bin_size / 2) : int(b + bin_size / 2)],
                axis=0,
            )
        )[6]
    #    fisher_l[i] = 1 / sigma_l[i]
    #    fisher_l_diff[i] = 1 / fisher_to_sigmas(
    #        np.sum(
    #            fisher_matrix[int(b - bin_size / 2) : int(b + bin_size / 2)], axis=0
    #        ))[6]

    # Plot sigma
    axs[0].plot(bins, sigma_l, label=label, color=color)
    axs[0].set_title("Sigma alpha")
    axs[0].legend()
    axs[0].semilogy()

    # Plot fisher elements
    axs[1].plot(bins, fisher_l, color=color, alpha=0.7, linestyle="--")
    axs[1].plot(bins, fisher_l_diff, label=label, color=color)
    axs[1].set_title("Fisher Elements")
    axs[1].legend()
    axs[1].semilogy()

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.savefig("Figures/biref_l_dependance_full2.png")
