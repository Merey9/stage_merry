import sys

sys.path.append("/data/stage_merry/functions")
import healpy as hp
from pspy import *
from matplotlib import pyplot as plt
import numpy as np
from math import pi
import pysm3
from pysm3 import units as u
from tqdm import tqdm
from plot_functions import *
from fisher_matrix import *
from itertools import combinations_with_replacement, product


def fac(ls):
    return ls * (ls + 1) / (2 * pi)


spec_keys_pspy = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

plot_bool = False
comp_bool = False
plot_err_bool = False
plot_err_comp = True

date_float = "0530_0810"

cross_spec_key = "TT"

filename = "Planck/Figures/Planck_all_band_%s_%s_new_fsky" % (cross_spec_key, date_float)
if comp_bool:
    filename += "_comp"
if plot_err_comp:
    filename+= "_errcomp"

filename += ".png"

binning_file = "data/spectra/planck/binning/bin_planck.dat"
bin_lo, bin_hi, l_bin, bin_size = pspy_utils.read_binning_file(binning_file, lmax=3000)

spec_xlims = {
    "TT": (0, 3000),
    "EE": (0, 1500),
    "TE": (0, 1500),
    "ET": (0, 1500),
    "EB": (0, 1200),
    "BB": (0, 1500),
}

spec_ylims = {
    "TT": (0, 7000),
    "EE": (0, 70),
    "TE": (-200, 200),
    "ET": (-200, 200),
    "EB": (-10, 10),
    "BB": (-2, 10),
}

spec_ylims_res = {
    "TT": None,
    "TE": None,
    "EE": None,
    "ET": (-200, 200),
    "EB": (-10, 10),
    "BB": None,
}

# Computed with np.sum(mask.data)/(2048**2 *12)
fskys = {
    "100": [0.6862273216247559, 0.7199982007344564],
    "143": [0.5980335076649984, 0.5186915397644043],
    "217": [0.4959452946980794, 0.4274274508158366],
}

# Temp noise and pol noise
noise_levels = {
    "100": [1.29 * pi / 180, 1.96 * pi / 180],
    "143": [0.55 * pi / 180, 1.17 * pi / 180],
    "217": [0.78 * pi / 180, 1.75 * pi / 180],
}

BAND_1 = "100"
BAND_2 = "143"

BAND_iter = [
    ("100", "100"),
    ("143", "143"),
    ("217", "217"),
    ("100", "143"),
    ("100", "217"),
    ("143", "217"),
]
# BAND_iter = [("100", "100"), ("100", "143"), ("143", "143"), ("100", "217")]

rows, cols = best_subplot_shape(len(BAND_iter))

fig, axs = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6), dpi=300)
axs = np.array(axs).reshape(rows, cols)
fig.subplots_adjust(hspace=0.2, wspace=0.2)  # Adjust the spacing between subplots
plt.tight_layout()

for i, (BAND_1, BAND_2) in tqdm(enumerate(BAND_iter)):
    row = i // cols
    col = i % cols
    axs[row, col].set_title("%sx%s" % (BAND_1, BAND_2))
    Dls_11 = {}
    Dls_22 = {}
    Dls_12 = {}
    for hm_B1, hm_B2 in [('1', '1'), ('1', '2'), ('2', '1'), ('2', '2')]:
        lb, Dls_12["hm%shm%s" % (hm_B1, hm_B2)] = so_spectra.read_ps(
            "data/spectra/maison/Dls_%shm%sx%shm%s_%s.dat" % (BAND_1, hm_B1, BAND_2, hm_B2, date_float),
            spectra=spec_keys_pspy,
        )
        lb, Dls_11["hm%shm%s" % (hm_B1, hm_B2)] = so_spectra.read_ps(
            "data/spectra/maison/Dls_%shm%sx%shm%s_%s.dat" % (BAND_1, hm_B1, BAND_1, hm_B2, date_float),
            spectra=spec_keys_pspy,
        )
        lb, Dls_22["hm%shm%s" % (hm_B1, hm_B2)] = so_spectra.read_ps(
            "data/spectra/maison/Dls_%shm%sx%shm%s_%s.dat" % (BAND_2, hm_B1, BAND_2, hm_B2, date_float),
            spectra=spec_keys_pspy,
        )
    try:
        data_Planck = np.loadtxt(
            "data/spectra/planck/planck_spectrum_%s_%sx%s.dat"
            % (cross_spec_key, BAND_1, BAND_2)
        ).T
    except:
        data_Planck = "erm what the sigma"

    # Compute indices for the binning file
    bin_indices = [i for i, value in enumerate(l_bin) if value in set(lb)]

    Dls_sum = {
        cross_spec_key: np.zeros_like(Dls_12["hm1hm2"][cross_spec_key], dtype=float)
    }

    for key in [cross_spec_key, cross_spec_key[::-1]]:
        Dls_sum[cross_spec_key] += Dls_12["hm1hm2"][key]
        Dls_sum[cross_spec_key] += Dls_12["hm2hm1"][key]

    Dls_sum[cross_spec_key] /= 4

    if plot_err_bool or plot_err_comp:
        spec_index_1 = 0
        spec_index_2 = 0
        if cross_spec_key[0] != "T":
            spec_index_1 = 1
        if cross_spec_key[1] != "T":
            spec_index_2 = 1
        bin_size = [bin_size[i] for i in bin_indices]
        error_dict = get_crossfreq_variance(
            Dls_11,
            Dls_12,
            Dls_22,
            lb,
            bin_size,
            fsky_1=np.sqrt(fskys[BAND_1][spec_index_1] * fskys[BAND_1][spec_index_2]),
            fsky_2=np.sqrt(fskys[BAND_2][spec_index_1] * fskys[BAND_2][spec_index_2]),
        )

    if comp_bool:
        try:
            comp_indices = [
                i for i, value in enumerate(lb) if value in set(data_Planck[0])
            ]
            Dls_comp = [Dls_sum[cross_spec_key][i] for i in comp_indices]
            Dls_res = (
                Dls_comp
                - (
                    data_Planck[1][: len(Dls_comp)]
                    * fac(data_Planck[0][: len(Dls_comp)])
                )
            ) / (data_Planck[2][: len(Dls_comp)] * fac(data_Planck[0][: len(Dls_comp)]))
            axs[row, col].plot(
                data_Planck[0][: len(Dls_comp)],
                Dls_res,
                color="black",
                label="mean relative",
            )
            axs[row, col].set_ylim(spec_ylims_res[cross_spec_key])
        except:
            pass

    if plot_bool:
        for key in [cross_spec_key, cross_spec_key[::-1]]:
            axs[row, col].plot(
                lb,
                Dls_12['hm1hm2'][key],
                alpha=0.5,
                label="%s_%sx%s" % (key, BAND_1, BAND_2),
                linewidth=1,
            )
            axs[row, col].plot(
                lb,
                Dls_11['hm1hm1'][key],
                alpha=0.5,
                label="%s_%sx%s" % (key, BAND_1, BAND_1),
                linewidth=1,
            )
            axs[row, col].plot(
                lb,
                Dls_22['hm1hm1'][key],
                alpha=0.5,
                label="%s_%sx%s" % (key, BAND_2, BAND_2),
                linewidth=1,
            )
        axs[row, col].plot(lb, Dls_sum[cross_spec_key], color="black", label="mean")
        try:
            axs[row, col].errorbar(
                data_Planck[0],
                data_Planck[1] * fac(data_Planck[0]),
                data_Planck[2] * fac(data_Planck[0]),
                color="crimson",
                label="Planck",
                linestyle="None",
                marker=".",
            )
        except:
            pass
        axs[row, col].set_ylim(spec_ylims[cross_spec_key])

        if plot_err_bool:
            bin_size = [bin_size[i] for i in bin_indices]
            axs[row, col].fill_between(
                lb,
                Dls_sum[key] - error_dict[key] / bin_size,
                Dls_sum[key] + error_dict[key] / bin_size,
                alpha=0.3,
                color="grey",
                label="Effective noise",
                zorder=-10,
            )

    if plot_err_comp:
        axs[row, col].plot(
            lb,
            error_dict[key],
            alpha=1,
            color="black",
            label="Effective noise",
        )
        axs[row, col].semilogy()
        try:
            axs[row, col].plot(
                data_Planck[0],
                data_Planck[2] * fac(data_Planck[0]),
                alpha=1,
                color="red",
                label="Planck noise",
            )
        except:
            pass

    axs[row, col].set_xlim(spec_xlims[cross_spec_key])
    axs[row, col].legend()

plt.savefig(filename, dpi=300)
