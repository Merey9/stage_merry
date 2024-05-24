import sys

sys.path.append("/data/stage_merry/functions")
import numpy as np
from matplotlib import pyplot as plt
from simu import *
from noise_functions import *
from fisher_matrix import *
import math
import copy

# plt.rcParams['text.usetex'] = True


def plot_spectrum(
    cosmo_params={
        "H0": 67.5,
        "ombh2": 0.022,
        "omch2": 0.122,
        "mnu": 0.06,
        "omk": 0,
        "tau": 0.06,
    },
    spectra="total",
    lmax=1000,
    charact_detec=None,
    keys_to_plot=["TT", "EE", "BB", "TE"],
    var_fac=1,
    filename="Figures/plot_spetrum_test.png",
    eff_noise_bool=False,
):

    ls = np.arange(2, lmax)
    fac = ls * (ls + 1) / (2 * math.pi)
    cls = mk_ini_spectra(
        cosmo_params, spectra, lmax, cut_2_l=True, wanted_keys=keys_to_plot
    )

    nls_dict = {}
    if charact_detec == None:
        nls = {}
        for key in cls.keys():
            nls[key] = np.zeros_like(cls, dtype=float)
        nls_dict[""] = nls
    elif type(charact_detec) == list:
        theta_FWHM, s_T, s_pol = charact_detec
        s = {"TT": s_T, "EE": s_pol, "BB": s_pol, "TE": 0}
        nls = get_nls(s, theta_FWHM, lmax=lmax)
        nls_dict[""] = nls
    elif type(charact_detec) == dict:
        for band in charact_detec.keys():
            theta_FWHM, s_T, s_pol = charact_detec[band]
            s = {"TT": s_T, "EE": s_pol, "BB": s_pol, "TE": 0}
            nls = get_nls(s, theta_FWHM, lmax=lmax)
            nls_dict[band] = nls

    if eff_noise_bool:
        nls_dict = get_effective_noise(nls_dict)

    fig, axs = plt.subplots(1, len(keys_to_plot), figsize=(5 * len(keys_to_plot), 5))
    plt.tight_layout()
    for c, band in enumerate(nls_dict.keys()):

        cnls = copy.deepcopy(cls)
        for key in nls.keys() & cnls.keys():
            cnls[key] += nls_dict[band][key]

        cov_matrix = get_cov_matrix(cnls, data_cls_bool=True, cut_2_l=True)
        sigmas_l = np.sqrt(cov_matrix.diagonal(axis1=1, axis2=2))

        dls = {}
        for key in cls.keys():
            dls[key] = cls[key] * fac

        for i, key in enumerate(keys_to_plot):
            axs[i].fill_between(
                ls,
                dls[key] - sigmas_l[:, i] * fac * var_fac,
                dls[key] + sigmas_l[:, i] * fac * var_fac,
                alpha=0.5,
                color="C" + str(c),
                label=band,
            )
            axs[i].plot(ls, dls[key], color="black")

            axs[i].set_title(key)
            axs[i].set_xlabel("l")
            axs[i].set_xlim(0, lmax)
            if key not in ["TE"]:
                # axs[i].semilogy()
                axs[i].set_ylim(
                    min(value for value in list(dls[key]) if key in keys_to_plot),
                    max(value for value in list(dls[key]) if key in keys_to_plot) * 1.5,
                )
            else:
                axs[i].set_ylim(
                    min(value for value in list(dls[key]) if key in keys_to_plot) * 1.5,
                    max(value for value in list(dls[key]) if key in keys_to_plot) * 1.5,
                )

    axs[0].legend(loc="lower left")
    axs[0].set_ylabel(r"l(l+1)Cl")
    plt.tight_layout()
    plt.savefig(filename)


def plot_params_var(
    fisher_matrix,
    planck_var,
    params,
    lmax,
    filename="Figures/params_results",
    detect_name="Planck",
):

    fisher_var = fisher_to_sigmas(fisher_matrix)

    params_values_for_norm = list(params.values())
    params_values_for_norm = [1 if val == 0 else val for val in params.values()]
    params_values_for_norm = (
        np.maximum(np.array(list(planck_var.values())), fisher_var) * 0.9
    )

    fisher_var_norm = fisher_var / np.array(params_values_for_norm)
    planck_var_norm = np.array(list(planck_var.values())) / np.array(
        params_values_for_norm
    )
    tick_labels = list(params.keys())
    tick_labels.append("")
    tick_positions = np.arange(len(tick_labels))
    zeros = np.zeros_like(tick_positions)
    plt.figure(figsize=(len(params.keys()), 2))
    plt.tight_layout()
    plt.errorbar(
        tick_positions[:-1] - 0.05,
        zeros[:-1],
        fisher_var_norm,
        label="Fisher",
        color="black",
        capsize=2,
        linestyle="none",
    )
    plt.errorbar(
        tick_positions[:-1] + 0.05,
        zeros[:-1],
        planck_var_norm,
        label=detect_name,
        color="red",
        capsize=2,
        linestyle="none",
    )
    plt.xticks(tick_positions, tick_labels)
    plt.ylabel("Arbitrary units")
    plt.title("LMAX=" + str(lmax))
    plt.legend()
    plt.savefig(filename)


def plot_spectra_params(
    cosmo_params={
        "ombh2": 0.022,
        "omch2": 0.122,
        "cosmomc_theta": 1.04 / 100,
        "tau": 0.0544,
        "As": 2e-9,
        "ns": 0.965,
    },
    spectra="total",
    lmax=1000,
    keys_to_plot=["TT", "EE"],
    param_to_vary="ombh2",
    param_facs=[0.8, 0.9, 1, 1.1, 1.2],
    filename="Figures/plot_spetrum_param_variation.png",
    charact_detec=None,
    plot_var_bool=False,
):

    ls = np.arange(2, lmax)
    fac = ls * (ls + 1) / (2 * math.pi)
    fig, axs = plt.subplots(1, len(keys_to_plot), figsize=(5 * len(keys_to_plot), 5))

    cmap = plt.get_cmap("twilight")
    param_facs_norm = np.array(param_facs)
    norm = plt.Normalize(
        min(param_facs_norm) - (max(param_facs_norm) - min(param_facs_norm)) * 0.2,
        max(param_facs_norm) + (max(param_facs_norm) - min(param_facs_norm)) * 0.2,
    )
    colors = [cmap(norm(value)) for value in param_facs_norm]
    for c, param_fac in enumerate(param_facs):
        cosmo_params_fac = cosmo_params.copy()
        cosmo_params_fac[param_to_vary] = param_fac
        cls = mk_ini_spectra(
            cosmo_params_fac, spectra, lmax, cut_2_l=True, wanted_keys=keys_to_plot
        )
        dls = {}
        for key in cls.keys():
            dls[key] = cls[key] * fac
        for i, key in enumerate(keys_to_plot):
            axs[i].plot(
                ls,
                dls[key],
                label=param_to_vary
                + "="
                + str(round(cosmo_params_fac[param_to_vary] * 180 / 3.1415, 5)),
                color=colors[c],
            )

    cls = mk_ini_spectra(
        cosmo_params, spectra, lmax, cut_2_l=True, wanted_keys=keys_to_plot
    )
    dls = {}

    ylims = {}
    for i, key in enumerate(keys_to_plot):
        dls[key] = cls[key] * fac
        axs[i].plot(
            ls,
            dls[key],
            label=param_to_vary + "=" + str(cosmo_params[param_to_vary]),
            color="black",
            linewidth=2,
        )
        axs[i].set_title(key)
        axs[i].set_xlabel("l")
        axs[i].set_xlim(0, lmax)
        ylims[key] = axs[i].get_ylim()

    if plot_var_bool:
        nls_dict = {}
        for band in charact_detec.keys():
            theta_FWHM, s_T, s_pol = charact_detec[band]
            s = {"TT": s_T, "EE": s_pol, "BB": s_pol, "EB": 0}
            nls = get_nls(s, theta_FWHM, lmax=lmax)
            nls_dict[band] = nls
        nls_dict = get_effective_noise(nls_dict)
        cnls = copy.deepcopy(cls)
        for key in nls.keys() & cls.keys():
            cnls[key] += nls_dict["Effective"][key]

        cov_matrix = get_cov_matrix(cnls, data_cls_bool=True, cut_2_l=True)
        sigmas_l = np.sqrt(cov_matrix.diagonal(axis1=1, axis2=2))

        dls = {}
        for key in cls.keys():
            dls[key] = cls[key] * fac

        for i, key in enumerate(keys_to_plot):
            if key == "EB":
                binning = 20
                axs[i].fill_between(
                    bin_array(ls, binning),
                    (bin_array(dls[key] - sigmas_l[:, i] * fac, binning))
                    / np.sqrt(binning),
                    (bin_array(dls[key] + sigmas_l[:, i] * fac, binning))
                    / np.sqrt(binning),
                    alpha=0.3,
                    color="grey",
                    label="Effective noise",
                    zorder=-10,
                )
                axs[i].set_ylim(ylims[key])
            else:
                axs[i].fill_between(
                    ls,
                    dls[key] - sigmas_l[:, i] * fac,
                    dls[key] + sigmas_l[:, i] * fac,
                    alpha=0.3,
                    color="grey",
                    label="Effective noise",
                    zorder=-10,
                )
                axs[i].set_ylim(ylims[key])
    axs[0].legend()
    axs[0].set_ylabel("l(l+1)Cl")

    plt.tight_layout()
    plt.savefig(filename)


def best_subplot_shape(n):
    """
    Given the number of subplots `n`, find the best (rows, cols) shape for the subplots grid.
    The goal is to get a shape as close to a square as possible.
    """
    # Start with the assumption that the best shape is (1, n)
    best_shape = (1, n)
    min_difference = n - 1  # Maximum difference initially

    # Iterate over all possible pairs (rows, cols)
    for rows in range(1, int(math.sqrt(n)) + 1):
        cols = math.ceil(n / rows)
        difference = abs(rows - cols)

        if difference < min_difference:
            min_difference = difference
            best_shape = (rows, cols)

    return best_shape
