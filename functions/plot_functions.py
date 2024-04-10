from fisher_matrix import *
import numpy as np
from matplotlib import pyplot as plt


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
):

    ls = np.arange(2, lmax)
    fac = ls * (ls + 1)
    pars = camb.CAMBparams()
    pars.set_cosmology(**cosmo_params)
    pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    cls = mk_ini_spectra(pars, spectra, lmax, cut_2_l=True)

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
        assert nls.keys() == cls.keys()
        nls_dict[""] = nls
    elif type(charact_detec) == dict:
        for band in charact_detec.keys():
            theta_FWHM, s_T, s_pol = charact_detec[band]
            s = {"TT": s_T, "EE": s_pol, "BB": s_pol, "TE": 0}
            nls = get_nls(s, theta_FWHM, lmax=lmax)
            assert nls.keys() == cls.keys()
            nls_dict[band] = nls

    fig, axs = plt.subplots(1, len(keys_to_plot), figsize=(15, 3))
    for c, band in enumerate(charact_detec.keys()):

        cnls = {}
        for key in cls.keys():
            cnls[key] = cls[key] + nls_dict[band][key]

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
            axs[i].semilogy()
            axs[i].set_ylim(
                min(min(value) for key, value in dls.items() if key in keys_to_plot)
                / 3,
                max(max(value) for key, value in dls.items() if key in keys_to_plot)
                * 3,
            )
            axs[i].set_title(key)
            axs[i].set_xlabel("l")
            axs[i].set_xlim(0, lmax)
    axs[0].legend(loc="lower left")
    axs[0].set_ylabel("l(l+1)Cl")
    plt.tight_layout()
    plt.savefig("Figures/plot_spetrum_test.png")


def plot_params_var(
    fisher_matrix, planck_var, params, filename="Figures/params_results"
):
    fisher_var = fisher_to_sigmas(fisher_matrix)
    fisher_var_norm = fisher_var / np.array(list(params.values()))
    planck_var_norm = np.array(list(planck_var.values())) / np.array(
        list(params.values())
    )
    tick_labels = params.keys()
    tick_positions = np.arange(len(params.keys()))
    zeros = np.zeros_like(tick_positions)
    plt.errorbar(
        tick_positions - 0.05,
        zeros,
        fisher_var_norm,
        label="Fisher results",
        color="black",
        capsize=2,
        alpha=0.7,
        linestyle="none",
    )
    plt.errorbar(
        tick_positions + 0.05,
        zeros,
        planck_var_norm,
        label="Planck results",
        color="red",
        capsize=2,
        alpha=0.7,
        linestyle="none",
    )
    plt.xticks(tick_positions, tick_labels)
    plt.ylabel("sigma(param)/param")
    plt.legend()
    plt.savefig(filename)
