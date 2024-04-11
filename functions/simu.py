import os
import camb
import numpy as np
from matplotlib import pyplot as plt
import healpy as hp


def mk_ini_spectra(
    input_pars={
        "H0": 67.5,
        "ombh2": 0.022,
        "omch2": 0.122,
        "mnu": 0.06,
        "omk": 0,
        "tau": 0.06,
    },
    spectra="total",
    LMAX=1000,
    cut_2_l=False,
) -> dict:
    """Creates CMB spectra from cosmo params

    Args:
        input_pars (dict, optional): _description_. Defaults to { "H0": 67.5, "ombh2": 0.022, "omch2": 0.122, "mnu": 0.06, "omk": 0, "tau": 0.06, }.
        spectra (str, optional): _description_. Defaults to "total".
        LMAX (int, optional): _description_. Defaults to 1000.
        cut_2_l (bool, optional): If true, starts at l=2. Defaults to False.

    Returns:
        dict: Cls
    """
    inflation_keys = [
        "As",
        "ns",
        "nrun",
        "nrunrun",
        "r",
        "nt",
        "ntrun",
        "pivot_scalar",
        "pivot_tensor",
        "parameterization",
    ]
    inflation_params = {}
    other_params = {}
    if type(input_pars) == camb.CAMBparams:
        pars = input_pars
    else:
        for key, value in input_pars.items():
            if key in inflation_keys:
                inflation_params[key] = value
            else:
                other_params[key] = value
        pars = camb.CAMBparams()
        pars.set_cosmology(**other_params)
        pars.InitPower.set_params(**inflation_params)
        pars.set_for_lmax(LMAX, lens_potential_accuracy=0)

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")

    if cut_2_l:
        first_l = 2
    else:
        first_l = 0

    dls_TT = powers[spectra][first_l:LMAX, 0]
    dls_EE = powers[spectra][first_l:LMAX, 1]
    dls_BB = powers[spectra][first_l:LMAX, 2]
    dls_TE = powers[spectra][first_l:LMAX, 3]
    ls = np.arange(dls_TT.shape[0])
    fac = ls * (ls + 1) / (2 * np.pi)
    fac[0] = 1
    cls_TT = dls_TT / fac
    cls_EE = dls_EE / fac
    cls_BB = dls_BB / fac
    cls_TE = dls_TE / fac

    cls = {"TT": cls_TT, "EE": cls_EE, "BB": cls_BB, "TE": cls_TE}

    return cls


def mk_spectra(input_pars, spectra, LMAX, NSIDE):
    """_summary_

    Args:
        input_pars (class): input parameters from camb.model.CAMBparams()
        spectra (str): total, unlensed_scalar,
                        unlensed_total, lensed_scalar,
                        tensor, lens_potential

    Returns:
        cls_TT, cls_EE, cls_BB, cls_TE (ndarray) :
            cls of each type from l = 0 to l=LMAX from input_pars
    """

    cls = mk_ini_spectra(input_pars, spectra, LMAX=LMAX)

    maps = hp.synfast((cls["TT"], cls["TE"], cls["EE"], cls["BB"]), nside=NSIDE)

    map_I = maps[0]
    map_Q = maps[1]
    map_U = maps[2]

    cls_all = hp.anafast((map_I, map_Q, map_U), lmax=LMAX)

    cls_TT = cls_all[0]
    cls_EE = cls_all[1]
    cls_BB = cls_all[2]
    cls_TE = cls_all[3]

    return cls_TT, cls_EE, cls_BB, cls_TE
