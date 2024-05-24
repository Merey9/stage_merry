import sys

sys.path.append("/data/stage_merry/functions")
import camb
import numpy as np
import healpy as hp
from birefringence_functions import *


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
    wanted_keys=["TT", "EE", "BB", "TE"],
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
    angle_params = {"alpha": 0, "beta": 0}
    other_params = {}
    if type(input_pars) == camb.CAMBparams:
        pars = input_pars
    else:
        for key, value in input_pars.items():
            if key in inflation_keys:
                inflation_params[key] = value
            elif key in angle_params.keys():
                angle_params[key] = value
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

    ls = np.arange(first_l, LMAX)
    fac = ls * (ls + 1) / (2 * np.pi)
    fac = [1 if x == 0 else x for x in fac]

    dls = {}
    cls = {}
    for i, key in enumerate(["TT", "EE", "BB", "TE"]):
        dls[key] = powers[spectra][first_l:LMAX, i]
        cls[key] = dls[key] / fac

    cls = biref_cross_spectra(cls, angle_params, wanted_keys=wanted_keys)
    return cls


def mk_spectra(
    input_pars, spectra, LMAX, NSIDE, wanted_keys=["TT", "EE", "BB", "TE", "EB", "TB"]
):
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

    cls = mk_ini_spectra(input_pars, spectra, LMAX=LMAX, wanted_keys=wanted_keys)

    maps = hp.synfast((cls["TT"], cls["TE"], cls["EE"], cls["BB"]), nside=NSIDE)

    map_I = maps[0]
    map_Q = maps[1]
    map_U = maps[2]

    cls_all = hp.anafast((map_I, map_Q, map_U), lmax=LMAX)

    cls = {}
    for i, key in enumerate(wanted_keys):
        cls[key] = cls_all[i]

    return cls
