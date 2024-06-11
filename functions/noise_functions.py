import sys

sys.path.append("/data/stage_merry/functions")
import numpy as np
from simu import *
from matplotlib import pyplot as plt
import copy
from birefringence_functions import *
from math import pi


def get_nls(sensitivities: dict, theta_FWHM: float, lmax: int):
    """Compute noise Nls from detector caracteristics

    Args:
        sensitivities (dict): dictionnary of s(muK.rad) for 'TT', 'EE', 'TE' ...
        theta_FWHM (float): Effective beam FWHM(rad)
        lmax (int): lmax ou quoi

    Returns:
        dict: dictionnary of Nls from l=2 to l=LMAX for each 'TT' ...
    """

    nls = {}
    ls = np.arange(2, lmax)
    for key in sensitivities.keys():
        nls[key] = sensitivities[key] ** 2 * np.exp(
            ls * (ls + 1) * theta_FWHM**2 / (8 * np.log(2))
        )

    return nls


def get_total_noise(charact_detec: dict, lmax=1000):
    nls_dict = {}
    for band in charact_detec.keys():
        theta_FWHM, s_T, s_pol = charact_detec[band]
        s = {"TT": s_T, "EE": s_pol, "BB": s_pol}
        nls = get_nls(s, theta_FWHM, lmax=lmax)
        nls_dict[band] = nls
    total_nls1 = {}
    for band in charact_detec.keys():
        for key in nls.keys():
            total_nls1[key] = np.zeros_like(nls_dict[band][key], dtype=float)
    for band in charact_detec.keys():
        for key in nls.keys():
            total_nls1[key] += 1 / nls_dict[band][key]
    total_nls = {}
    for key in nls.keys():
        total_nls[key] = 1 / total_nls1[key]

    return total_nls


def get_effective_noise(nls: dict):
    nls_copy = copy.deepcopy(nls)
    for band in nls_copy.keys():
        for key in nls_copy[band].keys():
            nls_copy[band][key] = 1 / np.array(nls_copy[band][key])
        nls_dict = {
            "Effective": {
                key: np.sum([nls_copy[band][key] for band in nls_copy.keys()], axis=0)
                for key in set().union(
                    *[nls_copy[band].keys() for band in nls_copy.keys()]
                )
            }
        }
    for band in nls_dict.keys():
        for key in nls_dict[band].keys():
            nls_dict[band][key] = 1 / np.array(nls_dict[band][key])
    return nls_dict


def bin_array(array, binning):
    array = array[: len(array) // 20 * 20]
    x_reshaped = np.reshape(array, (-1, binning))
    return np.mean(x_reshaped, axis=1)


def get_nls_from_dat(
    file_names=["pa6_f090", "pa5_f090", "pa6_f150", "pa5_f150"],
    LMAX=2000,
    prepath="data/noise_ACT/noise_",
    postpath="_beam_deconv.dat",
    eff_bool=True,
):
    ls = np.arange(2, LMAX)
    fac = ls * (ls + 1) / 2 * pi

    nls = {}
    for filename in file_names:
        file_path = prepath + filename + postpath
        data = np.loadtxt(file_path).T
        nls[filename] = {"All": np.interp(ls, data[0], data[1])}
    if eff_bool:
        return get_effective_noise(nls)["Effective"]["All"]
    else:
        return nls


def get_effective_nmodes(win_1, win_2, l_bin, bin_size):
    fsky_1 = np.sum(win_1)
    w2_1 = np.sum(win_1.data**2)
    w4_1 = np.sum(win_1.data**4)
    w2_2 = np.sum(win_2.data**2)
    w4_2 = np.sum(win_2.data**4)
    return (2 * l_bin + 1) * bin_size * w2_1 * w2_2 / np.sqrt(w4_1 * w4_2)
