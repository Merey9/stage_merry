import sys
sys.path.append("/data/stage_merry/functions")
import numpy as np
from simu import *
from matplotlib import pyplot as plt
import copy
from birefringence_functions import *


list_keys = ["TT", "EE", "BB", "TE", "TB", "EB"]


def get_cov_matrix(
    data: dict, nls=None, cut_2_l=False, data_cls_bool=True, plot_noise=None, fsky=1, ls=None
) -> np.ndarray:
    """Compute the covariance matrix from Cls and Nls
    Args:
        data (dict): Dict of cls or dict of lists of Cls to mean
        nls (_type_, optional): Noise spectrum. Defaults to None.
        cut_2_l (bool, optional): _description_. Defaults to False.
        data_cls_bool (bool, optional): If False, mean over data rows
        If True, has to be sigle row Cls. Defaults to False.
        plot_noise (str, optional): Plot Cls and Nls of plot_noise='TT', 'TE' ... Defaults to None.

    Returns:
        _type_: Covariance matrices (LMAX-2, Ncls, Ncls)
    """

    if cut_2_l:
        first_l = 2
    else:
        first_l = 0

    cls = data.copy()
    if ls == None:
        ls = np.arange(len(cls[list(cls.keys())[0]])) + first_l
    nu = 2 * ls + 1

    for missing_key in [key for key in list_keys if key not in cls.keys()]:
        cls[missing_key] = np.zeros_like(cls[list(cls.keys())[0]], dtype=float)
    cov_matrix = np.full(
        (len(data.keys()), len(data.keys()), len(ls)), np.zeros_like(ls, dtype=float)
    )
    for i, XY in enumerate(data.keys()):
        for j, WZ in enumerate(data.keys()):
            if XY + WZ == "TTTT" or XY + WZ == "EEEE" or XY + WZ == "BBBB":
                cov_matrix[i, j] = 2 * cls[XY] ** 2 / nu
            elif XY + WZ == "TTEE" or XY + WZ == "EETT":
                cov_matrix[i, j] = 2 * cls["TE"] ** 2 / nu
            elif (
                XY + WZ == "TTTE"
                or XY + WZ == "TETT"
                or XY + WZ == "TEEE"
                or XY + WZ == "EETE"
            ):
                cov_matrix[i, j] = 2 * cls[XY] * cls[WZ] / nu
            elif XY + WZ == "TETE":
                cov_matrix[i, j] = (cls["TT"] * cls["EE"] + cls["TE"] ** 2) / nu
            elif XY + WZ == "TBTB":
                cov_matrix[i, j] = (cls["TT"] * cls["BB"] + cls["TB"] ** 2) / nu
            elif XY + WZ == "EBEB":
                cov_matrix[i, j] = (cls["EE"] * cls["BB"] + cls["EB"] ** 2) / nu
            elif XY + WZ == "TTBB" or XY + WZ == "BBTT":
                cov_matrix[i, j] = 2 * cls["TB"] ** 2 / nu
            elif XY + WZ == "BBTE" or XY + WZ == "TEBB":
                cov_matrix[i, j] = 2 * cls["EB"] * cls["TB"] / nu
            elif XY + WZ == "BBTB" or XY + WZ == "TBBB":
                cov_matrix[i, j] = 2 * cls["BB"] * cls["TB"] / nu
            elif XY + WZ == "BBEB" or XY + WZ == "EBBB":
                cov_matrix[i, j] = 2 * cls["BB"] * cls["EB"] / nu
            elif XY + WZ == "BBEE" or XY + WZ == "EEBB":
                cov_matrix[i, j] = 2 * cls["EB"] ** 2 / nu
            elif XY + WZ == "EBEE" or XY + WZ == "EEEB":
                cov_matrix[i, j] = 2 * cls["EE"] * cls["EB"] / nu
            elif XY + WZ == "TBEE" or XY + WZ == "EETB":
                cov_matrix[i, j] = 2 * cls["TE"] * cls["EB"] / nu
            elif XY + WZ == "TBTT" or XY + WZ == "TTTB":
                cov_matrix[i, j] = 2 * cls["TT"] * cls["TB"] / nu
            elif XY + WZ == "EBTT" or XY + WZ == "TTEB":
                cov_matrix[i, j] = 2 * cls["TE"] * cls["TB"] / nu
            elif XY + WZ == "TETB" or XY + WZ == "TBTE":
                cov_matrix[i, j] = (cls["TT"] * cls["EB"] + cls["TE"] * cls["TB"]) / nu
            elif XY + WZ == "TBEB" or XY + WZ == "EBTB":
                cov_matrix[i, j] = (cls["TE"] * cls["BB"] + cls["EB"] * cls["TB"]) / nu
            elif XY + WZ == "TEEB" or XY + WZ == "EBTE":
                cov_matrix[i, j] = (cls["TE"] * cls["EB"] + cls["EE"] * cls["TB"]) / nu
            else:
                cov_matrix[i, j] = np.zeros_like(cls[XY], dtype=float)
    cov_matrix = np.transpose(cov_matrix, (2, 0, 1)) / fsky
    return cov_matrix

def get_cosmic_variance(data, ls):
    cov_matrix = get_cov_matrix(data, ls)
    error_array = np.sqrt(cov_matrix.diagonal(axis1=1, axis2=2)).T
    print(error_array)
    error_dict = {
        key : error_array[i]
        for i, key in enumerate(data.keys())
    }
    return error_dict


def get_Cls_1dev(
    cosmo_params: dict,
    LMAX: int,
    spectra="total",
    step=0.02,
    cls_keys=["TT", "EE", "BB", "TE"],
    cut_2_l=False,
) -> list:
    """Gets first derivatives of Cls from a fiducial model

    Args:
        cosmo_params (dict): Fiducial model params{'H0': 67.5,
                'ombh2': 0.022,
                'omch2': 0.122,
                'mnu': 0.06,
                'omk': 0,
                'tau': 0.06}
        LMAX (int): _description_
        spectra (str, optional): which spectrum (total, lensed scalar, unlensed...). Defaults to 'total'.
        step (float, optional): Step for the derivative calculation. Defaults to 0.02.
        cls_keys (list, optional): _description_. Defaults to ['TT', 'EE', 'BB', 'TE'].

    Returns:
        list: First derivative matrices (LMAX-2, Nparams, Ncls)
    """

    if cut_2_l:
        numb_l = LMAX - 2
    else:
        numb_l = LMAX

    param_keys = cosmo_params.keys()
    cls_1dev = np.full(
        (len(param_keys), len(cls_keys), numb_l), np.zeros(numb_l, dtype=float)
    )

    angle_keys = ["alpha", "beta"]
    angle_params = {}
    other_params = {}

    for key, value in cosmo_params.items():
        if key in angle_keys:
            angle_params[key] = value
        else:
            other_params[key] = value

    #    for i, param in enumerate(angle_params.keys()):
    #        cls_1dev_angle = cls_1dev_alpha(mk_ini_spectra(
    #            cosmo_params, spectra, LMAX, cut_2_l=cut_2_l, wanted_keys=cls_keys
    #        ), angle_params[param])
    #        for j, key in enumerate(cls_keys):
    #            cls_1dev[i, j] = cls_1dev_angle[key]

    for i, param in enumerate(cosmo_params.keys()):
        if param in angle_keys:
            cls_1dev_angle = cls_1dev_alpha(
                mk_ini_spectra(
                    cosmo_params, spectra, LMAX, cut_2_l=cut_2_l, wanted_keys=cls_keys
                ),
                angle_params[param],
            )
            for j, key in enumerate(cls_keys):
                cls_1dev[i, j] = cls_1dev_angle[key]
        else:
            cosmo_params_inf = cosmo_params.copy()
            cosmo_params_inf[param] *= 1 - step
            if cosmo_params_inf[param] == 0:
                cosmo_params_inf[param] = -1e-6
            cls_inf = mk_ini_spectra(
                cosmo_params_inf, spectra, LMAX, cut_2_l=cut_2_l, wanted_keys=cls_keys
            )

            cosmo_params_sup = cosmo_params.copy()
            cosmo_params_sup[param] *= 1 + step
            if cosmo_params_sup[param] == 0:
                cosmo_params_sup[param] = 1e-6
            cls_sup = mk_ini_spectra(
                cosmo_params_sup, spectra, LMAX, cut_2_l=cut_2_l, wanted_keys=cls_keys
            )

            for j, key in enumerate(cls_keys):
                cls_1dev[i, j] = (cls_sup[key] - cls_inf[key]) / (
                    cosmo_params_sup[param] - cosmo_params_inf[param]
                )
    cls_1dev = np.transpose(cls_1dev, (2, 0, 1))
    return cls_1dev


def get_all_fisher_matrices(cls_1dev, covariance_matrix_inv):
    fisher_matrices = np.zeros(
        (len(cls_1dev), cls_1dev.shape[1], cls_1dev.shape[1]), dtype=float
    )
    for i in range(len(cls_1dev)):
        fisher_matrices[i] = np.dot(
            cls_1dev[i], np.dot(covariance_matrix_inv[i], cls_1dev[i].T)
        )
    return fisher_matrices


def get_fisher_matrix(
    data: dict,
    cosmo_params: dict,
    lmax: int,
    data_cls_bool=False,
    cut_2_l=False,
    nls=None,
    step=0.02,
    plot_noise=None,
    dont_sum=False,
    fsky=0.7,
):
    """Compute Fisher matrix for given cosmo parameters

    Args:
        data (dict): dict of cls or list of cls to mean (see data_cls_bool)
        cosmo_params (dict): Dict of fiducial cosmo parameters
        lmax (int): lmax
        data_cls_bool (bool, optional): if false, mean on data rows. If true, data has to be single row cls
                        Defaults to False.
        cut_2_l (bool, optional): If true, begin at l=2. Defaults to False.
        nls (dict, optional): Noise spectrum, same keys as data. Defaults to None.
        step (float, optional): Step for derivative computation. Defaults to 0.02.

    Returns:
        array: fisher matrix (n_params, n_params)
    """

    cls_1dev = get_Cls_1dev(
        cosmo_params=cosmo_params,
        LMAX=lmax,
        cut_2_l=cut_2_l,
        step=step,
        cls_keys=data.keys(),
    )

    covariance_matrix = get_cov_matrix(
        data,
        cut_2_l=cut_2_l,
        data_cls_bool=data_cls_bool,
        nls=nls,
        plot_noise=plot_noise,
        fsky=fsky,
    )
    covariance_matrix_inv = np.linalg.inv(covariance_matrix)

    fisher_matrices = get_all_fisher_matrices(cls_1dev, covariance_matrix_inv)

    if dont_sum:
        return fisher_matrices
    else:
        fisher_matrix = np.sum(fisher_matrices, axis=0)
        return fisher_matrix


def fisher_to_sigmas(fisher_matrix):
    """Compute sigmas of the params from the fisher matrix

    Args:
        fisher_matrix (array): (Nparams, Nparams) matrix

    Returns:
        array: (Nparams,1) array
    """
    fisher_matrix_inv = np.linalg.inv(fisher_matrix)
    return np.sqrt(fisher_matrix_inv.diagonal())


def bin_array(array, binning):
    array = array[: len(array) // binning * binning]
    x_reshaped = np.reshape(array, (-1, binning))
    return np.mean(x_reshaped, axis=1)
