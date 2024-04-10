import numpy as np
from Simu.simu import mk_ini_spectra
import camb
from matplotlib import pyplot as plt


def get_cov_matrix(
    data: dict, nls=None, cut_2_l=False, data_cls_bool=False, plot_noise=None, fsky=0.7
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

    assert list(data.keys()) == ["TT", "EE", "BB", "TE"]

    if data_cls_bool:
        cls = data
    else:
        cls = {}
        for name in data.keys():
            cls[name] = data[name].mean().to_numpy()[first_l:-1]

    ls = np.arange(len(cls["TT"])) + first_l
    nu = 2 * ls + 1

    cls_noise = {}
    if nls == None:
        cls_noise = cls
    else:
        assert list(nls.keys()) == ["TT", "EE", "BB", "TE"]
        for key in cls.keys():
            cls_noise[key] = cls[key] + nls[key]

    if plot_noise != None:
        plt.plot(nls[plot_noise] * ls * (ls + 1))
        plt.plot(cls[plot_noise] * ls * (ls + 1))
        plt.plot(cls_noise[plot_noise] * ls * (ls + 1))
        plt.semilogy()
        plt.savefig("Figures/test_nls_" + plot_noise + ".png")

    cov_matrix = np.full(
        (len(data.keys()), len(data.keys()), len(ls)), np.zeros_like(ls, dtype=float)
    )
    for i, XY in enumerate(data.keys()):
        for j, WZ in enumerate(data.keys()):
            if XY + WZ == "TTTT" or XY + WZ == "EEEE" or XY + WZ == "BBBB":
                cov_matrix[i, j] = 2 * cls_noise[XY] ** 2 / nu
            elif XY + WZ == "TTEE" or XY + WZ == "EETT":
                cov_matrix[i, j] = 2 * cls_noise["TE"] ** 2 / nu
            elif (
                XY + WZ == "TTTE"
                or XY + WZ == "TETT"
                or XY + WZ == "TEEE"
                or XY + WZ == "EETE"
            ):
                cov_matrix[i, j] = 2 * cls_noise[XY] * cls_noise[WZ] / nu
            elif XY + WZ == "TETE":
                cov_matrix[i, j] = (
                    cls_noise["TT"] * cls_noise["EE"] + cls_noise["TE"] ** 2
                ) / nu
            else:
                cov_matrix[i, j] = np.zeros_like(cls_noise[XY], dtype=float)

    cov_matrix = np.transpose(cov_matrix, (2, 0, 1)) / fsky
    return cov_matrix


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

    for i, param in enumerate(param_keys):

        cosmo_params_inf = cosmo_params.copy()
        cosmo_params_inf[param] *= 1 - step
        if cosmo_params_inf[param] == 0:
            cosmo_params_inf[param] = -1e-6
        cls_inf = mk_ini_spectra(cosmo_params_inf, spectra, LMAX, cut_2_l=cut_2_l)

        cosmo_params_sup = cosmo_params.copy()
        cosmo_params_sup[param] *= 1 + step
        if cosmo_params_sup[param] == 0:
            cosmo_params_sup[param] = 1e-6
        cls_sup = mk_ini_spectra(cosmo_params_sup, spectra, LMAX, cut_2_l=cut_2_l)

        for j, key in enumerate(cls_keys):
            cls_1dev[i, j] = (cls_sup[key] - cls_inf[key]) / (
                cosmo_params_sup[param] - cosmo_params_inf[param]
            )

    cls_1dev = np.transpose(cls_1dev, (2, 0, 1))
    return cls_1dev


def get_all_fisher_matrices(cls_1dev, covariance_matrix_inv):
    fisher_matrices = np.zeros((len(cls_1dev), 6, 6), dtype=float)
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
        cosmo_params=cosmo_params, LMAX=lmax, cut_2_l=cut_2_l, step=step
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
        s = {"TT": s_T, "EE": s_pol, "BB": s_pol, "TE": 0}
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


def fisher_to_sigmas(fisher_matrix):
    """Compute sigmas of the params from the fisher matrix

    Args:
        fisher_matrix (array): (Nparams, Nparams) matrix

    Returns:
        array: (Nparams,1) array
    """
    fisher_matrix_inv = np.linalg.inv(fisher_matrix)
    return np.sqrt(fisher_matrix_inv.diagonal())
