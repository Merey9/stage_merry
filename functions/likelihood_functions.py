import sys

sys.path.append("/data/stage_merry/functions")
import numpy as np
from simu import *
import copy
from pspy import *
from birefringence_functions import *
from fisher_matrix import *
from numpy import pi, cos, sin
from itertools import *
from tqdm import tqdm

# Formalism of 2006.15982v1


def get_spectra_covariance(
    ls: np.ndarray, data: dict, X: str, Y: str, W: str, Z: str, ignore_keys=['EB']
):
    """Compute :
        Cov(Cls^{X, Y}, Cls^{W, Z}) =
        1/(2l+1) (Cls^{X, W} Cls^{Y, Z}
        + Cls^{X, Z} Cls^{Y, W})

    Args:
        ls (_type_): _description_
        data (_type_): _description_
        X (_type_): _description_
        Y (_type_): _description_
        W (_type_): _description_
        Z (_type_): _description_

    Returns:
        cov: the covariance array
    """
    cov = np.zeros_like(ls, dtype=float)

    if ignore_keys == None:
        cov += (
            1 / (2 * ls + 1) * (data[X + W] * data[Y + Z] + data[X + Z] * data[Y + W])
        )
    else:
        if (
            X[0] + W[0] not in ignore_keys
            and W[0] + X[0] not in ignore_keys
            and Y[0] + Z[0] not in ignore_keys
            and Z[0] + Y[0] not in ignore_keys
        ):
            cov += 1 / (2 * ls + 1) * (data[X + W] * data[Y + Z])
        if (
            X[0] + Z[0] not in ignore_keys
            and Z[0] + X[0] not in ignore_keys
            and Y[0] + W[0] not in ignore_keys
            and W[0] + Y[0] not in ignore_keys
        ):
            cov += 1 / (2 * ls + 1) * (data[X + Z] * data[Y + W])
    return cov


def cross_spectra_variance(ls: np.ndarray, data: dict, spec_list: list):
    """Generate the variance of the sum of
    Cls^AB+Cls^CD+... with
    spec_list=[A,B,C,D,...]

    Args:
        ls (np.ndarray): _description_
        data (dict): _description_
        spec_list (list): _description_
    """
    assert len(spec_list) // 2 == len(spec_list) / 2

    first_spec = spec_list[::2]
    second_spec = spec_list[1::2]
    pairs = list(zip(first_spec, second_spec))

    variance_list = np.zeros_like(ls, dtype=float)
    for first_1, second_1 in zip(first_spec, second_spec):
        for first_2, second_2 in zip(first_spec, second_spec):
            variance_list += get_spectra_covariance(
                ls, data, first_1, second_1, first_2, second_2
            )
    variance_list /= len(pairs) ** 2
    return variance_list


def get_r_matrix(theta_i: float, theta_j: float) -> np.ndarray:
    """Compute \vec{R} from 2006.15982v1

    Args:
        theta_i (float): _description_
        theta_j (float): _description_

    Returns:
        np.ndarray: _description_
    """
    r_matrix = [
        cos(2 * theta_i) * sin(2 * theta_j),
        -sin(2 * theta_i) * cos(2 * theta_j),
    ]
    return np.array(r_matrix)


def get_R_matrix(theta_i: float, theta_j: float) -> np.ndarray:
    """Compute \mathbf{R} from 2006.15982v1

    Args:
        theta_i (float): _description_
        theta_j (float): _description_

    Returns:
        np.ndarray: _description_
    """
    R_matrix = [
        [cos(2 * theta_i) * cos(2 * theta_j), sin(2 * theta_i) * sin(2 * theta_j)],
        [sin(2 * theta_i) * sin(2 * theta_j), cos(2 * theta_i) * cos(2 * theta_j)],
    ]
    return np.array(R_matrix)


def get_A_matrix(theta_i: float, theta_j: float) -> np.ndarray:
    """Compute \mathbf{A} matrix from 2006.15982v1

    Args:
        theta_i (float): _description_
        theta_j (float): _description_

    Returns:
        np.ndarray: _description_
    """
    r_matrix = get_r_matrix(theta_i, theta_j)
    R_matrix = get_R_matrix(theta_i, theta_j)
    rTR1_matrix = -np.dot(r_matrix.T, np.linalg.inv(R_matrix))
    return np.diag(np.append(rTR1_matrix, 1))


def get_B_matrix(theta_i: float, theta_j: float, beta: float) -> np.ndarray:
    """Compute \mathbf{B} matrix from 2006.15982v1

    Args:
        theta_i (float): _description_
        theta_j (float): _description_
        beta (float): _description_

    Returns:
        np.ndarray: _description_
    """
    rTR1R_matrix = np.dot(
        get_r_matrix(theta_i, theta_j),
        np.dot(
            np.linalg.inv(get_R_matrix(theta_i, theta_j)),
            get_R_matrix(theta_i + beta, theta_j + beta),
        ),
    )
    B_matrix = np.diag(get_r_matrix(theta_i + beta, theta_j + beta) - rTR1R_matrix)
    return B_matrix

def get_C_inv_no_alpha(
    ls: np.ndarray,
    data: dict,
    split_i: str,
    split_j: str,
    split_p: str,
    split_q: str,
) -> np.ndarray:
    """_summary_

    Args:
        ls (np.ndarray): _description_
        data (dict): _description_
        split_i (str): _description_
        split_j (str): _description_
        split_p (str): _description_
        split_q (str): _description_
        alpha_i (str): _description_
        alpha_j (str): _description_
        alpha_p (str): _description_
        alpha_q (str): _description_

    Returns:
        np.ndarray: _description_
    """
    fill_values = np.zeros_like(ls, dtype=float)
    cov = np.array(
        [
            [
                get_spectra_covariance(
                    ls, data, "E" + split_i, "E" + split_j, "E" + split_p, "E" + split_q
                ),
                get_spectra_covariance(
                    ls, data, "E" + split_i, "E" + split_j, "B" + split_p, "B" + split_q
                ),
                get_spectra_covariance(
                    ls, data, "E" + split_i, "E" + split_j, "E" + split_p, "B" + split_q
                ),
            ],
            [
                get_spectra_covariance(
                    ls, data, "B" + split_i, "B" + split_j, "E" + split_p, "E" + split_q
                ),
                get_spectra_covariance(
                    ls, data, "B" + split_i, "B" + split_j, "B" + split_p, "B" + split_q
                ),
                get_spectra_covariance(
                    ls, data, "B" + split_i, "B" + split_j, "E" + split_p, "B" + split_q
                ),
            ],
            [
                get_spectra_covariance(
                    ls, data, "E" + split_i, "B" + split_j, "E" + split_p, "E" + split_q
                ),
                get_spectra_covariance(
                    ls, data, "E" + split_i, "B" + split_j, "B" + split_p, "B" + split_q
                ),
                get_spectra_covariance(
                    ls, data, "E" + split_i, "B" + split_j, "E" + split_p, "B" + split_q
                ),
            ],
        ]
    )
    cov = np.transpose(cov, (2, 0, 1))
    C_inv = np.zeros_like(cov, dtype=float)
    for b in range(len(cov)):
        C_inv[b] = np.linalg.inv(cov[b])
        # print(C_inv)
    return C_inv


def get_C_inv(
    ls: np.ndarray,
    data: dict,
    split_i: str,
    split_j: str,
    split_p: str,
    split_q: str,
    alpha_i: str,
    alpha_j: str,
    alpha_p: str,
    alpha_q: str,
) -> np.ndarray:
    """_summary_

    Args:
        ls (np.ndarray): _description_
        data (dict): _description_
        split_i (str): _description_
        split_j (str): _description_
        split_p (str): _description_
        split_q (str): _description_
        alpha_i (str): _description_
        alpha_j (str): _description_
        alpha_p (str): _description_
        alpha_q (str): _description_

    Returns:
        np.ndarray: _description_
    """
    fill_values = np.zeros_like(ls, dtype=float)
    cov = np.array(
        [
            [
                get_spectra_covariance(
                    ls, data, "E" + split_i, "E" + split_j, "E" + split_p, "E" + split_q
                ),
                get_spectra_covariance(
                    ls, data, "E" + split_i, "E" + split_j, "B" + split_p, "B" + split_q
                ),
                get_spectra_covariance(
                    ls, data, "E" + split_i, "E" + split_j, "E" + split_p, "B" + split_q
                ),
            ],
            [
                get_spectra_covariance(
                    ls, data, "B" + split_i, "B" + split_j, "E" + split_p, "E" + split_q
                ),
                get_spectra_covariance(
                    ls, data, "B" + split_i, "B" + split_j, "B" + split_p, "B" + split_q
                ),
                get_spectra_covariance(
                    ls, data, "B" + split_i, "B" + split_j, "E" + split_p, "B" + split_q
                ),
            ],
            [
                get_spectra_covariance(
                    ls, data, "E" + split_i, "B" + split_j, "E" + split_p, "E" + split_q
                ),
                get_spectra_covariance(
                    ls, data, "E" + split_i, "B" + split_j, "B" + split_p, "B" + split_q
                ),
                get_spectra_covariance(
                    ls, data, "E" + split_i, "B" + split_j, "E" + split_p, "B" + split_q
                ),
            ],
        ]
    )
    cov = np.transpose(cov, (2, 0, 1))
    C_inv = np.zeros_like(cov, dtype=float)
    for b in range(len(cov)):
        C_inv[b] = np.linalg.inv(
            np.dot(
                get_A_matrix(alpha_i, alpha_j),
                np.dot(cov[b], get_A_matrix(alpha_p, alpha_q).T)
            )
        )
        # print(C_inv)
    return C_inv


def get_fiduc_Cls(
    lb,
    split_1,
    split_2,
    binning_file="data/spectra/planck/binning/bin_planck.dat",
):
    ls, fiducial_dict = so_spectra.read_ps(
        "data/spectra/fiducial/camb_1.dat", spectra=["TT", "EE", "BB", "TE", "TB", "EB"]
    )
    ls_beam_1, bls_1 = np.loadtxt("data/beam_legacy/bl_pol_legacy_%s.dat" % (split_1)).T
    ls_beam_2, bls_2 = np.loadtxt("data/beam_legacy/bl_pol_legacy_%s.dat" % (split_2)).T
    # for spec in ["EE", "BB"]:
    # fiducial_dict[spec] *= (np.array(bls_1[: len(ls)]) * np.array(bls_2[: len(ls)]))
    l_binned, fiducial_binned = so_spectra.bin_spectra(
        ls,
        fiducial_dict,
        binning_file,
        3000,
        "Cl",
        spectra=["TT", "EE", "BB", "TE", "TB", "EB"],
    )
    bin_indices = [i for i, value in enumerate(lb) if value in set(l_binned)]
    Cls_fiduc = [fiducial_binned["EE"][bin_indices], fiducial_binned["BB"][bin_indices]]
    Cls_fiduc = np.transpose(Cls_fiduc, (1, 0))
    return Cls_fiduc


def get_obs_Cls(lb, data, split_1, split_2):
    Cls_EE = data["E%sE%s" % (split_1, split_2)]
    Cls_BB = data["B%sB%s" % (split_1, split_2)]
    Cls_EB = data["E%sB%s" % (split_1, split_2)]
    return np.transpose(np.array([Cls_EE, Cls_BB, Cls_EB]), (1, 0))

def get_chi2_beta(ls, data, splits, beta, lmin, lmax, binning_file, method='product'):
    # for s1, s2, s3, s4 in tqdm(product(np.arange(len(alphas)), repeat=4)):
    #     if s1 == s2 or s3 == s4 or s1 == s3 or s1 == s4 or s2 == s3 or s2 == s4:
    #         continue

    rT_Cls_ij = np.zeros((len(ls), 1), dtype=float)
    rT_Cls_pq = np.zeros((len(ls), 1), dtype=float)
    model_EB = np.zeros((len(ls), 1), dtype=float)
    observation_EB = np.zeros((len(ls), 1), dtype=float)
    chi2_list = np.zeros_like(ls, dtype=float)
    error_EB = np.zeros_like(ls, dtype=float)

    if method == 'product':
        splits_iter = product(np.arange(len(splits)), repeat=4)
    if method == 'one':
        splits_iter = [tuple(np.arange(len(splits)))]
    else:
        splits_iter = method
    for s1, s2, s3, s4 in tqdm(splits_iter):
        if s1 == s2 or s3 == s4: #  or s1 == s3 or s1 == s4 or s2 == s3 or s2 == s4:
            continue
        C_inv = get_C_inv_no_alpha(
            ls,
            data,
            splits[s1],
            splits[s2],
            splits[s3],
            splits[s4],
        )
        
        observation_ij = get_obs_Cls(ls, data, splits[s1], splits[s2])
        observation_pq = get_obs_Cls(ls, data, splits[s3], splits[s4])

        for b in range(len(ls)):
            if ls[b] < lmin or ls[b] > lmax:
                continue
            # if b != 100:
            #     continue
            model_EB[b] += np.dot(get_r_matrix(beta, beta), observation_ij[b][0:2])
            rT_Cls_ij[b] += np.dot(get_r_matrix(beta, beta), observation_ij[b][0:2]) - observation_ij[b][2]
            rT_Cls_pq[b] += np.dot(get_r_matrix(beta, beta), observation_pq[b][0:2]) - observation_pq[b][2]
            observation_EB[b] += observation_ij[b][2]
            chi2_list[b] += rT_Cls_ij[b] * C_inv[b][2, 2] * rT_Cls_pq[b]
            error_EB[b] += 1 / np.sqrt(abs(C_inv[b][2, 2]))
    return observation_EB, model_EB, error_EB, chi2_list


def get_chi2(ls, data, splits, alphas, beta, lmin, lmax, binning_file):
    # for s1, s2, s3, s4 in tqdm(product(np.arange(len(alphas)), repeat=4)):
    #     if s1 == s2 or s3 == s4 or s1 == s3 or s1 == s4 or s2 == s3 or s2 == s4:
    #         continue

    A_Clobs_ij = np.zeros((len(ls), 3), dtype=float)
    B_Clfid_ij = np.zeros((len(ls), 3), dtype=float)
    A_Clobs_pq = np.zeros((len(ls), 3), dtype=float)
    B_Clfid_pq = np.zeros((len(ls), 3), dtype=float)
    chi2_list = np.zeros_like(ls, dtype=float)

    # for s1, s2, s3, s4 in [
    #     (4, 5, 2, 3),
    #     (4, 5, 3, 2),
    #     (5, 4, 2, 3),
    #     (5, 4, 3, 2),
    #     (4, 2, 5, 3),
    #     (4, 3, 5, 2),
    #     (5, 2, 3, 4),
    #     (5, 3, 2, 4),
    # ]:
    for s1, s2, s3, s4 in tqdm(product(np.arange(len(splits)), repeat=4)):
        if s1 == s2 or s3 == s4 or s1 == s3 or s1 == s4 or s2 == s3 or s2 == s4:
            continue
        C_inv = get_C_inv(
            ls,
            data,
            splits[s1],
            splits[s2],
            splits[s3],
            splits[s4],
            alphas[s1],
            alphas[s2],
            alphas[s3],
            alphas[s4],
        )
        A_ij = get_A_matrix(alphas[s1], alphas[s2])
        B_ij = get_B_matrix(alphas[s1], alphas[s2], beta)
        fiducial_ij = get_fiduc_Cls(
            ls, splits[s1], splits[s2], binning_file=binning_file
        )
        observation_ij = get_obs_Cls(ls, data, splits[s1], splits[s2])

        A_pq = get_A_matrix(alphas[s3], alphas[s4])
        B_pq = get_B_matrix(alphas[s3], alphas[s4], beta)
        fiducial_pq = get_fiduc_Cls(
            ls, splits[s3], splits[s4], binning_file=binning_file
        )
        observation_pq = get_obs_Cls(ls, data, splits[s3], splits[s4])

        for b in range(len(ls)):
            if ls[b] < lmin or ls[b] > lmax:
                continue
            # if b != 100:
            #     continue
            A_Clobs_ij[b] += np.dot(A_ij, observation_ij[b])
            B_Clfid_ij[b] += np.append(np.dot(B_ij, fiducial_ij[b]), 0.0)
            A_Clobs_pq[b] += np.dot(A_pq, observation_pq[b])
            B_Clfid_pq[b] += np.append(np.dot(B_pq, fiducial_pq[b]), 0.0)
            chi2_list[b] += np.dot(
                (A_Clobs_ij[b] - B_Clfid_ij[b]).T,
                np.dot(C_inv[b], (A_Clobs_pq[b] - B_Clfid_pq[b])),
            )
    return A_Clobs_ij, A_Clobs_pq, B_Clfid_ij, B_Clfid_pq, chi2_list
