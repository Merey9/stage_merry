import sys

sys.path.append("/data/stage_merry/functions")
from pspy import *
import numpy as np
from math import pi
from tqdm import tqdm
from plot_functions import *
from fisher_matrix import *
from likelihood_functions import *
from itertools import combinations_with_replacement, product

date_float = "0606_0552"
spec_keys_pspy = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

# All spectra are stored with conveniant keys:
# ex : T1431E2172 for TE 143GHzhm1x217GHzhm2
# 9(keys) * 9(band combinaitions) * 4(hm combinaitions) = 324 keys
data = {}
for BAND_1, BAND_2 in product(["100", "143", "217"], repeat=2):
    for hm_1, hm_2 in product(["1", "2"], repeat=2):
        lb, current_spectra = so_spectra.read_ps(
            "data/spectra/maison/Dls_%shm%sx%shm%s_%s.dat"
            % (BAND_1, hm_1, BAND_2, hm_2, date_float),
            spectra=spec_keys_pspy,
        )
        for XY in spec_keys_pspy:
            X, Y = XY[0], XY[1]
            data[X + BAND_1 + hm_1 + Y + BAND_2 + hm_2] = (
                current_spectra[XY] * 2 * pi / (lb * (lb + 1))
            )

# theta_i, theta_j, beta = -0.1 * pi / 180, 0.3 * pi / 180, 0.05 * pi / 180

split_i, split_j = "1001", "1002"
splits = ["1001", "1002", "1431", "1432", "2171", "2172"]
alphas = np.full_like(splits, 0.0, dtype=float)
alphas = np.array([-0.28, -0.28, 0.07, 0.07, -0.07, -0.07]) / 180 * pi
beta = 0.5 * pi / 180


A_Clobs_ij, A_Clobs_pq, B_Clfid_ij, B_Clfid_pq, chi2_list = get_chi2(
    lb,
    data,
    splits,
    alphas,
    beta,
    lmin=50,
    lmax=1400,
    binning_file="data/spectra/maison/binning/binning%s.dat" % (date_float),
)
# print(beta * 180 / pi, chi2)
loop_beta = False
if loop_beta:
    from matplotlib import pyplot as plt

    for beta_deg in np.linspace(0, 1, 10, dtype=float):
        chi2 = get_chi2(
            lb,
            data,
            splits,
            alphas,
            beta_deg * pi / 180,
            lmin=50,
            lmax=1000,
            binning_file="data/spectra/maison/binning/binning%s.dat" % (date_float),
        )[-1]
        plt.plot(beta_deg, chi2, marker=".")
    plt.savefig("Planck/Figures/chi2_beta.png")
