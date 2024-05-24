import sys 
sys.path.append('/data/stage_merry/functions')
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

date_float = "0524_0907"

cross_spec_key = 'TE'

BAND_1 = "100"
BAND_2 = "143"

BAND_iter = [("100", "100"), ("143", "143"), ("217", "217"), ("100", "143"), ("100", "217"), ("143", "217")]
rows, cols = best_subplot_shape(len(BAND_iter))

fig, axs = plt.subplots(rows, cols, figsize=(cols*6, rows*6), dpi=300)
axs = np.array(axs).reshape(rows, cols)
fig.subplots_adjust(hspace=0.2, wspace=0.2)  # Adjust the spacing between subplots
plt.tight_layout()

for i, (BAND_1, BAND_2) in tqdm(enumerate(BAND_iter)):
    row = i // cols
    col = i % cols
    axs[row, col].set_title("%sx%s" % (BAND_1, BAND_2))
    
    lb, Dls_12 = so_spectra.read_ps("data/spectra/maison/Dls_%sx%s_%s.dat" % (BAND_1, BAND_2, date_float), spectra=spec_keys_pspy)
    ls_21, Dls_21 = so_spectra.read_ps("data/spectra/maison/Dls_%sx%s_%s.dat" % (BAND_2, BAND_1, date_float), spectra=spec_keys_pspy)

    data_Planck = np.loadtxt("data/spectra/planck/planck_spectrum_%s_%sx%s.dat" % (cross_spec_key, BAND_1, BAND_2)).T

    assert lb.all() == ls_21.all()

    Dls_sum = {cross_spec_key: np.zeros_like(Dls_12[cross_spec_key], dtype=float)}

    for key in [cross_spec_key, cross_spec_key[::-1]]:
        Dls_sum[cross_spec_key] += Dls_12[key]
        Dls_sum[cross_spec_key] += Dls_21[key]

    Dls_sum[cross_spec_key] /= 4


    for key in [cross_spec_key, cross_spec_key[::-1]]:
        axs[row, col].plot(lb, Dls_12[key], alpha=0.5, label="%s_%sx%s" % (key, BAND_1, BAND_2), linewidth=1)
        if BAND_1 != BAND_2:
            axs[row, col].plot(lb, Dls_21[key], alpha=0.5, label="%s_%sx%s" % (key, BAND_2, BAND_1), linewidth=1)
    axs[row, col].errorbar(data_Planck[0], data_Planck[1]*fac(data_Planck[0]), data_Planck[2]*fac(data_Planck[0]), color='crimson', label="Planck", linestyle='None', marker='.')
    axs[row, col].plot(lb, Dls_sum[cross_spec_key], color='black', label="mean")
    axs[row, col].set_ylim(-200,200)
    axs[row, col].set_xlim(0, 1200)
    axs[row, col].legend()
plt.savefig("Planck/Figures/Planck_all_band_%s_test.png" % (cross_spec_key), dpi=300)

